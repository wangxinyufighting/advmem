"""少量基础设施：原子写盘、配置、原始模型输出、精确token预算。"""
from __future__ import annotations

import hashlib
import json
import re
import os
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")
    tmp.replace(path)


class Progress:
    """Minimal dependency-free progress bar.

    On a TTY the bar redraws in place; when output is redirected it emits one
    line per update so that ``tail -f`` shows real-time progress without
    carriage-return noise.  It never raises and never reads stdin.
    """

    def __init__(self, total, label="", stream=None, width=28):
        self.total = max(0, int(total))
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.width = width
        self.done = 0
        self._closed = False

    def update(self, suffix="", step=1):
        self.done = min(self.total, self.done + step)
        self._render(suffix)

    def _render(self, suffix):
        total = self.total
        frac = (self.done / total) if total else 1.0
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"{self.label} [{bar}] {self.done}/{total} {frac * 100:5.1f}%"
        if suffix:
            line += f" {suffix}"
        try:
            tty = self.stream.isatty()
        except (AttributeError, ValueError):
            tty = False
        self.stream.write(("\r" + line + "\x1b[K") if tty else (line + "\n"))
        self.stream.flush()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.stream.isatty():
                self.stream.write("\n")
                self.stream.flush()
        except (AttributeError, ValueError):
            pass


class InvalidAction(ValueError):
    """模型输出无效：有明确负奖励。网络异常绝不能转换为此类型。"""


class Unknown(RuntimeError):
    """环境未完成：停止或跳过整组，不产生负奖励。"""


def parse(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise InvalidAction("Output must be a JSON object without fences") from exc
    if not isinstance(value, dict):
        raise InvalidAction("Output must be a JSON object")
    return value


def parse_json_object(text):
    """Parse a model response while preserving strict action parsing elsewhere.

    Judge/reader APIs occasionally wrap an otherwise valid object in a Markdown
    fence or a short explanation.  Builder and attacker actions continue to use
    ``parse`` above so that such wrappers remain illegal training outputs.
    """
    if not isinstance(text, str):
        raise InvalidAction("Model response is not text")
    raw = text.lstrip("\ufeff").strip()
    if not raw:
        raise InvalidAction("Model response is empty")
    if "[TRUNCATED_OUTPUT]" in raw:
        raise InvalidAction("Model response was truncated")

    candidates = []
    if re.search(r"</think>\s*", raw, flags=re.IGNORECASE):
        candidates.append(re.split(r"</think>\s*", raw, flags=re.IGNORECASE)[-1])
    fences = list(re.finditer(r"```(?:json)?\s*\n?(.*?)```", raw,
                              flags=re.IGNORECASE | re.DOTALL))
    if len(fences) == 1:
        candidates.append(fences[0].group(1).strip())
    candidates.append(raw)
    decoder = json.JSONDecoder()

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            value = None
        if isinstance(value, dict):
            return value
        if value is not None:
            continue

        # Allow one object surrounded by prose, but reject ambiguous output
        # containing two independent top-level objects.
        objects = []
        for start, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, end = decoder.raw_decode(candidate, start)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                objects.append((start, end, value))
        maximal = [item for item in objects if not any(
            other[0] <= item[0] and item[1] <= other[1] and other[:2] != item[:2]
            for other in objects)]
        if len(maximal) == 1:
            return maximal[0][2]
    raise InvalidAction("Model response does not contain one JSON object")


@lru_cache(maxsize=8)
def implementation_hash(project):
    sources = {"admem/" + p.name: p for p in Path(__file__).parent.glob("*.py")}
    sources.update({"legacy/" + n: Path(project) / n for n in ["memory.py", "packs.py", "retrieve.py", "llm.py"]})
    return digest({name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in sources.items() if p.is_file()})


@dataclass
class Config:
    project: str = "/root/autodl-tmp/advmem"
    cache: str = ".cache/admem"
    tokenizer: str = "/root/autodl-tmp/model/Qwen3-4B-Instruct-2507"
    tokenizer_revision: str | None = None
    hint_mode: str = "target_type"  # 用户要求的基线；另有 hidden 对照。
    embedding: str = "local"
    embed_model: str | None = None
    reranker: str | None = None
    reader_k: int = 10
    reader_tokens: int = 12000
    builder_input_tokens: int = 8192
    attacker_input_tokens: int = 16384
    judge_input_tokens: int = 30000
    window_tokens: int = 1800
    old_entries: int = 20
    entry_tokens: int = 300
    timeline_tokens: int = 300
    max_ops: int = 16
    neighbors: int = 8
    pack_chars: int = 48000
    questions_per_pack: int = 4
    sweeps: int = 2
    refine_rounds: int = 1
    replay_questions: int = 12
    size_weight: float = 0.2
    gate_mode: str = "full"
    defect_mode: str = "strict"  # strict 是v1；direct另作诊断性扩展。
    fallback: bool = True
    seed: int = 0
    # attacker：compact只给非种子round的user消息；type_filter按种子内容预筛可出题型。
    attacker_view: str = "full"
    attacker_type_filter: bool = False
    seed_min_personal: int = 1
    attacker_memory_entries: int = 20
    attacker_context_items: int = 24
    # 防reward hacking：反复引用同一round按1/(1+平均已用次数)衰减；非SSA题证据无一人称个人陈述时降权。
    evidence_novelty: bool = True
    impersonal_weight: float = 0.25

    @classmethod
    def load(cls, path):
        c = cls(**read(path)) if path else cls()
        if c.hint_mode not in {"target_type", "hidden"} or c.defect_mode not in {"strict", "direct"}:
            raise ValueError("Unknown hint_mode/defect_mode")
        if c.embedding not in {"none", "local", "api"} or c.gate_mode not in {"basic", "full"}:
            raise ValueError("Unknown embedding/gate_mode")
        if c.attacker_view not in {"full", "compact"}:
            raise ValueError("Unknown attacker_view")
        if (min(c.seed_min_personal, c.attacker_memory_entries, c.attacker_context_items) < 0
                or not 0 <= c.impersonal_weight <= 1):
            raise ValueError("Invalid attacker settings")
        numeric = [c.reader_k, c.reader_tokens, c.window_tokens, c.old_entries, c.max_ops,
                   c.builder_input_tokens, c.attacker_input_tokens, c.judge_input_tokens,
                   c.entry_tokens, c.timeline_tokens, c.questions_per_pack, c.sweeps]
        if min(numeric) < 1 or c.refine_rounds < 0 or not 0 <= c.size_weight <= 1:
            raise ValueError("Invalid budgets")
        return c

    def fingerprint(self):
        value = asdict(self)
        value.pop("cache")
        value.pop("project")  # 挪到GPU机器不改变语义；源码仍由implementation_hash验证。
        return digest([value, implementation_hash(self.project)])


def connect(project):
    """不猜旧项目路径；仅依赖之前已发布的四个核心模块。"""
    path = Path(project).resolve()
    print(path)
    required = ["memory.py", "packs.py", "retrieve.py", "llm.py"]
    missing = [p for p in required if not (path / p).is_file()]
    if missing:
        raise RuntimeError(f"--project / config.project 必须指向原full_memory_lab，缺少 {missing}")
    sys.path.insert(0, str(path))
    import memory, packs, retrieve, llm
    for module in [memory, packs, retrieve, llm]:
        if Path(module.__file__).resolve().parent != path:
            raise RuntimeError(f"导入了错误的同名模块：{module.__file__}")
    return memory, packs, retrieve, llm


class TokenBudget:
    def __init__(self, model, revision=None):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        # TokenBudget counts complete prompts; it never truncates them. Some
        # locally exported tokenizers carry a stale 256-token model_max_length
        # even though the configured Builder/Reader budgets are much larger.
        # Raise only the tokenizer-side warning threshold; the actual model
        # service remains responsible for enforcing its context window.
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", 0)
        if not isinstance(tokenizer_limit, (int, float)) or tokenizer_limit < 10**9:
            self.tokenizer.model_max_length = 10**9

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def prompt_count(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def split(self, text, limit):
        """按字符边界切成token受限块，拼回去逐字相同；不decode token造成原文变化。"""
        start = 0
        while start < len(text):
            lo, hi = start + 1, len(text)
            best = start
            while lo <= hi:
                mid = (lo + hi) // 2
                if self.count(text[start:mid]) <= limit:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if best == start:
                raise ValueError("Token budget cannot fit one character")
            yield start, best, text[start:best]
            start = best


def messages(system, payload):
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


class Remote:
    """复用旧Client.post缓存/退避，保留raw输出供SFT/奖励，区分格式错与调用错。"""
    def __init__(self, role, core, cache):
        self.role = role
        self.client = core.from_env(role, cache)
        self.tag = digest([role, self.client.model, self.client.base_url])
        self.retry_dir = Path(cache) / "judge_retries"

    def complete(self, prompt, nonce, temperature=0.0, max_tokens=None,
                 output_tokens=None):
        """Complete one request, optionally overriding its output-token budget.

        ``output_tokens`` is used only by the JSON parser's bounded retry.  It
        is applied after role-specific ``*_EXTRA_BODY`` overrides so that
        services using ``max_completion_tokens`` (rather than the legacy
        ``max_tokens``) receive the larger retry budget as well.  The initial
        request remains byte-for-byte compatible with the configured body.
        """
        if max_tokens is None:
            max_tokens = int(os.getenv(f"{self.role}_MAX_TOKENS",
                                       os.getenv("LLM_MAX_TOKENS", "4096")))
        body = {"model": self.client.model, "messages": prompt, "temperature": temperature,
                "max_tokens": max_tokens}
        
        if os.getenv("LLM_JSON_MODE", "1") == "1":
            body["response_format"] = {"type": "json_object"}
        extras = os.getenv(self.role + "_EXTRA_BODY", os.getenv("LLM_EXTRA_BODY", "{}"))
        for key, value in json.loads(extras).items():
            if value is None:
                body.pop(key, None)
            else:
                body[key] = value
        if output_tokens is not None:
            # Keep the caller's chosen parameter spelling.  A few compatible
            # APIs reject a request containing both spellings, so do not add a
            # second key when one is already present.  If EXTRA_BODY removed
            # both defaults, restore the legacy spelling for this retry only.
            budget_keys = [key for key in ("max_tokens", "max_completion_tokens")
                           if key in body]
            if not budget_keys:
                budget_keys = ["max_tokens"]
            for key in budget_keys:
                body[key] = output_tokens
        if os.getenv("DEBUG_LLM_REQUEST") == "1":
            print(
                json.dumps(
                    {
                        "model": body.get("model"),
                        "temperature": body.get("temperature"),
                        "top_p": body.get("top_p"),
                        "seed": body.get("seed"),
                        "max_tokens": body.get("max_tokens"),
                        "max_completion_tokens": body.get("max_completion_tokens"),
                        "nonce": nonce,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
        if self.client.calls >= self.client.max_calls:
            raise Unknown("MAX_API_CALLS exhausted; stop and inspect the budget before resuming")
        try:
            response = self.client.post("/chat/completions", body, nonce=nonce)
            choice = response["choices"][0]
            value = choice["message"]["content"]
            if value is None and choice.get("finish_reason") == "length":
                # A reasoning model can spend the whole output budget before
                # emitting any content. Treat that as (retryable) truncation
                # instead of a fatal transport failure; json() then retries
                # with a doubled budget.
                value = ""
            if not isinstance(value, str):
                raise ValueError("No text content")
            # 截断视作模型格式失败；不在奖励路径重试生成并偷选成功者。
            return value if choice.get("finish_reason") != "length" else value + "\n[TRUNCATED_OUTPUT]"
        except Exception as exc:
            raise Unknown(f"{self.role} transport/service failure: {exc}") from exc

    def json(self, prompt, nonce):
        # 判官格式错误属于环境未知，不能变成被训练策略的负奖励。
        path = self.retry_dir / (digest([self.tag, prompt, nonce]) + ".json")
        previous = read(path) if path.exists() else {}
        attempt = previous.get("attempt", 0)
        # A reasoning model can spend the whole output budget before emitting
        # its JSON.  On a retry, increase only the request's output budget; do
        # not change builder/attacker sampling or silently accept partial JSON.
        retry_budget = self._json_retry_budget(attempt)
        for _ in range(2):
            raw = None
            try:
                request_nonce = f"{nonce}:json_attempt={attempt}"
                if retry_budget is None:
                    # Preserve the original call shape for the initial
                    # request, including compatibility with lightweight test
                    # doubles that only accept ``prompt, nonce``.
                    raw = self.complete(prompt, request_nonce)
                else:
                    raw = self.complete(prompt, request_nonce,
                                        output_tokens=retry_budget)
                return parse_json_object(raw)
            except InvalidAction as exc:
                attempt += 1
                write(path, {"attempt": attempt, "error": str(exc), "raw": raw})
                retry_budget = self._json_retry_budget(attempt)
        raise Unknown(f"{self.role} JSON invalid after two attempts; inspect {path}")

    def _json_retry_budget(self, attempt):
        """Return a bounded retry budget, or ``None`` for the first request."""
        if attempt <= 0:
            return None
        configured = int(os.getenv(f"{self.role}_MAX_TOKENS",
                                   os.getenv("LLM_MAX_TOKENS", "4096")))
        if configured < 1:
            raise ValueError(f"{self.role}_MAX_TOKENS must be positive")
        try:
            extras = json.loads(os.getenv(self.role + "_EXTRA_BODY",
                                          os.getenv("LLM_EXTRA_BODY", "{}")))
        except (TypeError, ValueError):
            extras = {}
        # The effective cap is whichever spelling the configured service uses.
        # Ignore null/non-numeric values; complete() will retain the configured
        # request shape and the retry falls back to the role default.
        for key in ("max_completion_tokens", "max_tokens"):
            value = extras.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                configured = int(value)
                break
        cap = int(os.getenv(f"{self.role}_JSON_RETRY_MAX_TOKENS",
                            os.getenv("LLM_JSON_RETRY_MAX_TOKENS", "16384")))
        if cap < 1:
            raise ValueError("LLM_JSON_RETRY_MAX_TOKENS must be positive")
        # Never make the retry smaller than the already configured request if
        # a user intentionally chose a cap above the default retry ceiling.
        return max(configured, min(configured * (2 ** min(attempt, 30)), cap))
