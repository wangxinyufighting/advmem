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

    @classmethod
    def load(cls, path):
        c = cls(**read(path)) if path else cls()
        if c.hint_mode not in {"target_type", "hidden"} or c.defect_mode not in {"strict", "direct"}:
            raise ValueError("Unknown hint_mode/defect_mode")
        if c.embedding not in {"none", "local", "api"} or c.gate_mode not in {"basic", "full"}:
            raise ValueError("Unknown embedding/gate_mode")
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

    def complete(self, prompt, nonce, temperature=0.0, max_tokens=None):
        if max_tokens is None:
            max_tokens = int(os.getenv(f"{self.role}_MAX_TOKENS",
                                       os.getenv("LLM_MAX_TOKENS", "4096")))
        body = {"model": self.client.model, "messages": prompt, "temperature": temperature,
                "max_tokens": max_tokens}
        
        if os.getenv("DEBUG_LLM_REQUEST") == "1":
            print(
                json.dumps(
                    {
                        "model": body.get("model"),
                        "temperature": body.get("temperature"),
                        "top_p": body.get("top_p"),
                        "seed": body.get("seed"),
                        "max_tokens": body.get("max_tokens"),
                        "nonce": nonce,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
        
        if os.getenv("LLM_JSON_MODE", "1") == "1":
            body["response_format"] = {"type": "json_object"}
        extras = os.getenv(self.role + "_EXTRA_BODY", os.getenv("LLM_EXTRA_BODY", "{}"))
        for key, value in json.loads(extras).items():
            if value is None:
                body.pop(key, None)
            else:
                body[key] = value
        if self.client.calls >= self.client.max_calls:
            raise Unknown("MAX_API_CALLS exhausted; stop and inspect the budget before resuming")
        try:
            response = self.client.post("/chat/completions", body, nonce=nonce)
            choice = response["choices"][0]
            value = choice["message"]["content"]
            if not isinstance(value, str):
                raise ValueError("No text content")
            # 截断视作模型格式失败；不在奖励路径重试生成并偷选成功者。
            return value if choice.get("finish_reason") != "length" else value + "\n[TRUNCATED_OUTPUT]"
        except Exception as exc:
            raise Unknown(f"{self.role} transport/service failure: {exc}") from exc

    def json(self, prompt, nonce):
        # 判官格式错误属于环境未知，不能变成被训练策略的负奖励。
        path = self.retry_dir / (digest([self.tag, prompt, nonce]) + ".json")
        attempt = read(path)["attempt"] if path.exists() else 0
        for _ in range(2):
            raw = None
            try:
                raw = self.complete(prompt, f"{nonce}:json_attempt={attempt}")
                return parse_json_object(raw)
            except InvalidAction as exc:
                attempt += 1
                write(path, {"attempt": attempt, "error": str(exc), "raw": raw})
        raise Unknown(f"{self.role} JSON invalid after two attempts; inspect {path}")
