"""批量实验共用工具：固定 case、缓存模型实例、原子写盘、显式统计未知结果。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from common import digest, iter_cases, read_json
from llm import Client
from memory import FullMemory, history_only
from packs import TYPES
from retrieve import CrossReranker, Embedder, Retriever

VERSION = "reader-attacker-benchmark-1.0"


def save(path: Path, value) -> None:
    """先写临时文件再替换；中断不会把现有 checkpoint 写成半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(temp, path)


def read_lines(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def table(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def kind(case: dict) -> str:
    return "abstention" if case["question_id"].endswith("_abs") else case["question_type"]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def bool_stats(values: list) -> dict:
    """错误/未判定记 unknown；completed 上的比例与全体保守界同时公开。"""
    n, yes, no = len(values), sum(x is True for x in values), sum(x is False for x in values)
    return {"selected": n, "completed": yes + no, "successes": yes, "failures": no,
            "unknown": n - yes - no, "rate_completed": yes / (yes + no) if yes + no else None,
            "lower_all_selected": yes / n if n else None,
            "upper_all_selected": (n - no) / n if n else None}


def common_args(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", required=True)
    p.add_argument("--packs-dir", help="复用 benchmark_packs 的 selected_cases.json 和 cases/；不重新抽样")
    p.add_argument("--per-type", type=int, default=5, help="独立运行时每类数量，0=全部；复用 packs-dir 时忽略")
    p.add_argument("--types", nargs="+", choices=[*TYPES, "abstention"], help="不传时：独立运行取六类，复用时取原清单")
    p.add_argument("--ids", help="独立运行时限制为 train/dev 的 question_id JSON数组")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list", action="store_true", help="只打印选中案例，不加载模型/调用API")
    p.add_argument("--embedding", choices=["none", "local", "api"], default="local")
    p.add_argument("--embed-model")
    p.add_argument("--reranker-model")
    p.add_argument("--cache", default=".cache")
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true", help="保留已完成阶段，仅重试错误或未完成阶段")
    return p


def selection(args) -> list[dict]:
    if args.per_type < 0:
        raise ValueError("per-type 不能为负")
    if args.packs_dir and args.ids:
        raise ValueError("--packs-dir 已固定案例，不同时使用 --ids")
    source = None
    allowed = None
    if args.packs_dir:
        source = read_json(Path(args.packs_dir) / "selected_cases.json")
        allowed = {r["question_id"] for r in source}
        if len(allowed) != len(source):
            raise ValueError("selected_cases.json 含重复ID")
    elif args.ids:
        values = read_json(args.ids)
        if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
            raise ValueError("--ids 必须是非空 question_id 字符串数组")
        allowed = set(values)
    rows, found, seen = [], set(), set()
    for index, case in enumerate(iter_cases(args.data)):
        qid = case["question_id"]
        if qid in seen:
            raise ValueError(f"数据存在重复 question_id: {qid}")
        seen.add(qid)
        if allowed is not None and qid not in allowed:
            continue
        found.add(qid)
        rows.append({"case_index": index, "question_id": qid, "question_type": kind(case),
                     "stored_question_type": case["question_type"], "case_hash": digest(case)})
    if allowed is not None and allowed - found:
        raise ValueError(f"输入数据缺少指定案例: {sorted(allowed - found)[:5]}")
    types = args.types if args.types is not None else (None if source is not None else TYPES)
    rows = [r for r in rows if types is None or r["question_type"] in types]
    if source is not None:
        original = {r["question_id"]: r for r in source}
        for row in rows:
            if row["case_index"] != original[row["question_id"]]["case_index"]:
                raise ValueError("数据顺序与原 pack 实验不同，禁止按错误目录复用")
    else:
        picked = []
        for t in types:
            candidates = sorted((r for r in rows if r["question_type"] == t),
                                key=lambda r: digest([args.seed, "case-selection", r["question_id"]]))
            picked.extend(candidates[:args.per_type] if args.per_type else candidates)
        rows = picked
    rows.sort(key=lambda r: r["case_index"])
    if not rows:
        raise ValueError("没有选中案例")
    return rows


def selected_data(args, rows):
    wanted = {r["question_id"]: r for r in rows}
    for case in iter_cases(args.data):
        if case["question_id"] in wanted:
            meta = wanted[case["question_id"]]
            if digest(case) != meta["case_hash"]:
                raise ValueError("选样后数据已改变")
            yield meta, case


def init_run(args, rows) -> str:
    """不记录密钥；模型/参数/源码改变时拒绝混入已有结果。"""
    import agents, common, llm, memory, packs, retrieve
    env_names = ["LLM_MODEL", "LLM_BASE_URL", "LLM_JSON_MODE", "LLM_EXTRA_BODY", "LLM_MAX_TOKENS",
                 "EMBED_MODEL", "EMBED_BASE_URL", "EMBED_QUERY_PREFIX", "EMBED_DOCUMENT_PREFIX",
                 "EMBED_CHUNK_CHARS"]
    env_names += [f"{role}_{suffix}" for role in ["ATTACKER", "DEFENDER", "JUDGE", "PLANNER"]
                  for suffix in ["MODEL", "BASE_URL"]]
    sources = {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
               for m in [agents, common, llm, memory, packs, retrieve]}
    for name in ["_benchmark.py", "benchmark_run.py", "benchmark_attack.py"]:
        p = Path(__file__).with_name(name)
        if p.exists():
            sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    settings = {k: v for k, v in vars(args).items() if k not in {"resume", "list", "out"}}
    config = {"version": VERSION, "settings": settings, "selection": rows,
              "model_settings": {k: os.environ[k] for k in env_names if k in os.environ}, "source_hashes": sources}
    if getattr(args, "weights", None):
        config["type_weights_content"] = read_json(args.weights)
    # pack 内容也属于实验输入，断点续跑前验证；缺失文件会在对应case明确报错。
    if args.packs_dir:
        hashes = {}
        for row in rows:
            folder = Path(args.packs_dir) / "cases" / f"{row['case_index']:04d}"
            for name in ["full_memory.json", "packs.jsonl", "case_report.json"]:
                p = folder / name
                hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
        config["input_hashes"] = hashes
    path = Path(args.out) / "config.json"
    if path.exists():
        if read_json(path) != config:
            raise ValueError("输出目录已有不同模型/参数/输入/源码的实验，请更换 --out")
        if not args.resume:
            raise ValueError("输出目录已有实验；继续请加 --resume，独立重跑请换 --out")
    save(path, config)
    save(Path(args.out) / "selected_cases.json", rows)
    return digest(config)


class BudgetExhausted(RuntimeError):
    """预算耗尽应暂停，而不是把剩余所有样本标成失败。"""


class ScopedClient(Client):
    """失败阶段重试时更换缓存命名空间，避免永久重放已缓存的坏JSON/坏字段。"""
    scope = ""

    def json(self, system, data, *, temperature=0, nonce=""):
        try:
            return super().json(system, data, temperature=temperature, nonce=digest([self.scope, nonce]))
        except Exception as exc:
            if "达到 API 调用预算" in str(exc):
                raise BudgetExhausted(str(exc)) from exc
            raise


class Runtime:
    """模型惰性加载；全部案例共用一个 encoder 和各角色的一个API客户端。"""
    def __init__(self, args, run_id):
        self.args, self.run_id = args, run_id
        self.clients, self.embedder, self.reranker = {}, None, None
        self.index_models_loaded = False

    def client(self, role, scope):
        if role not in self.clients:
            self.clients[role] = ScopedClient.from_env(role, Path(self.args.cache) / "api")
        self.clients[role].scope = f"{self.run_id}:{scope}"
        return self.clients[role]

    def index(self, full):
        if not self.index_models_loaded:
            if self.args.embedding != "none":
                self.embedder = Embedder(self.args.embedding, self.args.embed_model, Path(self.args.cache) / "api")
            if self.args.reranker_model:
                self.reranker = CrossReranker(self.args.reranker_model)
            self.index_models_loaded = True
        return Retriever(full.documents(), self.embedder, Path(self.args.cache) / "index", self.reranker)

    def stats(self):
        return {role: {"model": c.model, "network_calls": c.calls, "cache_hits": c.cache_hits}
                for role, c in self.clients.items()}


def load_full(args, meta, case):
    if args.packs_dir:
        folder = Path(args.packs_dir) / "cases" / f"{meta['case_index']:04d}"
        record = folder / "case_report.json"
        if record.exists() and read_json(record).get("case_hash") != meta["case_hash"]:
            raise ValueError("当前数据与原 pack 实验的 case_hash 不一致")
        path = folder / "full_memory.json"
        full = FullMemory.load(path) if path.exists() else FullMemory.build(case)
    else:
        full = FullMemory.build(case)
    if full.export_history() != history_only(case):
        raise ValueError("full memory 与当前 case 的历史不一致")
    return full


def checkpoint(path, meta, args, run_id):
    if path.exists() and args.resume:
        state = read_json(path)
        if state.get("run_id") != run_id or state.get("case_hash") != meta["case_hash"]:
            raise ValueError("checkpoint 来自不同实验或数据")
        return state
    return {**meta, "run_id": run_id, "status": "pending", "attempts": {}}


def attempt(state, stage, path):
    number = state.setdefault("attempts", {}).get(stage, 0) + 1
    state["attempts"][stage] = number
    save(path, state)
    return number


def error_text(exc):
    return f"{type(exc).__name__}: {exc}"


def print_selection(rows):
    print(json.dumps({"counts": dict(Counter(r["question_type"] for r in rows)), "cases": rows},
                     ensure_ascii=False, indent=2), flush=True)
