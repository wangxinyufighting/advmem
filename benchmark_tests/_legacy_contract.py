"""仅显式 BENCHMARK_CONTRACT_TESTS=1 时启用的旧模块接口替身。

不提供任何生产检索/推理能力。用于没有旧源码的环境里检查新增编排代码；
在用户的 full_memory_lab 目录正常pytest时，完全不加载本文件。
"""
import copy
import hashlib
import json
import random
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path


def install():
    names = ["common", "llm", "memory", "packs", "retrieve", "agents", "metrics"]
    modules = {}
    for name in names:
        m = types.ModuleType(name)
        m.__file__ = __file__
        modules[name] = m
        sys.modules[name] = m

    def digest(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def read_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    modules["common"].digest = digest
    modules["common"].read_json = read_json
    modules["common"].iter_cases = lambda path: iter(read_json(path))
    modules["common"].normalize_answer = lambda text: " ".join(str(text).lower().split())

    def history_only(case):
        h = {k: copy.deepcopy(case[k]) for k in ["haystack_session_ids", "haystack_dates", "haystack_sessions"]}
        for session in h["haystack_sessions"]:
            for msg in session:
                msg.pop("has_answer", None)
        return h

    @dataclass
    class Round:
        rid: str
        session_id: str
        messages: list

    @dataclass
    class Session:
        session_id: str
        rids: list

    @dataclass
    class Doc:
        id: str
        prov: list
        text: str

    class FullMemory:
        def __init__(self, h):
            self.h, self.rounds, self.sessions = h, {}, {}
            for i, (sid, msgs) in enumerate(zip(h["haystack_session_ids"], h["haystack_sessions"]), 1):
                rids = []
                for j in range(0, len(msgs), 2):
                    rid = f"s{i}:r{j // 2 + 1}"
                    self.rounds[rid] = Round(rid, sid, msgs[j:j+2])
                    rids.append(rid)
                self.sessions[sid] = Session(sid, rids)
            self.fingerprint = digest(h)

        @classmethod
        def build(cls, case):
            return cls(history_only(case))

        def export_history(self):
            return copy.deepcopy(self.h)

        def documents(self):
            return [Doc(rid, [rid], "\n".join(m["content"] for m in r.messages)) for rid, r in self.rounds.items()]

        def render(self, ids, marks=None):
            return "\n".join(f"[{rid}] " + " ".join(m["content"] for m in self.rounds[rid].messages)
                             for rid in sorted(set(ids)))

        def save(self, path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.h))

        @classmethod
        def load(cls, path):
            return cls(read_json(path))

    modules["memory"].FullMemory = FullMemory
    modules["memory"].history_only = history_only

    @dataclass
    class Pack:
        pack_id: str
        full_hash: str
        seed_id: str
        kind: str
        sweep: int
        seed_rids: list
        rids: list
        dropped_rids: list = None
        searches: list = None

        def to_dict(self):
            return asdict(self)

    TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
             "multi-session", "knowledge-update", "temporal-reasoning"]

    def choose_type(pack, full, rng, weights=None):
        allowed = TYPES if len({full.rounds[r].session_id for r in pack.rids}) > 1 else TYPES[:3]
        weights = weights or {t: 1 for t in TYPES}
        return rng.choices(allowed, [weights.get(t, 0) for t in allowed], k=1)[0]

    def check_pack(p, full):
        if p.full_hash != full.fingerprint or not set(p.rids) <= full.rounds.keys():
            raise ValueError("bad pack")

    class Sampler:
        def __init__(self, full, index, seed=0, **kwargs):
            self.full, self.seed = full, seed
            self.seeds = list(full.sessions)

        def sample(self, n):
            seeds = self.seeds[:]
            random.Random(self.seed).shuffle(seeds)
            return [Pack(f"p{i:05d}", self.full.fingerprint, sid, "session", 0,
                         self.full.sessions[sid].rids, list(self.full.rounds))
                    for i, sid in enumerate(seeds[:n], 1)]

    for key, value in locals().copy().items():
        if key in ["Pack", "TYPES", "choose_type", "check_pack", "Sampler"]:
            setattr(modules["packs"], key, value)

    class Client:
        @classmethod
        def from_env(cls, role, cache):
            return cls()

        def json(self, system, data, **kwargs):
            raise AssertionError("测试没有允许真实模型调用")

    modules["llm"].Client = Client

    def forbidden(*args, **kwargs):
        raise AssertionError("请显式mock模型/检索；兼容替身不执行生产推理")

    for key in ["CrossReranker", "Embedder", "Retriever"]:
        setattr(modules["retrieve"], key, forbidden)
    for key in ["answer", "grade", "Attacker", "gate"]:
        setattr(modules["agents"], key, forbidden)

    @dataclass
    class Gold:
        qid: str
        question: str
        answer: str
        date: str
        qtype: str
        sessions: set
        rounds: set
        abstention: bool

        @classmethod
        def from_case(cls, case, full):
            ids = set()
            for si, session in enumerate(case["haystack_sessions"], 1):
                for mi, m in enumerate(session):
                    if m.get("has_answer") is True:
                        ids.add(f"s{si}:r{mi//2+1}")
            return cls(case["question_id"], case["question"], str(case["answer"]), case["question_date"],
                       case["question_type"], set(case["answer_session_ids"]), ids,
                       case["question_id"].endswith("_abs"))

    modules["metrics"].Gold = Gold
