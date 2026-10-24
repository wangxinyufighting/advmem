"""出题流水线修复的离线回归测试：预算溢出、token pack、事实句邻居、题型计划、针对M出题、划分对齐。"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import Full, Pack
from admem.audit_prompts import TYPES, plan_item_type, structural_types
from admem.common import BudgetExceeded, Config, Unknown, connect, read, write
from admem.data import TypeScheduler, policy_type, resolve_type_policy
from admem.environment import Environment
from admem.packing import diversify, fact_sentences, token_pool
from admem.pipeline import _bank_pack, attacker_state, generate_bank, make_pool

ROOT = Path(__file__).resolve().parents[2]


class Chars:
    def count(self, text):
        return len(text)

    def prompt_count(self, messages):
        return sum(len(m["content"]) for m in messages)

    def split(self, text, limit):
        for start in range(0, len(text), limit):
            yield start, min(len(text), start + limit), text[start:start + limit]


def history(sessions, dates=None):
    dates = dates or [f"2023-0{i + 1}-01" for i in range(len(sessions))]
    return {"haystack_session_ids": [f"src_{i}" for i in range(len(sessions))],
            "haystack_dates": dates, "haystack_sessions": sessions}


def pair(user, assistant="Noted."):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def context(qtype="single-session-user"):
    return {"key": "c0000", "split": "train", "question_type": qtype, "question_date": "2023-09-01"}


def question(q="What is my cat's name?", E=("s1:r1",), qtype="single-session-user"):
    return {"q": q, "a": "Milo", "type": qtype, "question_date": "2023-04-01", "E": list(E)}


# ---------------------------------------------------------------- 1. 预算溢出

def test_budget_exceeded_is_an_unknown_with_details():
    exc = BudgetExceeded("too long", stage="attacker_prompt", tokens=10, limit=5)
    assert isinstance(exc, Unknown)
    assert exc.details() == {"stage": "attacker_prompt", "tokens": 10, "limit": 5, "reason": "too long"}


def test_over_budget_packs_are_logged_without_retry_or_case_abort(setup_case, env, tmp_path, monkeypatch):
    full, path, ctx = setup_case
    packs = [Pack(f"p{i}", full.fingerprint, "seed", "session", 0, ["s1:r1"], list(full.rounds)) for i in range(7)]
    monkeypatch.setattr("admem.pipeline.make_pool", lambda *a: packs)
    monkeypatch.setenv("PACK_RETRIES", "3")
    monkeypatch.setenv("PACK_BACKOFF", "0")
    monkeypatch.setenv("PACK_FAIL_LIMIT", "5")
    env.cfg.attacker_input_tokens = 50  # 连种子都装不下
    assert generate_bank(full, path, ctx, env, tmp_path / "bank") == 0
    logs = read(tmp_path / "bank" / "bank_log.json")
    assert [row["status"] for row in logs] == ["over_budget"] * 7
    assert {row["stage"] for row in logs} == {"attacker_prompt"}
    assert all(row["tokens"] > row["limit"] == 50 for row in logs)
    assert not env.control["prompts"]  # 一次模型调用都没有
    assert len(list((tmp_path / "bank" / "pack_traces").glob("*.json"))) == 7  # 每个pack一次，不重试


def test_lowest_ranked_neighbors_are_dropped_before_the_seed(env, tmp_path):
    full = Full.build(history([pair("I adopted a cat named Milo."),
                               pair("Short note about Milo's vet visit."),
                               pair("x" * 4000)]))
    pack = Pack("p1", full.fingerprint, "src_0", "session", 0, ["s1:r1"], ["s1:r1", "s2:r1", "s3:r1"])
    env.cfg.attacker_view = "full"
    roomy = attacker_state(full, tmp_path / "f.json", context(), [], pack, [], env, 0)
    assert roomy["pack"]["rids"] == ["s1:r1", "s2:r1", "s3:r1"] and "budget_dropped_rids" not in roomy
    env.cfg.attacker_input_tokens = env.counter.prompt_count(roomy["prompt"]) - 1000
    tight = attacker_state(full, tmp_path / "f.json", context(), [], pack, [], env, 0)
    assert tight["pack"]["rids"] == ["s1:r1", "s2:r1"]  # 末尾（排名最低）的邻居先被去掉
    assert tight["budget_dropped_rids"] == ["s3:r1"]
    assert env.counter.prompt_count(tight["prompt"]) <= env.cfg.attacker_input_tokens
    # 排名靠前的超大邻居装不下时跳过它，仍保留后面放得下的小邻居。
    giant_first = Pack("p2", full.fingerprint, "src_0", "session", 0, ["s1:r1"], ["s1:r1", "s3:r1", "s2:r1"])
    kept = attacker_state(full, tmp_path / "f.json", context(), [], giant_first, [], env, 0)
    assert kept["pack"]["rids"] == ["s1:r1", "s2:r1"] and kept["budget_dropped_rids"] == ["s3:r1"]


def test_unchanged_prompt_when_everything_fits(env, setup_case, tmp_path):
    full, path, ctx = setup_case
    pack = Pack("p1", full.fingerprint, "seed", "session", 0, ["s1:r1"], list(full.rounds))
    state = attacker_state(full, path, ctx, [], pack, [question("Earlier question?")], env, 0)
    payload = json.loads(state["prompt"][1]["content"])
    assert payload["type"] == ctx["question_type"] and "audit_context" in payload  # 旧版逐字相同的格式


def test_gate_overflow_only_drops_that_question(setup_case, env, monkeypatch):
    full, path, ctx = setup_case
    items = [question("Which name did I give my cat?"), question()]
    monkeypatch.setattr(env.policy("ATTACKER"), "complete", lambda *a, **kw: json.dumps({"items": items}))
    calls = []

    def gate(full, pack, item, date, qtype):
        calls.append(item)
        if len(calls) == 1:
            raise BudgetExceeded("Judge context exceeds configured token budget", stage="judge:oracle",
                                 tokens=9, limit=1)
        return {"status": "accepted", "item": item}

    monkeypatch.setattr(env, "gate", gate)
    pack = Pack("p1", full.fingerprint, "seed", "session", 0, ["s1:r1"], list(full.rounds))
    _, logs, added = _bank_pack(full, path, ctx, pack, 0, [], env, "ATTACKER")
    assert [row["status"] for row in logs] == ["over_budget", "accepted"]
    assert logs[0]["stage"] == "judge:oracle" and logs[0]["item"] == items[0]
    assert added == [items[1]]


def test_screen_is_fitted_to_the_judge_budget(env):
    sessions = [pair("My cat is named Milo.")] + [pair(f"Story {i} about my cat " + "y" * 3000) for i in range(6)]
    full = Full.build(history(sessions))
    q = question("What is my cat's name, currently?")  # "currently"会触发宽检索screen
    all_rids, info = env._screen_rids(full, q)
    assert info is None and all_rids == set(full.rounds)  # 装得下时与旧实现相同
    env.cfg.judge_input_tokens = 9000
    rids, info = env._screen_rids(full, q)
    assert "s1:r1" in rids and len(rids) < len(full.rounds)
    assert info["omitted"] > 0 and info["kept"] == len(rids)
    env.cfg.judge_input_tokens = 100
    with pytest.raises(BudgetExceeded, match="Screen evidence alone"):
        env._screen_rids(full, q)


# ---------------------------------------------------------------- 2. token pack与事实句邻居

def test_token_pool_splits_long_sessions_and_reports_unfittable_rounds(env):
    long_session = [m for i in range(6) for m in pair(f"I bought item {i}. " + "z" * 1500)]
    full = Full.build(history([long_session, pair("w" * 40000), pair("I like tea.")]))
    env.cfg.pack_builder, env.cfg.neighbor_mode, env.cfg.attacker_input_tokens = "token", "topical", 14000
    report = {}
    packs = make_pool(full, env, report)
    seeds = [p.seed_rids for p in packs]
    assert [rid for chunk in seeds if chunk[0].startswith("s1:") for rid in chunk] == full.sessions["src_0"].rids
    assert sum(chunk[0].startswith("s1:") for chunk in seeds) >= 2  # 长session被切成多个片段
    assert report["split_sessions"][0]["session"] == "s1"
    assert [r["rid"] for r in report["skipped_rounds"]] == ["s2:r1"]  # 装不下的round被记录而不是静默丢弃
    assert all("s2:r1" not in p.rids for p in packs)  # 也不会被选作别的种子的邻居
    for pack in packs:  # 每个pack的种子部分都装得进attacker预算
        state = attacker_state(full, "f.json", context(), [], pack, [], env, pack.pack_id)
        assert state is not None and set(pack.seed_rids) <= set(state["pack"]["rids"])


def test_fact_sentences_keep_incidental_personal_facts():
    texts = ["Can you plan a 3-day Lisbon trip? I just bought a road bike for $900 last weekend, "
             "so maybe add a cycling day. Please write a short poem too. The weather is nice.",
             "I want you to act as a Linux terminal. My daughter turns 5 next week."]
    queries = fact_sentences(texts)
    assert "I just bought a road bike for $900 last weekend, so maybe add a cycling day." in queries
    assert "My daughter turns 5 next week." in queries
    assert not any(q.startswith(("Please write", "The weather", "I want you to act")) for q in queries)


def test_fact_sentences_survive_run_on_text_without_punctuation():
    filler = " ".join(["detail step option idea"] * 20)
    queries = fact_sentences([f"Can you help with my resume {filler} by the way I just bought a road bike for $900"])
    assert any("I just bought a road bike for $900" in q for q in queries)
    assert all(len(q.split()) <= 50 for q in queries)


def test_diversify_covers_more_sessions_first():
    ranked = ["a1", "a2", "a3", "b1", "c1"]
    assert diversify(ranked, lambda r: r[0], 4, 2) == ["a1", "b1", "c1", "a2"]
    assert diversify(ranked, lambda r: r[0], 4, 1) == ["a1", "b1", "c1"]


def real_env(tmp_path, **overrides):
    cfg = Config(embedding="none", cache=str(tmp_path / "cache"), attacker_input_tokens=60000,
                 judge_input_tokens=100000, **overrides)
    return Environment(cfg, Chars(), connect(str(ROOT)))


def test_fact_neighbors_find_an_incidental_mention_that_topical_retrieval_misses(tmp_path):
    trip = ("Can you plan a three day Lisbon trip with hotels near the old town, a food tour, "
            "museum tickets and a day trip to Sintra? ")
    case = history([
        pair(trip + "I just bought a road bike for $900 last weekend, so maybe add a cycling day."),
        pair("Which Lisbon hotels near the old town have rooftop views? Compare museum tickets "
             "and food tour prices for a three day trip including Sintra."),
        pair("Can you give me a lentil soup recipe? By the way, my new road bike helmet arrived yesterday."),
    ], ["2023/05/01 (Mon) 10:00", "2023/05/02 (Tue) 10:00", "2023/05/03 (Wed) 10:00"])
    env = real_env(tmp_path, neighbors=1, pack_builder="token")
    full = env.memory_module.FullMemory.build(case)
    env.cfg.neighbor_mode = "topical"
    topical = {p.seed_rids[0]: [r for r in p.rids if r not in p.seed_rids] for p in token_pool(full, env)[0]}
    env.cfg.neighbor_mode = "facts"
    facts_pool, _ = token_pool(full, env)
    facts = {p.seed_rids[0]: [r for r in p.rids if r not in p.seed_rids] for p in facts_pool}
    assert topical["s1:r1"] == ["s2:r1"]  # 整段发言检索：被旅行话题主导
    assert facts["s1:r1"] == ["s3:r1"]    # 事实句检索：找到顺带提到自行车的session
    assert any("road bike" in q for q in facts_pool[0].searches)


# ---------------------------------------------------------------- 3. 题型计划

def test_auto_policy_keeps_old_behaviour():
    cfg = Config()
    assert resolve_type_policy(cfg) == "official"
    cfg.hint_mode = "hidden"
    assert resolve_type_policy(cfg) == "uniform"
    cfg.type_policy = "balanced"
    assert resolve_type_policy(cfg) == "balanced"
    cfg.hint_mode, cfg.type_policy = "target_type", "uniform"
    assert policy_type({"question_type": "multi-session"}, cfg, 0) in TYPES  # official类型不再被使用


def test_scheduler_balances_slots_is_idempotent_and_respects_feasibility():
    cfg = Config(type_policy="balanced", types_per_pack=2, questions_per_pack=4)
    scheduler = TypeScheduler(cfg)
    for i in range(30):
        plan = scheduler.plan(TYPES, ["bank", i])
        assert sum(plan.values()) == 4 and len(plan) == 2
    counts = scheduler.requested
    assert max(counts.values()) - min(counts.values()) <= 2
    assert scheduler.plan(TYPES, ["bank", 3]) == scheduler.plans[json_key(["bank", 3])]
    assert sum(scheduler.requested.values()) == 120  # 重复调用不重复计数
    only = scheduler.plan(["single-session-user"], ["bank", 99])
    assert only == {"single-session-user": 4}
    assert scheduler.plan([], ["bank", 100]) == {}


def json_key(identity):
    from admem.common import digest
    return digest(identity)


def test_balanced_policy_never_reads_the_official_type(env, tmp_path):
    full = Full.build(history([pair("I adopted a cat named Milo.", "1. Feed twice a day."),
                               pair("I adopted a dog named Rex.")]))
    pack = Pack("p1", full.fingerprint, "src_0", "session", 0, ["s1:r1"], ["s1:r1", "s2:r1"])
    env.cfg.type_policy = "balanced"
    ctx = {**context(), "question_type": "TARGET_CANARY_TYPE"}
    states = [attacker_state(full, tmp_path / "f.json", ctx, [], pack, [], env, i) for i in range(12)]
    planned = Counter(t for s in states for t in s.get("type_plan", {s["qtype"]: 4}))
    assert set(planned) <= set(TYPES) and len(planned) >= 5
    assert all("TARGET_CANARY_TYPE" not in json.dumps(s["prompt"]) for s in states)
    single = Pack("p2", full.fingerprint, "src_0", "session", 0, ["s1:r1"], ["s1:r1"])
    assert "multi-session" not in structural_types(full, single)
    no_assistant = Full.build(history([[{"role": "user", "content": "I adopted a cat named Milo."}]]))
    lone = Pack("p3", no_assistant.fingerprint, "src_0", "session", 0, ["s1:r1"], ["s1:r1"])
    assert "single-session-assistant" not in structural_types(no_assistant, lone)


def test_type_plan_prompt_and_per_item_validation(env, monkeypatch, tmp_path):
    full = Full.build(history([pair("My cat is named Milo."), pair("My cat Milo is three.")]))
    pack = Pack("p1", full.fingerprint, "src_0", "session", 0, ["s1:r1"], ["s1:r1", "s2:r1"])
    env.cfg.type_policy, env.cfg.types_per_pack = "balanced", 2
    scheduler = TypeScheduler(env.cfg)
    monkeypatch.setattr(scheduler, "plan", lambda feasible, identity: {"single-session-user": 2, "multi-session": 2})
    state = attacker_state(full, tmp_path / "f.json", context(), [], pack, [], env, 0, scheduler)
    payload = json.loads(state["prompt"][1]["content"])
    assert payload["slots"] == {"single-session-user": 2, "multi-session": 2}
    assert payload["max_evidence_rounds"] == {"single-session-user": 3, "multi-session": 8}
    assert "slots maps each requested type" in state["prompt"][0]["content"]
    used = Counter()
    assert plan_item_type(state, question(), used) == "single-session-user"
    assert plan_item_type(state, question(qtype="temporal-reasoning"), used) == "outside-type-plan"
    items = [question(), question("Is my cat Milo?", ("s2:r1",)),  # 单session题引用了邻居 → 拒绝
             question("How old is my cat Milo?", ("s1:r1", "s2:r1"), "multi-session"),
             question("When did I get Milo?", ("s1:r1",), "temporal-reasoning")]  # 不在计划内
    for item in items:
        item["question_date"] = context()["question_date"]
    monkeypatch.setattr(env.policy("ATTACKER"), "complete", lambda *a, **kw: json.dumps({"items": items}))
    monkeypatch.setattr("admem.pipeline.TypeScheduler", lambda cfg: scheduler)
    _, logs, added = _bank_pack(full, tmp_path / "f.json", context(), pack, 0, [], env, "ATTACKER",
                                scheduler=scheduler)
    statuses = [(row["status"], row.get("reason")) for row in logs]
    assert len(statuses) == 4, statuses
    assert statuses[0] == ("accepted", None), statuses
    assert statuses[1] == ("rejected", "Single-session question cites non-seed rounds")
    assert statuses[2][0] == "accepted" and logs[2]["qtype"] == "multi-session"
    assert statuses[3] == ("rejected", "Question type/date mismatch")
    assert [q["type"] for q in added] == ["single-session-user", "multi-session"]


# ---------------------------------------------------------------- 4. 针对当前M出题

def test_bank_against_memory_shows_provenance_and_records_defects(setup_case, env, tmp_path):
    full, path, ctx = setup_case
    pack = Pack("p1", full.fingerprint, "seed", "session", 0, ["s1:r1"], list(full.rounds))
    memory = [{"id": "m1", "text": "The user enjoys gardening.", "prov": ["s2:r1"], "kind": "card"}]
    env.cfg.attacker_memory_entries = 5
    _, logs, added = _bank_pack(full, path, ctx, pack, 0, [], env, "ATTACKER", memory=memory)
    prompt = [p for role, p, _ in env.control["prompts"] if role == "ATTACKER"][0]
    payload = json.loads(prompt[1]["content"])
    marks = {r["rid"]: r["referenced_by_memory"] for r in payload["rounds"]}
    assert marks == {"s1:r1": [], "s2:r1": ["m1"]} and payload["memory_entries"][0]["id"] == "m1"
    assert logs[0]["status"] == "accepted" and logs[0]["defect"] == "missing" and len(added) == 1
    remembered = memory + [{"id": "m2", "text": "The user's cat is named Milo.", "prov": ["s1:r1"], "kind": "card"}]
    _, logs, added = _bank_pack(full, path, ctx, pack, 0, [], env, "ATTACKER", memory=remembered,
                                defects_only=True)
    assert logs[0]["defect"] == "none" and logs[0]["status"] == "accepted_no_defect" and added == []


def test_generate_bank_merges_base_bank_and_writes_manifest(setup_case, env, tmp_path):
    full, path, ctx = setup_case
    base = [question("Earlier question about gardening?", ("s2:r1",))]
    memory = [{"id": "m1", "text": "The user enjoys gardening.", "prov": ["s2:r1"], "kind": "card"}]
    out = tmp_path / "bank_r1"
    total = generate_bank(full, path, ctx, env, out, memory=memory, base_bank=base, memory_source="build/M_build")
    bank = read(out / "bank.json")
    assert total == len(bank) == 2 and bank[0] == base[0]
    manifest = read(out / "bank_manifest.json")
    assert manifest["base_bank"]["questions"] == 1 and manifest["memory"]["entries"] == 1
    assert manifest["type_policy"] == "official" and manifest["label_informed_types"] is True
    pool = read(out / "pack_pool.json")
    assert pool["packs"] and all("seed_id" not in p for p in pool["packs"])
    first_prompt = [p for role, p, _ in env.control["prompts"] if role == "ATTACKER"][0]
    context_items = json.loads(first_prompt[1]["content"])["audit_context"]["accepted_items"]
    assert context_items[0]["q"] == base[0]["q"]  # 旧题进入audit_context，避免重复出题


# ---------------------------------------------------------------- 5. 较小问题

def test_legacy_pool_reports_over_budget_seed_sessions(tmp_path):
    env = real_env(tmp_path, pack_chars=500)
    full = env.memory_module.FullMemory.build(history([pair("I like tea."), pair("x" * 2000)]))
    report = {}
    pool = make_pool(full, env, report)
    assert [p.seed_rids for p in pool] == [["s1:r1"]]
    assert report["skipped_sessions"] == [{"session": "s2", "rounds": 1, "reason": "seed_over_char_budget",
                                           "pack_chars": 500}]


def _cases_file(path):
    shared = history([pair("I adopted a cat named Milo."), pair("I like tea.")])
    other = history([pair("I moved to Denver.")])
    cases = []
    for qid, hist in [("q1", shared), ("q1_abs", shared), ("q2", other), ("q3", other)]:
        sessions = deepcopy(hist["haystack_sessions"])
        sessions[0][0]["has_answer"] = True
        cases.append({**hist, "haystack_sessions": sessions, "question_id": qid, "question_type": "single-session-user",
                      "question": "Q?", "answer": "A", "question_date": "2023-09-01",
                      "answer_session_ids": [hist["haystack_session_ids"][0]]})
    path.write_text(json.dumps(cases), encoding="utf-8")


def _run_prepare(tmp_path, *extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "cases.json"
    _cases_file(data)
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "prepare_longmemeval_cases.py"), "--input", str(data),
                    "--out-dir", str(out), *map(str, extra)], check=True, capture_output=True)
    splits = {s: [c["question_id"] for c in read(out / f"{s}.json")] for s in ("train", "valid", "test")}
    return out, splits


def test_case_splitter_keeps_shared_histories_together_and_can_follow_a_manifest(tmp_path):
    for seed in range(12):
        _, splits = _run_prepare(tmp_path / str(seed), "--seed", seed, "--train-ratio", 0.4, "--valid-ratio", 0.3)
        where = {qid: s for s, ids in splits.items() for qid in ids}
        assert where["q1"] == where["q1_abs"] and where["q2"] == where["q3"]
    manifest = tmp_path / "manifest.json"
    write(manifest, {"cases": [{"question_id": "q1", "split": "test"}, {"question_id": "q1_abs", "split": "test"},
                               {"question_id": "q2", "split": "val"}, {"question_id": "q3", "split": "train"}]})
    out, splits = _run_prepare(tmp_path / "m", "--manifest", manifest)
    assert splits == {"train": ["q3"], "valid": ["q2"], "test": ["q1", "q1_abs"]}
    assert read(out / "report.json")["split_source"] == "manifest"


def test_case_splitter_label_free_session_selection(tmp_path):
    manifest = tmp_path / "manifest.json"
    write(manifest, {"cases": [{"question_id": q, "split": "train"} for q in ("q1", "q1_abs", "q2", "q3")]})
    out, _ = _run_prepare(tmp_path, "--select-sessions", "all", "--no-question-meta", "--manifest", manifest)
    rows = [json.loads(line) for line in (out / "train_builder.jsonl").read_text().splitlines()]
    assert len(rows) == 2 + 2 + 1 + 1  # 每个session一条，包括_abs变体
    assert not any("Question type:" in row["prompt"] for row in rows)
    report = read(out / "report.json")
    assert report["label_informed_selection"] is False and report["question_meta_in_prompt"] is False


def _runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("probe_runner_fixes", ROOT / "scripts" / "probe_longmemeval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_runner_bank_rounds_target_the_previous_memory(tmp_path, monkeypatch):
    runner = _runner()
    config = tmp_path / "config.json"
    write(config, {"project": str(ROOT), "tokenizer": "test/tokenizer", "embedding": "none"})
    prepared = tmp_path / "prepared"
    write(prepared / "manifest.json", {"cases": [{"key": "c0000", "split": "val",
                                                "question_type": "single-session-user"}]})
    write(prepared / "cases" / "c0000" / "context.json", {
        "key": "c0000", "split": "val", "question_type": "single-session-user", "full_hash": "full-c0000"})
    write(prepared / "cases" / "c0000" / "full.json", {"fingerprint": "full-c0000", "sessions": [{"session_id": "s1"}]})
    monkeypatch.setenv("LLM_MODEL", "offline-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    for role in ("BUILDER", "ATTACKER", "DEFENDER", "JUDGE"):
        monkeypatch.delenv(role + "_MODEL", raising=False)
        monkeypatch.delenv(role + "_BASE_URL", raising=False)
        monkeypatch.delenv(role + "_EXTRA_BODY", raising=False)
    monkeypatch.delenv("LLM_EXTRA_BODY", raising=False)
    args = runner.parser().parse_args([
        "--config", str(config), "--prepared", str(prepared), "--data", str(tmp_path / "unused.json"),
        "--out", str(tmp_path / "out"), "--case-limit", "1", "--state-limit", "1", "--samples", "2",
        "--bank-rounds", "2", "--bank-defects-only"])
    fingerprint = Config.load(config).fingerprint()
    calls = []

    def fake_cli(command, config, *arguments):
        arguments = [str(a) for a in arguments]
        calls.append((command, arguments))
        out = Path(arguments[arguments.index("--out") + 1])
        if command in {"bank", "run"}:
            write(out / "summary.json", [{"key": "c0000", "status": "ok"}])
            name = "bank.json" if command == "bank" else "M_final.json"
            write(out / "c0000" / name, [{"q": "generated"}] if command == "bank" else [])
        elif command == "export":
            row = {"id": "valid", "state": {"case_key": "c0000", "role": "builder", "mode": "stream",
                                            "split": "val", "full_hash": "full-c0000",
                                            "environment_fingerprint": fingerprint,
                                            "prompt": [{"role": "user", "content": "x"}], "tests": [{"q": "g"}]}}
            row["prompt"] = row["state"]["prompt"]
            Path(out).write_text(json.dumps(row) + "\n")
        elif command == "probe":
            write(out / "summary.json", {"groups": []})

    monkeypatch.setattr(runner, "_run_cli", fake_cli)
    summary = runner.run(args)
    assert [c for c, _ in calls] == ["bank", "run", "bank", "run", "export", "probe"]
    first_bank, first_run, second_bank, final_run = (a for _, a in calls[:4])
    assert "--memory-run" not in first_bank and "--collect" not in first_run
    assert second_bank[second_bank.index("--memory-run") + 1] == first_run[first_run.index("--out") + 1]
    assert second_bank[second_bank.index("--base-bank") + 1] == first_bank[first_bank.index("--out") + 1]
    assert "--defects-only" in second_bank and "--collect" in final_run
    assert summary["bank_rounds"] == 2 and summary["final_bank"].endswith("bank_r1")
