"""salvage建记忆（按操作取舍、faith复核定位prov、局部回退）、M_old预算优先级、reader按时间排序、诊断脚本。"""
import importlib.util
import json
import re
from pathlib import Path

import pytest

from conftest import Full
from admem import prompts
from admem.cli import _builder_case_metrics
from admem.common import read, write
from admem.pipeline import builder_state, run_case, salvage_edit, windows
from admem.store import entry_date

ROOT = Path(__file__).resolve().parents[2]
STOP = {"user", "users", "the", "assistant", "that", "with"}


def words(text):
    """内容词的5字母前缀（粗略词干：enjoys/enjoy算同一个词）。"""
    return {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4 and w not in STOP}


def supports(entry_text, history_text):
    return "UNSUPPORTED" not in entry_text and words(entry_text) <= words(history_text)


@pytest.fixture
def two_rounds(tmp_path):
    case = {"haystack_session_ids": ["a", "b"], "haystack_dates": ["2023/01/01 (Sun) 10:00", "2023/02/01 (Wed) 10:00"],
            "haystack_sessions": [[{"role": "user", "content": "My cat is named Milo."},
                                   {"role": "assistant", "content": "Noted."},
                                   {"role": "user", "content": "I enjoy gardening on weekends."},
                                   {"role": "assistant", "content": "Nice."}],
                                  [{"role": "user", "content": "I moved to Denver."},
                                   {"role": "assistant", "content": "Great."}]]}
    full = Full.build(case)
    path = tmp_path / "full.json"
    full.save(path)
    context = {"key": "c0000", "split": "test", "question_type": "single-session-user", "question_date": "2023/03/01"}
    return full, path, context


@pytest.fixture
def judging(env):
    """faith/faith_locate按“条目的内容词都出现在给出的round里”判定，其余请求交给原替身。"""
    original = env.clients["JUDGE"].json

    def judge(prompt, nonce, prompt_tokens=None):
        system, payload = prompt[0]["content"], json.loads(prompt[1]["content"])
        if system == prompts.FAITH:
            entry = payload["entries"][0]
            text = " ".join(h["text"] for h in payload["history"])
            return {"faithful": supports(entry["text"], text), "reason": "stub check"}
        if system == prompts.FAITH_LOCATE:
            entry = payload["entry"]["text"]
            support = [h["rid"] for h in payload["history"] if words(entry) & words(h["text"])]
            ok = supports(entry, " ".join(h["text"] for h in payload["history"]))
            return {"faithful": ok, "support_rids": support if ok else [], "reason": "stub locate"}
        return original(prompt, nonce, prompt_tokens)

    env.clients["JUDGE"].json = judge
    env.cfg.salvage = True
    return env


def first_state(full, path, context, env, memory=None, old_ids=None):
    x = next(windows(full, env.counter, env.cfg.window_tokens))
    return builder_state(full, path, context, memory or [], x, "stream", [], env, old_ids or [])


def out(*ops):
    return json.dumps({"ops": list(ops)})


# ================================================================ salvage_edit

def test_keeps_good_entries_and_falls_back_only_for_the_dropped_round(two_rounds, judging):
    full, path, context = two_rounds
    state = first_state(full, path, context, judging)
    assert [s["rid"] for s in state["x"]] == ["s1:r1", "s1:r2"]
    memory, checks, lost = salvage_edit(full, state, out(
        {"op": "ADD", "text": "The user's cat is named Milo.", "prov": ["s1:r1"]},
        {"op": "ADD", "text": "UNSUPPORTED the user owns a boat.", "prov": ["s1:r2"]}), judging)
    assert [e["text"] for e in memory] == ["The user's cat is named Milo."]
    assert lost == {"s1:r2"} and checks["legal"] is True and checks["clean"] is False
    assert checks["faith"]["faithful"] is False
    assert [d["stage"] for d in checks["salvage"]["dropped"]] == ["faith"]


def test_wrong_provenance_is_relocated_instead_of_rejected(two_rounds, judging):
    full, path, context = two_rounds
    state = first_state(full, path, context, judging)
    memory, checks, lost = salvage_edit(full, state, out(
        {"op": "ADD", "text": "The user enjoys gardening on weekends.", "prov": ["s1:r1"]}), judging)
    assert memory[0]["prov"] == ["s1:r2"] and not lost
    assert checks["salvage"]["relocated"] == 1 and checks["clean"] is False
    assert checks["faith"]["faithful"] is False  # 严格口径：首次判定没过


def test_schema_repairs_and_per_operation_rejection(two_rounds, judging):
    full, path, context = two_rounds
    state = first_state(full, path, context, judging)
    memory, checks, lost = salvage_edit(full, state, out(
        {"op": "add", "text": "The user's cat is named Milo.", "prov": "s1:r1", "reason": "x"},
        {"op": "ADD", "text": "The user enjoys gardening on weekends.", "prov": ["s9:r9"]},
        {"op": "UPSERT", "text": "nonsense", "prov": ["s1:r1"]}), judging)
    assert checks["legal"] is False and checks["reason"]
    assert sorted(e["text"] for e in memory) == ["The user enjoys gardening on weekends.",
                                                 "The user's cat is named Milo."]
    assert {e["text"]: e["prov"] for e in memory}["The user enjoys gardening on weekends."] == ["s1:r2"]
    assert checks["salvage"]["repairs"] == {"op_name": 1, "extra_fields": 1, "prov": 1, "prov_window": 1}
    assert [d["stage"] for d in checks["salvage"]["dropped"]] == ["schema"] and lost == {"s1:r1"}


def test_unparseable_output_falls_back_to_the_whole_window(two_rounds, judging):
    full, path, context = two_rounds
    state = first_state(full, path, context, judging)
    memory, checks, lost = salvage_edit(full, state, '```json\n{"ops": []}\n```', judging)
    assert memory == [] and lost == {"s1:r1", "s1:r2"} and checks["salvage"]["unparsed"]


def test_a_failed_write_also_cancels_deletes(two_rounds, judging):
    full, path, context = two_rounds
    old = [{"id": "m_old", "text": "The user's cat is named Milo.", "prov": ["s1:r1"], "kind": "card"}]
    state = first_state(full, path, context, judging, memory=old, old_ids=["m_old"])
    memory, checks, lost = salvage_edit(full, state, out(
        {"op": "DELETE", "id": "m_old"},
        {"op": "ADD", "text": "UNSUPPORTED the cat is named Tom.", "prov": ["s1:r1"]}), judging)
    assert [e["id"] for e in memory] == ["m_old"]
    assert {d["stage"] for d in checks["salvage"]["dropped"]} == {"faith", "delete_guard"}


def test_clean_outputs_are_unchanged_and_marked_clean(two_rounds, judging):
    full, path, context = two_rounds
    state = first_state(full, path, context, judging)
    memory, checks, lost = salvage_edit(full, state, out(
        {"op": "ADD", "text": "The user's cat is named Milo.", "prov": ["s1:r1"]}), judging)
    assert checks["clean"] and checks["legal"] and checks["faith"]["faithful"] and not lost
    assert len(memory) == 1


# ================================================================ run_case with salvage

def test_run_case_salvage_metrics_and_demos(two_rounds, judging, tmp_path):
    full, path, context = two_rounds
    replies = iter([out({"op": "ADD", "text": "The user's cat is named Milo.", "prov": ["s1:r1"]},
                        {"op": "ADD", "text": "UNSUPPORTED claim about boats.", "prov": ["s1:r2"]}),
                    out({"op": "ADD", "text": "The user moved to Denver.", "prov": ["s2:r1"]})])
    judging.clients["BUILDER"].complete = lambda *a, **k: next(replies)
    from admem.pipeline import Collector
    run_case(full, path, context, judging, tmp_path / "case", mode="build", collect=Collector(tmp_path / "col"))
    memory = read(tmp_path / "case" / "M_final.json")
    assert [e["kind"] for e in memory] == ["card", "raw", "card"]
    assert memory[1]["prov"] == ["s1:r2"]
    metrics = _builder_case_metrics(tmp_path / "case")
    assert metrics["fallback_windows"] == 1 and metrics["salvage"]["partial_fallback_windows"] == 1
    assert metrics["salvage"]["ops_kept"] == 2 and metrics["salvage"]["ops_dropped"] == 1
    demos = [read(p).get("completion") for p in (tmp_path / "col").rglob("*.json")]
    assert sum(d is not None for d in demos) == 1  # 只有第二个（干净的）窗口是示范


def test_default_mode_is_unchanged(two_rounds, env, tmp_path):
    full, path, context = two_rounds
    env.cfg.salvage = False
    run_case(full, path, context, env, tmp_path / "case", mode="build")
    events = read(tmp_path / "case" / "checkpoint.json")["events"]
    assert all("salvage" not in (e.get("checks") or {}) for e in events)


# ================================================================ prompt budget and reader order

def test_editable_priority_keeps_m_old_when_the_prompt_is_tight(two_rounds, env):
    full, path, context = two_rounds
    memory = [{"id": f"m{i}", "text": f"Fact number {i} about the user's cat Milo and gardening.",
               "prov": ["s1:r1"], "kind": "card"} for i in range(30)]
    x = next(windows(full, env.counter, env.cfg.window_tokens))
    ids = [e["id"] for e in memory[:10]]
    sizes = {}
    for priority in ("catalog", "editable"):
        env.cfg.budget_priority = priority
        base = builder_state(full, path, context, memory, x, "stream", [], env, ids)
        env.cfg.builder_input_tokens = env.counter.prompt_count(base["prompt"]) - 600
        state = builder_state(full, path, context, memory, x, "stream", [], env, ids)
        payload = json.loads(state["prompt"][-1]["content"])
        sizes[priority] = (len(payload["M_old"]), len(payload["stored_catalog"]))
        env.cfg.builder_input_tokens = 20000
    assert sizes["editable"][0] == 10 and sizes["catalog"][0] < 10
    assert sizes["editable"][1] < sizes["catalog"][1]


def test_reader_time_order_sorts_selected_entries_by_source_date(env):
    memory = [{"id": "a", "text": "Recorded on 2023/05/01 (Mon) 10:00. I live in Denver.", "prov": ["s2:r1"], "kind": "raw"},
              {"id": "b", "text": "Lives in Boston.", "prov": ["s1:r1"], "kind": "card", "dates": ["2023/01/01 (Sun) 10:00"]},
              {"id": "c", "text": "Lives somewhere.", "prov": ["s1:r1"], "kind": "card"}]
    assert entry_date(memory[0]) == "2023/05/01 (Mon) 10:00" and entry_date(memory[2]) is None
    env.cfg.reader_order = "time"
    _, kept = env.context(memory, {"q": "Where do I live?"}, all_memory=True)
    assert kept == ["b", "a", "c"]
    env.cfg.reader_order = "score"
    assert env.context(memory, {"q": "Where do I live?"}, all_memory=True)[1] == ["a", "b", "c"]


# ================================================================ diagnose_builder.py

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnosis_reports_outcomes_growth_and_starved_prompts(tmp_path, capsys):
    module = load_script("diagnose_builder")
    case = tmp_path / "run" / "c0000"
    events = [{"stage": "build", "window": 0, "checks": {"legal": False, "reason": "Entry text exceeds 300 tokens (412)"},
               "fallback": True, "ops": None, "M_tokens": 1800},
              {"stage": "build", "window": 1, "checks": {"legal": True, "faith": {"faithful": False, "reason": "assistant suggestion, not user action"}},
               "fallback": True, "ops": 3, "M_tokens": 3600},
              {"stage": "build", "window": 2, "checks": {"legal": True, "faith": {"faithful": True}},
               "fallback": False, "ops": 2, "M_tokens": 3700},
              {"stage": "build", "window": 3, "checks": {"legal": False, "reason": "Entry text exceeds 300 tokens (999)"},
               "fallback": True, "ops": None, "M_tokens": 5500}]
    write(case / "checkpoint.json", {"events": events})
    write(case / "M_final.json", [{"id": "r", "text": "Recorded on x. " + "a" * 900, "kind": "raw", "prov": []},
                                  {"id": "m", "text": "card", "kind": "card", "prov": []}])
    cache = tmp_path / "api"
    cache.mkdir()
    payload = {"mode": "stream", "x": [], "M_old": [], "stored_catalog": [{"id": "m1"}],
               "editable_entry_ids": [], "allowed_source_rids": ["s2:r1"], "max_ops": 16, "max_entry_tokens": 300}
    reply = {"ops": [{"op": "ADD", "text": "t", "prov": ["m1"], "date": "x"}]}
    write(cache / "b.json", {"model": "qwen", "base_url": "u", "request": {"model": "qwen", "messages": [
        {"role": "system", "content": prompts.BUILDER_V2}, {"role": "user", "content": json.dumps(payload)}]},
        "response": {"choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 7000, "completion_tokens": 300}}})
    write(cache / "f.json", {"model": "ds", "base_url": "u", "request": {"messages": [
        {"role": "system", "content": prompts.FAITH}, {"role": "user", "content": "{}"}]},
        "response": {"choices": [{"message": {"content": json.dumps({"faithful": False, "reason": "The assistant suggested it"})}}]}})
    report = module.main(["--run", str(tmp_path / "run"), "--cache", str(cache), "--out", str(tmp_path / "d.json")])
    run = report["run"]
    assert run["outcome"]["illegal"]["windows"] == 2 and run["illegal_reasons"][0] == ("Entry text exceeds N tokens (N)", 2)
    assert run["memory_growth"]["fallback"]["windows"] == 3 and run["final_memory"]["raw_entries"] == 1
    assert report["cache"]["starved_share"] == 1.0
    problems = dict(report["cache"]["schema_problems_in_json_outputs"])
    assert problems["cites a catalog entry id"] == 1 and problems["extra field 'date'"] == 1
    assert dict(report["cache"]["faith_reason_keywords"])["speaker/attribution"] == 1
    assert any("raw fallback" in n for n in report["notes"]) and (tmp_path / "d.json").exists()


# ================================================================ judge failures and the oracle ceiling

def test_a_judge_failure_drops_only_that_entry(two_rounds, judging, monkeypatch):
    from admem.common import Unknown
    full, path, context = two_rounds
    state = first_state(full, path, context, judging)
    real = judging.faith

    def flaky(full_, changed):
        if "gardening" in changed[0]["text"]:
            raise Unknown("JUDGE JSON invalid after two attempts")
        return real(full_, changed)

    monkeypatch.setattr(judging, "faith", flaky)
    monkeypatch.setattr(judging, "faith_locate", lambda *a: (_ for _ in ()).throw(Unknown("down")))
    memory, checks, lost = salvage_edit(full, state, out(
        {"op": "ADD", "text": "The user's cat is named Milo.", "prov": ["s1:r1"]},
        {"op": "ADD", "text": "The user enjoys gardening on weekends.", "prov": ["s1:r2"]}), judging)
    assert [e["text"] for e in memory] == ["The user's cat is named Milo."] and lost == {"s1:r2"}
    assert checks["salvage"]["dropped"][0]["reason"].startswith("judge unavailable")


def test_evaluate_oracle_snapshot_gives_the_reader_only_the_evidence(case, env, tmp_path):
    import copy
    from admem.cli import evaluate
    from admem.data import prepare
    values = []
    for i in range(2):
        c = copy.deepcopy(case)
        c["question_id"] = f"q{i}"
        c["question"], c["answer"] = "What is my cat's name?", "Milo"
        c["haystack_sessions"][1][0]["content"] += str(i)
        values.append(c)
    (tmp_path / "data.json").write_text(json.dumps(values))
    prepare(tmp_path / "data.json", tmp_path / "prepared", env.memory_module, [0, 0, 2])
    seen = []
    original = env.reader
    env.reader = lambda q, history, key, **kw: seen.append(history) or original(q, history, key, **kw)
    summary = evaluate(tmp_path / "prepared", None, "test", "oracle", env, tmp_path / "eval", None, None)
    assert all("Milo" in h and "gardening" not in h for h in seen) and len(seen) == 2
    rows = [json.loads(l) for l in (tmp_path / "eval" / "cases.jsonl").read_text().splitlines()]
    assert all(r["status"] == "ok" and r["correct"] and r["entries"] == 1 for r in rows)
