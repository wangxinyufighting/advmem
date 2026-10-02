"""新增批量代码的离线测试。模型、检索分数均mock，不报告任何真实模型能力。"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import _benchmark as shared
import benchmark_run as reader
import benchmark_attack as attacker
from common import digest, read_json
from memory import FullMemory
from metrics import Gold
from packs import Pack, TYPES


def case_data(qid="demo", qtype="multi-session", labels=True):
    return {"question_id": qid, "question_type": qtype,
            "question": "OFFICIAL_TARGET_DO_NOT_LEAK", "answer": "SECRET_GOLD_ANSWER",
            "question_date": "2023-09-01", "answer_session_ids": ["early", "late"] if labels else [],
            "haystack_session_ids": ["early", "late"], "haystack_dates": ["2023-03-01", "2023-08-01"],
            "haystack_sessions": [[{"role": "user", "content": "I adopted a cat Milo in March.", "has_answer": labels},
                                   {"role": "assistant", "content": "Congratulations on adopting Milo."}],
                                  [{"role": "user", "content": "I adopted a dog Rex in August.", "has_answer": labels},
                                   {"role": "assistant", "content": "Congratulations on adopting Rex."}]]}


def make_data(tmp_path, cases=None):
    p = tmp_path / "data.json"
    shared.save(p, cases or [case_data()])
    return p


def make_pool(full):
    return [Pack(f"p{i:05d}", full.fingerprint, sid, "session", 0, s.rids, list(full.rounds))
            for i, (sid, s) in enumerate(full.sessions.items(), 1)]


def source_run(tmp_path, data):
    root = tmp_path / "packs_source"
    selected = []
    for i, case in enumerate(read_json(data)):
        meta = {"case_index": i, "question_id": case["question_id"], "question_type": shared.kind(case)}
        selected.append(meta)
        folder = root / "cases" / f"{i:04d}"
        full = FullMemory.build(case)
        full.save(folder / "full_memory.json")
        shared.save(folder / "case_report.json", {**meta, "case_hash": digest(case)})
        shared.jsonl(folder / "packs.jsonl", [p.to_dict() for p in make_pool(full)])
    shared.save(root / "selected_cases.json", selected)
    return root


@dataclass
class Hit:
    id: str
    score: float = 1.0
    channels: tuple = ("TEST_ONLY",)


class FakeClient:
    def __init__(self, role):
        self.model, self.scope = "MOCK-" + role, ""
        self.calls, self.cache_hits = 0, 0


@pytest.fixture
def engine(monkeypatch):
    """这里mock的是模型/检索，所有选样、恢复、导出和汇总仍执行真实新增代码。"""
    env = SimpleNamespace(retrieve=[], answers=[], grades=[], generated=[], gates=[],
                          answer_effect=None, grade_effect=None, generation_effect=None, gate_effect=None,
                          contexts={})

    class Index:
        def __init__(self, full):
            self.docs = {d.id: d for d in full.documents()}

        def retrieve(self, query, date, k, **kwargs):
            env.retrieve.append((query, date, k, kwargs))
            return [Hit(rid) for rid in list(self.docs)[:k]], []

        def context(self, hits, max_chars):
            chosen, text = [], ""
            for h in hits:
                part = f"[{h.id}] {self.docs[h.id].text}\n"
                if max_chars and len(text + part) > max_chars:
                    continue
                chosen.append(h)
                text += part
            return text, chosen

        def search_many(self, queries, k, **kwargs):
            excluded = kwargs.get("exclude_sessions", set())
            return [Hit(d.id) for d in self.docs.values()
                    if not set(getattr(d, "session_ids", [])) & excluded][:k]

    monkeypatch.setattr(shared.Runtime, "index", lambda self, full: Index(full))
    monkeypatch.setattr(shared.ScopedClient, "from_env", classmethod(lambda cls, role, cache: FakeClient(role)))

    def answer(model, q, date, history):
        env.answers.append({"q": q, "date": date, "history": history, "scope": model.scope})
        if env.answer_effect:
            return env.answer_effect(model, q, date, history)
        return "MODEL_RESPONSE"

    def grade(model, q, reference, prediction, qtype, abstention):
        env.grades.append({"q": q, "reference": reference, "prediction": prediction,
                           "qtype": qtype, "abstention": abstention, "scope": model.scope})
        if env.grade_effect:
            return env.grade_effect(model, q, reference, prediction, qtype, abstention)
        return {"correct": True, "reason": "test stub"}

    class Generator:
        def __init__(self, model, full, marks):
            self.model, self.full = model, full

        def generate(self, p, qtype, date, n, nonce=""):
            env.generated.append({"pack": p.to_dict(), "type": qtype, "date": date,
                                  "n": n, "nonce": nonce, "scope": self.model.scope})
            if env.generation_effect:
                return env.generation_effect(p, qtype, date, n)
            return [{"q": "GENERATED_QUESTION_" + p.pack_id, "a": "GENERATED_ANSWER", "type": qtype,
                     "question_date": date, "E": list(p.rids)}]

    def gate(item, p, full, index, judge, defender, date, qtype, mode, max_chars):
        env.gates.append({"item": copy.deepcopy(item), "scope": judge.scope})
        if env.gate_effect:
            return env.gate_effect(item, p, full)
        return {"generated": copy.deepcopy(item), "item": copy.deepcopy(item), "status": "accepted"}

    monkeypatch.setattr(reader, "answer", answer)
    monkeypatch.setattr(reader, "grade", grade)
    monkeypatch.setattr(attacker, "Attacker", Generator)
    monkeypatch.setattr(attacker, "gate", gate)
    return env


def command(data, out, source=None):
    args = ["--data", str(data), "--out", str(out), "--embedding", "none"]
    return args + (["--packs-dir", str(source)] if source else [])


def group(path, name="ALL_MICRO"):
    return next(r for r in read_json(path)["groups"] if r["group"] == name)


def item(rids, q="query"):
    return {"q": q, "a": "answer", "question_date": "2023-09-01", "type": "multi-session", "E": rids}


def pack_state(items=None, statuses=None, error=False):
    if error:
        return {"generation": {"status": "error", "error": "synthetic failure"}}
    items = items or []
    statuses = statuses or ["accepted"] * len(items)
    return {"generation": {"status": "ok", "items": items},
            "results": [{"generated": x, "item": x, "status": s} for x, s in zip(items, statuses)]}


def test_bool_stats_keeps_unknown():
    r = shared.bool_stats([True, False, None, "true"])
    assert r["completed"] == 2 and r["unknown"] == 2 and r["rate_completed"] == 0.5
    assert r["lower_all_selected"] == 0.25 and r["upper_all_selected"] == 0.75


def test_empty_stats_are_not_perfect():
    assert shared.bool_stats([])["rate_completed"] is None


def test_selection_matches_hash_sampling(tmp_path):
    cases = [case_data(f"q{i}", TYPES[i % 6]) for i in range(24)]
    data = make_data(tmp_path, cases)
    p = shared.common_args("test")
    args = p.parse_args(command(data, tmp_path / "out") + ["--per-type", "2"])
    chosen = shared.selection(args)
    for t in TYPES:
        ids = [r["question_id"] for r in chosen if r["question_type"] == t]
        expected = sorted([c["question_id"] for c in cases if c["question_type"] == t],
                          key=lambda q: digest([0, "case-selection", q]))[:2]
        assert set(ids) == set(expected)
    assert len(chosen) == 12


def test_packs_dir_preserves_all_selected_not_per_type(tmp_path):
    data = make_data(tmp_path, [case_data("a"), case_data("b")])
    source = source_run(tmp_path, data)
    args = shared.common_args("t").parse_args(command(data, tmp_path / "o", source) + ["--per-type", "1"])
    assert len(shared.selection(args)) == 2


def test_selection_lists_abstention_separately(tmp_path):
    data = make_data(tmp_path, [case_data(), case_data("q_abs", TYPES[0])])
    args = shared.common_args("t").parse_args(command(data, tmp_path / "o") + ["--types", "abstention"])
    assert [r["question_type"] for r in shared.selection(args)] == ["abstention"]


def test_selection_missing_ids_rejected(tmp_path):
    data = make_data(tmp_path)
    ids = tmp_path / "ids.json"
    shared.save(ids, ["missing"])
    args = shared.common_args("t").parse_args(command(data, tmp_path / "o") + ["--ids", str(ids)])
    with pytest.raises(ValueError):
        shared.selection(args)


def test_reader_end_to_end_and_no_answer_leak(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    source = source_run(tmp_path, data)
    reader.main(command(data, out, source))
    r = group(out / "summary.json")
    assert r["accuracy_completed"] == 1 and r["all_rounds_retrieved_rate"] == 1
    assert r["judged"] == 1 and r["correct"] == 1
    assert "SECRET_GOLD_ANSWER" not in json.dumps(engine.answers)
    assert engine.grades[0]["reference"] == "SECRET_GOLD_ANSWER"
    assert len(shared.read_lines(out / "predictions.jsonl")) == 1


def test_reader_resume_completed_calls_nothing(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    args = command(data, out)
    reader.main(args)
    calls = len(engine.answers), len(engine.grades), len(engine.retrieve)
    reader.main(args + ["--resume"])
    assert calls == (len(engine.answers), len(engine.grades), len(engine.retrieve))


def test_reader_judge_retry_preserves_answer(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    args = command(data, out)
    engine.grade_effect = lambda *x: (_ for _ in ()).throw(RuntimeError("test judge failed"))
    with pytest.raises(SystemExit) as exc:
        reader.main(args)
    assert exc.value.code == 2
    assert group(out / "summary.json")["unknown"] == 1
    first_scope = engine.grades[0]["scope"]
    engine.grade_effect = None
    reader.main(args + ["--resume"])
    assert len(engine.answers) == 1 and len(engine.retrieve) == 1 and len(engine.grades) == 2
    assert engine.grades[-1]["scope"] != first_scope
    assert group(out / "summary.json")["correct"] == 1


def test_reader_answer_retry_preserves_retrieval(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    args = command(data, out)
    engine.answer_effect = lambda *x: (_ for _ in ()).throw(RuntimeError("test reader failed"))
    with pytest.raises(SystemExit):
        reader.main(args)
    assert group(out / "summary.json")["answered"] == 0
    engine.answer_effect = None
    reader.main(args + ["--resume"])
    assert len(engine.retrieve) == 1 and len(engine.answers) == 2


def test_reader_judge_false_is_not_error(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    engine.grade_effect = lambda *x: {"correct": False}
    reader.main(command(data, out))
    r = group(out / "summary.json")
    assert r["incorrect"] == 1 and r["error_cases"] == 0 and r["accuracy_completed"] == 0


def test_reader_nonbool_judge_is_error(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    engine.grade_effect = lambda *x: {"correct": "true"}
    with pytest.raises(SystemExit):
        reader.main(command(data, out))
    assert group(out / "summary.json")["unknown"] == 1


def test_reader_visible_context_budget_not_candidate_coverage(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    reader.main(command(data, out) + ["--context-chars", "1"])
    assert group(out / "summary.json")["mean_round_recall"] == 0
    assert engine.answers[0]["history"] == ""


def test_reader_retrieval_only_has_no_accuracy(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    reader.main(command(data, out) + ["--retrieval-only"])
    assert group(out / "summary.json")["accuracy_completed"] is None
    assert not engine.answers and not engine.grades


def test_reader_abstention_accuracy_without_positive_recall(tmp_path, engine):
    data = make_data(tmp_path, [case_data("q_abs", TYPES[0], labels=False)])
    out = tmp_path / "reader"
    reader.main(command(data, out) + ["--types", "abstention"])
    r = group(out / "summary.json")
    assert r["accuracy_completed"] == 1 and r["all_rounds_retrieved_rate"] is None
    assert engine.grades[0]["abstention"] is True


def test_reader_interrupt_exports_completed_answer(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    engine.grade_effect = lambda *x: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(SystemExit) as exc:
        reader.main(command(data, out))
    assert exc.value.code == 130
    assert len(shared.read_lines(out / "predictions.jsonl")) == 1
    engine.grade_effect = None
    reader.main(command(data, out) + ["--resume"])
    assert len(engine.answers) == 1


def test_different_config_rejected_before_calls(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    reader.main(command(data, out))
    with pytest.raises(ValueError):
        reader.main(command(data, out) + ["--resume", "--k", "1"])
    assert len(engine.answers) == 1


def test_existing_dir_requires_explicit_resume(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    reader.main(command(data, out))
    with pytest.raises(ValueError):
        reader.main(command(data, out))


def test_original_pack_case_hash_mismatch_is_error(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "reader"
    source = source_run(tmp_path, data)
    changed = read_json(data)
    changed[0]["answer"] = "CHANGED_LABEL"
    shared.save(data, changed)
    with pytest.raises(SystemExit):
        reader.main(command(data, out, source))
    assert not engine.answers
    assert "case_hash" in read_json(out / "cases/0000/case_report.json")["error"]


def test_reader_list_calls_no_models(tmp_path, engine):
    reader.main(command(make_data(tmp_path), tmp_path / "r") + ["--list"])
    assert not engine.retrieve and not engine.answers


def test_full_pool_validation_rejects_missing_seed():
    full = FullMemory.build(case_data())
    with pytest.raises(ValueError):
        attacker.inspect_pool(make_pool(full)[:1], full)


def test_full_pool_validation_rejects_duplicate_seed():
    full = FullMemory.build(case_data())
    p = make_pool(full)
    with pytest.raises(ValueError):
        attacker.inspect_pool([p[0], p[0], p[1]], full)


def test_prefix_same_evidence_union_not_single_question():
    case = case_data()
    full, gold = FullMemory.build(case), None
    gold = Gold.from_case(case, full)
    pool = make_pool(full)
    states = [pack_state([item(["s1:r1"])]), pack_state([item(["s2:r1"])])]
    report = attacker.prefix_metrics(pool, states, full, gold)
    last = report["curve"][-1]["stages"]["accepted"]
    assert last["any_question_all_rounds"] is False
    assert last["question_E_union_all_rounds"] is True
    assert report["min_observed_prefix_N"]["accepted"]["question_E_union_all_rounds"] == 2


def test_gate_enrichment_not_credited_to_original():
    case = case_data()
    full = FullMemory.build(case)
    pool = make_pool(full)[:1]
    original, final = item(["s1:r1"]), item(["s1:r1", "s2:r1"])
    state = pack_state([original])
    state["results"][0]["item"] = final
    stages = attacker.prefix_metrics(pool, [state], full, Gold.from_case(case, full))["curve"][0]["stages"]
    assert stages["raw"]["any_question_all_rounds"] is False
    assert stages["accepted_original"]["any_question_all_rounds"] is False
    assert stages["accepted"]["any_question_all_rounds"] is True


def test_generation_error_is_unknown_but_known_hit_wins():
    case = case_data()
    full = FullMemory.build(case)
    pool = make_pool(full)
    states = [pack_state(error=True), pack_state([])]
    stages = attacker.prefix_metrics(pool, states, full, Gold.from_case(case, full))["curve"][-1]["stages"]
    assert stages["raw"]["any_question_all_rounds"] is None
    states[1] = pack_state([item(list(full.rounds))])
    stages = attacker.prefix_metrics(pool, states, full, Gold.from_case(case, full))["curve"][-1]["stages"]
    assert stages["raw"]["any_question_all_rounds"] is True
    assert stages["accepted"]["any_question_all_rounds"] is True


def test_gate_error_keeps_raw_result_but_accepted_unknown():
    case = case_data()
    full = FullMemory.build(case)
    pool = make_pool(full)[:1]
    state = pack_state([item(list(full.rounds))], ["error"])
    stages = attacker.prefix_metrics(pool, [state], full, Gold.from_case(case, full))["curve"][0]["stages"]
    assert stages["raw"]["any_question_all_rounds"] is True
    assert stages["accepted"]["any_question_all_rounds"] is None


def test_empty_questions_known_no_hit():
    case = case_data()
    full = FullMemory.build(case)
    result = attacker.prefix_metrics(make_pool(full), [pack_state([]), pack_state([])], full, Gold.from_case(case, full))
    assert result["curve"][-1]["stages"]["accepted"]["any_question_all_rounds"] is False
    assert result["curve"][-1]["counts"]["empty_packs"] == 2


def test_invalid_raw_evidence_cannot_game_coverage():
    case = case_data()
    full = FullMemory.build(case)
    candidate = item([*full.rounds, "FAKE_RID"])
    result = attacker.prefix_metrics(make_pool(full)[:1], [pack_state([candidate], ["rejected"])], full,
                                     Gold.from_case(case, full))
    raw = result["curve"][0]["stages"]["raw"]
    assert raw["any_question_all_rounds"] is False and raw["invalid_evidence_questions"] == 1


def test_missing_labels_unknown_not_perfect():
    case = case_data(labels=False)
    full = FullMemory.build(case)
    r = attacker.prefix_metrics(make_pool(full), [pack_state([]), pack_state([])], full, Gold.from_case(case, full))
    assert r["curve"][-1]["stages"]["raw"]["any_question_all_rounds"] is None


def test_attack_full_pipeline_and_no_target_in_model_inputs(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "attack"
    source = source_run(tmp_path, data)
    attacker.main(command(data, out, source) + ["--n-grid", "1", "2"])
    report = read_json(out / "cases/0000/case_report.json")
    assert report["full_pool_used"] is True and report["status"] == "ok"
    assert report["counts"]["accepted"] == 2
    payload = json.dumps(engine.generated + engine.gates)
    assert "OFFICIAL_TARGET_DO_NOT_LEAK" not in payload and "SECRET_GOLD_ANSWER" not in payload
    assert len(engine.generated) == 2
    assert (out / "cases/0000/full_memory.json").exists()
    assert len(shared.read_lines(out / "cases/0000/questions.jsonl")) == 2


def test_attack_completed_resume_no_model_calls(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "attack"
    source = source_run(tmp_path, data)
    args = command(data, out, source)
    attacker.main(args)
    before = len(engine.generated), len(engine.gates)
    attacker.main(args + ["--resume"])
    assert before == (len(engine.generated), len(engine.gates))


def test_attack_gate_retry_does_not_regenerate_or_repeat_good_items(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "attack"
    source = source_run(tmp_path, data)
    args = command(data, out, source)
    def generation(p, t, d, n):
        return [item(list(p.rids), p.pack_id + ":0"), item(list(p.rids), p.pack_id + ":1")]
    engine.generation_effect = generation
    failed = {"done": False}
    def gate_effect(x, p, f):
        if p.pack_id == "p00001" and x["q"].endswith(":0") and not failed["done"]:
            failed["done"] = True
            return {"status": "error", "generated": x, "item": x, "reason": "test gate timeout"}
        return {"status": "accepted", "generated": x, "item": x}
    engine.gate_effect = gate_effect
    with pytest.raises(SystemExit):
        attacker.main(args)
    assert len(engine.generated) == 2 and len(engine.gates) == 4
    first_scope = engine.gates[0]["scope"]
    attacker.main(args + ["--resume"])
    assert len(engine.generated) == 2 and len(engine.gates) == 5
    assert engine.gates[-1]["scope"] != first_scope
    assert read_json(out / "cases/0000/case_report.json")["counts"]["accepted"] == 4


def test_attack_generation_error_retries_only_failed_pack(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "attack"
    source = source_run(tmp_path, data)
    args = command(data, out, source)
    failed = {"done": False}
    def generation(p, t, d, n):
        if p.pack_id == "p00001" and not failed["done"]:
            failed["done"] = True
            raise RuntimeError("synthetic generation failure")
        return [item(p.rids)]
    engine.generation_effect = generation
    with pytest.raises(SystemExit):
        attacker.main(args)
    attacker.main(args + ["--resume"])
    assert len(engine.generated) == 3
    assert engine.generated[-1]["pack"]["pack_id"] == "p00001"
    assert len(engine.gates) == 2


def test_attack_interrupted_gate_resumes_generated_items(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "attack"
    source = source_run(tmp_path, data)
    args = command(data, out, source)
    engine.gate_effect = lambda *x: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(SystemExit) as exc:
        attacker.main(args)
    assert exc.value.code == 130
    assert len(engine.generated) == 1
    engine.gate_effect = None
    attacker.main(args + ["--resume"])
    assert len(engine.generated) == 2  # 第一pack不重出题，只生成之前未处理的第二pack。
    assert read_json(out / "cases/0000/case_report.json")["status"] == "ok"


def test_attack_max_packs_explicitly_not_full_pool(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "attack"
    source = source_run(tmp_path, data)
    attacker.main(command(data, out, source) + ["--max-packs", "1"])
    report = read_json(out / "cases/0000/case_report.json")
    assert report["pool_size"] == 2 and report["packs_planned"] == 1
    assert report["full_pool_used"] is False


def test_attack_oversized_pack_is_error_not_empty(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "attack"
    source = source_run(tmp_path, data)
    with pytest.raises(SystemExit):
        attacker.main(command(data, out, source) + ["--pack-chars", "1"])
    report = read_json(out / "cases/0000/case_report.json")
    assert report["counts"]["generation_errors"] == 2
    assert not engine.generated
    assert report["curve"][-1]["stages"]["accepted"]["any_question_all_rounds"] is None


def test_attack_missing_source_pool_is_explicit_error(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "attack"
    source = source_run(tmp_path, data)
    (source / "cases/0000/packs.jsonl").unlink()
    with pytest.raises(SystemExit):
        attacker.main(command(data, out, source))
    assert not engine.generated
    assert read_json(out / "cases/0000/case_report.json")["status"] == "error"


def test_attack_abstention_rejected_before_model_calls(tmp_path, engine):
    data = make_data(tmp_path, [case_data("q_abs")])
    with pytest.raises(SystemExit):
        attacker.main(command(data, tmp_path / "a") + ["--types", "abstention"])
    assert not engine.generated


def test_attack_standalone_constructs_full_pool(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "attack"
    attacker.main(command(data, out))
    assert read_json(out / "cases/0000/case_report.json")["pool_size"] == 2


def test_attack_summary_unknown_not_false():
    record = {"question_type": TYPES[0], "status": "error"}
    summary = attacker.summarize([record], [5])
    r = next(x for x in summary if x["group"] == "ALL_MICRO" and x["N"] == 5 and x["stage"] == "accepted")
    assert r["any_question_all_rounds_unknown"] == 1
    assert r["any_question_all_rounds_rate_completed"] is None
    assert r["any_question_all_rounds_lower_all_selected"] == 0
    assert r["any_question_all_rounds_upper_all_selected"] == 1


def test_reader_multitype_and_macro_reporting(tmp_path, engine):
    data = make_data(tmp_path, [case_data("q1", TYPES[0]), case_data("q2", TYPES[1])])
    out = tmp_path / "r"
    reader.main(command(data, out))
    assert group(out / "summary.json")["selected"] == 2
    assert group(out / "summary.json", "ANSWERABLE_MACRO")["included_types"] == 2


def test_reader_budget_exhaustion_pauses_pending_not_wrong(tmp_path, engine):
    data = make_data(tmp_path, [case_data("q1"), case_data("q2")])
    out = tmp_path / "reader"
    engine.grade_effect = lambda *x: (_ for _ in ()).throw(shared.BudgetExhausted("test budget"))
    with pytest.raises(SystemExit) as exc:
        reader.main(command(data, out))
    assert exc.value.code == 2
    r = group(out / "summary.json")
    assert r["unknown"] == 2 and r["error_cases"] == 0
    assert len(engine.answers) == 1
    engine.grade_effect = None
    reader.main(command(data, out) + ["--resume"])
    assert group(out / "summary.json")["correct"] == 2
    assert len(engine.answers) == 2


def test_attack_budget_exhaustion_keeps_pending_gate(tmp_path, engine):
    data, out = make_data(tmp_path), tmp_path / "attack"
    source = source_run(tmp_path, data)
    engine.gate_effect = lambda *x: (_ for _ in ()).throw(shared.BudgetExhausted("test budget"))
    with pytest.raises(SystemExit):
        attacker.main(command(data, out, source))
    report = read_json(out / "cases/0000/case_report.json")
    assert report["status"] == "pending"
    assert not report["counts"].get("gate_errors")
    assert report["counts"]["pending_questions"] == 1
    engine.gate_effect = None
    attacker.main(command(data, out, source) + ["--resume"])
    assert len(engine.generated) == 2
    assert read_json(out / "cases/0000/case_report.json")["status"] == "ok"


def test_scoped_client_changes_only_cache_nonce(monkeypatch):
    calls = []
    def original(self, system, data, **kwargs):
        calls.append((system, data, kwargs))
        return {"ok": True}
    monkeypatch.setattr(shared.Client, "json", original)
    client = object.__new__(shared.ScopedClient)
    client.scope = "attempt:1"
    client.json("system", {"history": "same"}, nonce="original")
    client.scope = "attempt:2"
    client.json("system", {"history": "same"}, nonce="original")
    assert calls[0][:2] == calls[1][:2]
    assert calls[0][2]["nonce"] != calls[1][2]["nonce"]


def test_scoped_client_converts_original_budget_error(monkeypatch):
    def original(*args, **kwargs):
        raise RuntimeError("mock-model 达到 API 调用预算 10000")
    monkeypatch.setattr(shared.Client, "json", original)
    client = object.__new__(shared.ScopedClient)
    with pytest.raises(shared.BudgetExhausted):
        client.json("system", {})


def test_model_change_rejects_mixed_resume(tmp_path, engine, monkeypatch):
    data, out = make_data(tmp_path), tmp_path / "reader"
    monkeypatch.setenv("LLM_MODEL", "version1")
    reader.main(command(data, out))
    monkeypatch.setenv("LLM_MODEL", "version2")
    with pytest.raises(ValueError):
        reader.main(command(data, out) + ["--resume"])


def test_api_key_not_recorded_and_budget_change_allowed(tmp_path, engine, monkeypatch):
    data, out = make_data(tmp_path), tmp_path / "reader"
    monkeypatch.setenv("LLM_API_KEY", "SENSITIVE_TEST_KEY")
    reader.main(command(data, out))
    assert "SENSITIVE_TEST_KEY" not in (out / "config.json").read_text()
    monkeypatch.setenv("MAX_API_CALLS", "99999")
    reader.main(command(data, out) + ["--resume"])
    assert len(engine.answers) == 1
