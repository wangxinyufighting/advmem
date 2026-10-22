"""Bank diagnostics work offline, including empty and failed cases."""
import importlib.util
import json
from pathlib import Path

from admem.common import write
from memory import FullMemory

_spec = importlib.util.spec_from_file_location(
    "attacker_report", Path(__file__).resolve().parents[2] / "scripts" / "attacker_report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def make_case(tmp_path, key, *, qtype="multi-session", logs=(), bank=(), partial=False):
    root, prepared = tmp_path / "bank", tmp_path / "prepared"
    write(prepared / "cases" / key / "context.json", {"key": key, "question_type": qtype})
    full = FullMemory.build({"haystack_session_ids": ["a", "b"],
                             "haystack_dates": ["2024-01-01", "2024-02-01"],
                             "haystack_sessions": [
                                 [{"role": "user", "content": "My books cost $20."}],
                                 [{"role": "user", "content": "My second purchase cost $30."}]]})
    full.save(prepared / "cases" / key / "full.json")
    suffix = ".partial.json" if partial else ".json"
    write(root / key / ("bank_log" + suffix), list(logs))
    write(root / key / ("bank" + suffix), list(bank))
    # The reporting path must never consult official benchmark questions/answers.
    (prepared / "private_eval.jsonl").write_text("DO NOT PARSE THIS", encoding="utf-8")
    return root, prepared


def test_empty_bank_uses_context_type_for_old_infeasible_logs_and_filter(tmp_path):
    root, prepared = make_case(tmp_path, "c0000", logs=[
        {"pack": 0, "status": "type_infeasible"},
        {"pack": 1, "status": "empty", "qtype": "multi-session", "prompt_tokens": 500},
    ])
    make_case(tmp_path, "c0001", qtype="single-session-user", logs=[
        {"pack": 0, "status": "type_infeasible"},
    ])
    value = report.summarize(root, prepared, ["multi-session"])
    assert value["cases"] == 1 and value["bank_questions"] == 0
    assert value["by_type"] == {"multi-session": {"type_infeasible": 1, "empty": 1}}
    assert value["case_details"][0]["pack_status_counts"] == {"empty": 1, "type_infeasible": 1}
    assert value["case_details"][0]["question_type"] == "multi-session"


def test_distinct_pack_counts_gate_reasons_and_bounded_examples(tmp_path):
    q = {"q": "How much did my two book purchases cost?", "type": "multi-session",
         "a": "$50", "question_date": "2024-03-01", "E": ["s1:r1", "s2:r1"]}
    logs = [
        {"pack": 0, "status": "rejected", "reason": "oracle_or_type", "item": q,
         "oracle": {"answerable": False, "type_valid": False, "user_relevant": True}},
        {"pack": 0, "status": "rejected", "reason": "oracle_or_type", "item": q,
         "oracle": {"answerable": False, "type_valid": True}},
        {"pack": 0, "status": "accepted", "item": q},
        {"pack": 1, "status": "rejected", "reason": "unsupported_answer", "item": q,
         "support": {"correct": False, "reason": "Wrong sum"}},
        {"pack": 2, "status": "rejected", "reason": "scope_or_completeness", "item": q,
         "screen": {"stable": False, "reason": "Missing a purchase"}},
        {"pack": 3, "status": "accepted", "item": q},
    ]
    root, prepared = make_case(tmp_path, "c0000", logs=logs, bank=[q])
    value = report.summarize(root, prepared, max_examples=2)
    case = value["case_details"][0]
    assert case["packs"] == 4
    assert case["pack_status_counts"] == {"accepted": 2, "rejected": 3}
    assert case["questions_proposed"] == 6
    assert case["gate_accepted_questions"] == 2 and case["bank_questions"] == 1
    assert case["rejection_reasons"] == {
        "oracle_or_type": 2, "unsupported_answer": 1, "scope_or_completeness": 1}
    assert case["oracle_false_flags"] == {"answerable": 2, "type_valid": 1}
    assert case["support_reasons_top10"] == [{"reason": "Wrong sum", "count": 1}]
    assert case["screen_reasons_top10"] == [{"reason": "Missing a purchase", "count": 1}]
    assert len(case["examples"]) == 2
    assert {r["reason"] for r in case["examples"]} == {"oracle_or_type", "unsupported_answer"}
    assert value["gate_accept_rate"] == .333


def test_partial_failed_case_is_visible_but_excluded_from_quality_metrics(tmp_path):
    root, prepared = make_case(tmp_path, "c0000", logs=[{"pack": 0, "status": "empty"}])
    make_case(tmp_path, "c0001", partial=True, logs=[
        {"pack": 0, "status": "transport_error", "reason": "Attacker pack exceeds token budget"},
    ])
    write(root / "c0001" / "failed.json", {"status": "unknown", "error": "Repeated service errors"})
    value = report.summarize(root, prepared, ["multi-session"])
    assert value["cases"] == 1 and value["failed_cases"] == 1
    assert value["packs"] == 1 and value["transport_error_packs"] == 0
    partial = value["case_details"][1]
    assert partial["included_in_metrics"] is False
    assert partial["pack_status_counts"] == {"transport_error": 1}
    assert partial["failure"]["error"] == "Repeated service errors"


def test_failure_before_first_pack_is_reported_without_full_or_bank(tmp_path):
    root, prepared = tmp_path / "bank", tmp_path / "prepared"
    write(root / "c0000" / "failed.json", {"status": "unknown", "error": "No server"})
    write(prepared / "cases" / "c0000" / "context.json", {"question_type": "multi-session"})
    value = report.summarize(root, prepared, ["multi-session"], max_examples=0)
    assert value["failed_cases"] == 1 and value["cases"] == 0
    case = value["case_details"][0]
    assert case["packs"] == 0 and case["bank_questions"] == 0
    assert case["examples"] == []
    json.dumps(value)
