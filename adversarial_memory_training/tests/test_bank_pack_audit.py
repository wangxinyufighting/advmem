"""Offline pack/log joins preserve uncertainty and historical request content."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from admem.audit_prompts import build_payload
from admem.common import digest, read, write
from memory import FullMemory

_spec = importlib.util.spec_from_file_location(
    "bank_pack_audit", Path(__file__).resolve().parents[2] / "scripts" / "audit_bank_packs.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def make_case(tmp_path, logs, *, key="c0000", partial=False):
    bank, prepared, out = (tmp_path / name for name in ("bank", "prepared", "audit"))
    full = FullMemory.build({
        "haystack_session_ids": ["first", "second"],
        "haystack_dates": ["2024-01-01", "2024-02-01"],
        "haystack_sessions": [
            [{"role": "user", "content": "My first purchase cost $20."},
             {"role": "assistant", "content": "Noted."}],
            [{"role": "user", "content": "My second purchase cost $30."},
             {"role": "assistant", "content": "That was the blue book."},
             {"role": "user", "content": "It was a gift for my sister."}],
        ],
    })
    context = {"key": key, "question_type": "multi-session", "question_date": "2024-03-01",
               "full_hash": full.fingerprint}
    write(prepared / "cases" / key / "context.json", context)
    full.save(prepared / "cases" / key / "full.json")
    write(bank / key / ("bank_log.partial.json" if partial else "bank_log.json"), logs)
    (prepared / "private_eval.jsonl").write_text("MUST NOT BE PARSED", encoding="utf-8")
    pack = SimpleNamespace(full_hash=full.fingerprint, seed_rids=["s1:r1"], rids=["s1:r1", "s2:r1"])
    payload = build_payload(full, pack, "multi-session", context["question_date"], view="compact")
    return bank, prepared, out, full, payload


def prompt(payload, system="Historical prompt that must not be regenerated."):
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)}]


def cache_entry(cache, name, payload, raw='{"items":[]}'):
    write(cache / name, {"request": {"model": "old-model", "messages": prompt(payload)},
                         "response": {"choices": [{"finish_reason": "stop", "message": {"content": raw}}]}})


def one_report(out, index):
    assert len(index["packs"]) == 1
    return read(out / index["packs"][0]["report_file"])


def test_sampling_covers_status_and_rejection_buckets_without_duplicates():
    logs = [{"pack": i, "status": status} for i, status in enumerate(["accepted", "empty", "type_infeasible"])]
    logs += [{"pack": 0, "status": "rejected", "reason": "oracle_or_type"},
             {"pack": 3, "status": "rejected", "reason": "scope_or_completeness"}]
    first = audit.select_packs(logs, per_status=1, seed=42)
    assert first == [0, 1, 2, 3]
    assert first == audit.select_packs(list(reversed(logs)), per_status=1, seed=42)


def test_trace_preserves_prior_context_raw_and_effective_request(tmp_path):
    trace_file = "pack_traces/pack_00007_attempt_00.json"
    logs = [{"pack": 7, "status": "invalid", "seed": "s1", "trace_file": trace_file}]
    bank, prepared, out, full, payload = make_case(tmp_path, logs)
    payload["audit_context"] = {"accepted_items": [{"q": "Previously accepted historical question"}]}
    policy_prompt = prompt(payload)
    effective = copy.deepcopy(payload)
    effective["n_questions"] = 2
    trace = {"schema": 1, "case_key": "c0000", "pack_index": 7, "attempt": 0,
             "full_hash": full.fingerprint, "prompt": policy_prompt, "prompt_hash": digest(policy_prompt),
             "request_body": {"model": "historical-model", "messages": prompt(effective)},
             "raw": "```broken original output", "logs": logs, "stage": "complete"}
    write(bank / "c0000" / trace_file, trace)
    index = audit.build_audits(bank, prepared, out)
    report = one_report(out, index)
    assert report["input_provenance"] == "trace"
    saved = report["traces"][0]
    assert saved["snapshot"] == trace
    assert saved["input_source"] == "effective_request"
    assert saved["payload"] == effective
    assert saved["payload"]["audit_context"] == payload["audit_context"]
    neighbor = next(r for r in saved["source_comparison"] if r["rid"] == "s2:r1")
    assert neighbor["content_hidden"] is True
    assert neighbor["visible_messages"] == [{"role": "user", "content": "My second purchase cost $30."}]
    assert neighbor["adjacent_unseen"][0]["rid"] == "s2:r2"


def test_legacy_ambiguous_cache_candidates_remain_unverified_and_wrong_case_excluded(tmp_path):
    bank, prepared, out, _, payload = make_case(tmp_path, [{"pack": 3, "status": "empty", "seed": "s1"}])
    cache = tmp_path / "cache"
    cache_entry(cache, "a.json", payload)
    second = copy.deepcopy(payload)
    second["audit_context"] = {"accepted_items": [{"q": "Different prior context"}]}
    cache_entry(cache, "b.json", second)
    wrong = copy.deepcopy(payload)
    wrong["rounds"][0]["messages"][0]["content"] = "A different case's private history."
    cache_entry(cache, "wrong_case.json", wrong)
    cache_entry(cache, "wrong_output.json", payload, '{"items":[{"q":"not empty"}]}')
    report = one_report(out, audit.build_audits(bank, prepared, out, cache=cache))
    assert report["input_provenance"] == "cache_candidates_unverified"
    assert len(report["cache_candidates"]) == 2
    assert {Path(c["cache_file"]).name for c in report["cache_candidates"]} == {"a.json", "b.json"}
    assert report["traces"] == []


def test_unique_cache_match_with_repaired_evidence_is_still_unverified(tmp_path):
    question = {"q": "How much did both purchases cost?", "a": "$50", "type": "multi-session",
                "question_date": "2024-03-01", "E": ["s1:r1", "s2:r1"]}
    logs = [{"pack": 0, "status": "accepted", "seed": "s1", "item": question, "repaired": True}]
    bank, prepared, out, _, payload = make_case(tmp_path, logs)
    cache = tmp_path / "cache"
    original = {**question, "E": ["s1:r1"]}
    cache_entry(cache, "match.json", payload, json.dumps({"items": [original]}))
    report = one_report(out, audit.build_audits(bank, prepared, out, cache=cache))
    assert report["input_provenance"] == "cache_candidates_unverified"
    assert len(report["cache_candidates"]) == 1
    assert json.loads(report["cache_candidates"][0]["raw"])["items"][0]["E"] == ["s1:r1"]
    assert report["bank_log"][0]["item"]["E"] == ["s1:r1", "s2:r1"]


def test_partial_logs_missing_cache_and_skipped_pack_do_not_invent_inputs(tmp_path):
    logs = [{"pack": 0, "status": "transport_error"}, {"pack": 1, "status": "type_infeasible"}]
    bank, prepared, out, _, _ = make_case(tmp_path, logs, partial=True)
    index = audit.build_audits(bank, prepared, out, cache=tmp_path / "missing_cache")
    assert index["cache_status"] == "missing_directory"
    reports = {p["pack"]: read(out / p["report_file"]) for p in index["packs"]}
    assert reports[0]["input_provenance"] == "unavailable"
    assert reports[1]["input_provenance"] == "not_sent"
    assert all(not r["traces"] and not r["cache_candidates"] for r in reports.values())


def test_changed_prepared_history_does_not_falsely_compare_against_trace(tmp_path):
    bank, prepared, out, _, payload = make_case(tmp_path, [{"pack": 0, "status": "empty"}])
    write(bank / "c0000" / "pack_traces" / "pack_00000_attempt_00.json", {
        "case_key": "c0000", "pack_index": 0, "full_hash": "older-history-hash",
        "prompt": prompt(payload), "raw": '{"items":[]}', "stage": "complete"})
    report = one_report(out, audit.build_audits(bank, prepared, out))
    assert report["input_provenance"] == "trace_candidates_unverified"
    assert report["traces"][0]["full_hash_matches"] is False
    assert report["traces"][0]["source_comparison"] == []


def test_linked_trace_excludes_stale_retry_from_reused_directory(tmp_path):
    trace_file = "pack_traces/pack_00000_attempt_01.json"
    logs = [{"pack": 0, "status": "empty", "trace_file": trace_file}]
    bank, prepared, out, full, payload = make_case(tmp_path, logs)
    base = {"case_key": "c0000", "pack_index": 0, "full_hash": full.fingerprint,
            "prompt": prompt(payload), "prompt_hash": digest(prompt(payload)),
            "environment_fingerprint": "env", "nonce": "current", "raw": '{"items":[]}'}
    for attempt, nonce in [(0, "current"), (1, "current"), (2, "old")]:
        write(bank / "c0000" / "pack_traces" / f"pack_00000_attempt_{attempt:02d}.json",
              {**base, "attempt": attempt, "nonce": nonce})
    report = one_report(out, audit.build_audits(bank, prepared, out))
    assert report["input_provenance"] == "trace"
    assert [t["snapshot"]["attempt"] for t in report["traces"]] == [0, 1]


def test_skipped_trace_records_diagnostics_without_claiming_an_input_was_sent(tmp_path):
    bank, prepared, out, full, _ = make_case(tmp_path, [{"pack": 0, "status": "type_infeasible"}])
    write(bank / "c0000" / "pack_traces" / "pack_00000_attempt_00.json", {
        "case_key": "c0000", "pack_index": 0, "full_hash": full.fingerprint,
        "prompt": None, "request_body": None, "raw": None, "stage": "skipped"})
    report = one_report(out, audit.build_audits(bank, prepared, out))
    assert report["input_provenance"] == "not_sent"
    assert len(report["traces"]) == 1
    assert report["traces"][0]["payload"] is None


def test_tampered_trace_prompt_fails_instead_of_claiming_exact_input(tmp_path):
    bank, prepared, out, full, payload = make_case(tmp_path, [{"pack": 0, "status": "empty"}])
    write(bank / "c0000" / "pack_traces" / "pack_00000_attempt_00.json", {
        "case_key": "c0000", "pack_index": 0, "full_hash": full.fingerprint,
        "prompt": prompt(payload), "prompt_hash": "wrong", "raw": '{"items":[]}'})
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        audit.build_audits(bank, prepared, out)
    assert not out.exists()
