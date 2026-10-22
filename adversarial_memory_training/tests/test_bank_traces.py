"""Bank audit traces retain actual policy inputs and every retry offline."""
from copy import deepcopy
import json

import pytest

from conftest import Pack
from admem.common import Unknown, digest, read
from admem.pipeline import _bank_pack, generate_bank


def pack_for(full):
    return Pack("p1", full.fingerprint, "seed", "session", 0, ["s1:r1"], list(full.rounds))


def question(text="What is my cat's name?"):
    return {"q": text, "a": "Milo", "type": "single-session-user",
            "question_date": "2023-04-01", "E": ["s1:r1"]}


def test_trace_snapshots_exact_policy_prompt_prior_context_and_request_body(setup_case, env, tmp_path, monkeypatch):
    full, path, context = setup_case
    trace_path = tmp_path / "pack_traces" / "pack_00003_attempt_00.json"
    prior = [question("What name did I give my cat?")]
    captured = {}
    remote = env.policy("ATTACKER")

    def complete(prompt, nonce, temperature=0):
        before = read(trace_path)
        assert before["stage"] == "prepared" and before["raw"] is None
        assert before["prompt"] == prompt
        captured["prompt"] = deepcopy(prompt)
        remote.last_request = {"model": "test", "messages": deepcopy(prompt), "temperature": temperature}
        return '  {"items": []}\n'

    monkeypatch.setattr(remote, "complete", complete)
    info, logs, added = _bank_pack(full, path, context, pack_for(full), 3, prior, env,
                                  "ATTACKER", trace_path=trace_path)
    trace = read(trace_path)
    assert trace["stage"] == "complete" and trace["raw"] == '  {"items": []}\n'
    assert trace["prompt"] == captured["prompt"]
    assert trace["prompt_hash"] == digest(captured["prompt"])
    assert json.loads(trace["prompt"][1]["content"])["audit_context"]["accepted_items"] == prior
    assert trace["request_body"] == remote.last_request
    assert trace["logs"] == logs and trace["added_items"] == added == []
    assert info["trace_file"] == "pack_traces/pack_00003_attempt_00.json"
    assert logs[0]["trace_file"] == info["trace_file"]
    assert "TARGET_CANARY" not in json.dumps(trace)


def test_invalid_raw_is_retained_verbatim(setup_case, env, tmp_path, monkeypatch):
    full, path, context = setup_case
    trace_path = tmp_path / "trace.json"
    monkeypatch.setattr(env.policy("ATTACKER"), "complete", lambda *a, **kw: "invalid [TRUNCATED_OUTPUT]")
    _, logs, added = _bank_pack(full, path, context, pack_for(full), 0, [], env,
                               "ATTACKER", trace_path=trace_path)
    trace = read(trace_path)
    assert trace["stage"] == "invalid"
    assert trace["raw"] == "invalid [TRUNCATED_OUTPUT]"
    assert trace["logs"] == logs and logs[0]["status"] == "invalid" and added == []


def test_gate_error_preserves_raw_and_completed_judgments_without_commit(setup_case, env, tmp_path, monkeypatch):
    full, path, context = setup_case
    trace_path = tmp_path / "trace.json"
    items = [question(), question("Which name belongs to my cat?")]
    raw = json.dumps({"items": items})
    monkeypatch.setattr(env.policy("ATTACKER"), "complete", lambda *a, **kw: raw)
    calls = []

    def gate(full, pack, item, date, qtype):
        calls.append(item)
        if len(calls) == 2:
            raise Unknown("judge unavailable")
        return {"status": "accepted", "item": item}

    monkeypatch.setattr(env, "gate", gate)
    accepted = []
    with pytest.raises(Unknown, match="judge unavailable"):
        _bank_pack(full, path, context, pack_for(full), 0, accepted, env,
                   "ATTACKER", trace_path=trace_path)
    trace = read(trace_path)
    assert trace["stage"] == "error" and trace["failed_stage"] == "gating"
    assert trace["raw"] == raw and trace["logs"][0]["item"] == items[0]
    assert trace["added_items"] == [items[0]] and accepted == []


def test_generate_bank_retains_each_retry_and_links_successful_attempt(setup_case, env, tmp_path, monkeypatch):
    full, path, context = setup_case
    monkeypatch.setattr("admem.pipeline.make_pool", lambda *a: [pack_for(full)])
    monkeypatch.setenv("PACK_RETRIES", "2")
    monkeypatch.setenv("PACK_BACKOFF", "0")
    remote, calls = env.policy("ATTACKER"), []

    def complete(prompt, nonce, temperature=0):
        calls.append(nonce)
        remote.last_request = {"messages": prompt, "model": "test"}
        if len(calls) == 1:
            raise Unknown("first request failed")
        return '{"items": []}'

    monkeypatch.setattr(remote, "complete", complete)
    out = tmp_path / "bank"
    assert generate_bank(full, path, context, env, out) == 0
    traces = sorted((out / "pack_traces").glob("*.json"))
    assert len(traces) == 2
    failed, succeeded = map(read, traces)
    assert failed["stage"] == "error" and failed["raw"] is None
    assert failed["request_body"] is not None and failed["attempt"] == 0
    assert succeeded["stage"] == "complete" and succeeded["attempt"] == 1
    assert read(out / "bank_log.json")[0]["trace_file"] == "pack_traces/pack_00000_attempt_01.json"


def test_skipped_pack_has_no_model_prompt_or_request(setup_case, env, tmp_path):
    full, path, context = setup_case
    context = {**context, "question_type": "multi-session"}
    pack = pack_for(full)
    pack.rids = pack.seed_rids
    env.cfg.attacker_type_filter = True
    trace_path = tmp_path / "trace.json"
    _, logs, added = _bank_pack(full, path, context, pack, 0, [], env,
                               "ATTACKER", trace_path=trace_path)
    trace = read(trace_path)
    assert trace["stage"] == "skipped"
    assert trace["prompt"] is trace["raw"] is trace["request_body"] is None
    assert logs[0]["status"] == "type_infeasible" and added == []
    assert not env.control["prompts"]


def test_without_trace_path_no_trace_io(setup_case, env, monkeypatch):
    full, path, context = setup_case
    monkeypatch.setattr("admem.pipeline.write", lambda *a, **kw: pytest.fail("unexpected trace write"))
    _, logs, _ = _bank_pack(full, path, context, pack_for(full), 0, [], env, "ATTACKER")
    assert all("trace_file" not in row for row in logs)
