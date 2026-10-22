"""Offline semantic-edit contract tests; no tokenizer/model/server required."""
import json

import pytest

from admem.common import Unknown, messages
from admem.edit_checks import COVERAGE, INVENTORY, _coverage, assess_edits
from admem.store import apply, neutral_round


def entry(eid, text, rid="s1:r1"):
    return {"id": eid, "text": text, "prov": [rid], "kind": "card"}


def fact(text, status="preserve"):
    return {"text": text, "status": status, "source_rids": [] if status == "unsupported" else ["s1:r1"],
            "reason": "explicit offline test inventory"}


def scripted_judge(env, inventory=None):
    calls = []
    inventory = inventory or {}

    def judge(system, payload, key):
        assert "TARGET_CANARY" not in json.dumps(payload)
        assert "REFERENCE_CANARY" not in json.dumps(payload)
        if env.counter.prompt_count(messages(system, payload)) > env.cfg.judge_input_tokens:
            raise Unknown("test judge budget exceeded")
        calls.append((system, payload, key))
        if system == INVENTORY:
            text = payload["entry"]["text"]
            return {"facts": inventory.get(text, [fact(text)])}
        assert system == COVERAGE
        results = []
        for f in payload["facts"]:
            ids = [e["id"] for e in payload["entries"] if f["text"] in e["text"]]
            results.append({"index": f["index"], "covered": bool(ids), "witness_ids": ids[:1],
                            "reason": "explicit offline substring witness"})
        return {"results": results}

    env.judge = judge
    return calls


def assess(env, full, memory, ops, *, mode="patch"):
    visible_sources = {rid for e in memory for rid in e["prov"]}
    new_sources = {rid for op in ops for rid in op.get("prov", [])} - visible_sources
    state = {"M": memory, "old_ids": [e["id"] for e in memory],
             "x": [neutral_round(full, rid) for rid in sorted(new_sources)],
             "tests": [{"q": "TARGET_CANARY", "a": "REFERENCE_CANARY"}], "mode": mode}
    action = {"ops": ops}
    proposed, changed = apply(memory, action, mode=mode, visible_ids=state["old_ids"],
                              source_ids=[x["rid"] for x in state["x"]], full_ids=full.rounds,
                              counter=env.counter, limit=10000)
    return assess_edits(env, full, state, action, proposed, changed)


def test_noop_needs_no_semantic_judge(setup_case, env):
    calls = scripted_judge(env)
    result = assess(env, setup_case[0], [entry("m1", "Milo")], [], mode="refine")
    assert result["retention_pass"] and result["information_loss"] == 0
    assert result["redundancy_ratio"] == 0 and result["redundant_ops"] == []
    assert calls == []


def test_provenance_only_update_is_redundant_without_judge(setup_case, env):
    calls = scripted_judge(env)
    result = assess(env, setup_case[0], [entry("m1", "Milo")],
                    [{"op": "UPDATE", "id": "m1", "text": "Milo", "prov": ["s2:r1"]}])
    assert result["retention_pass"] and result["redundancy_ratio"] == 1
    assert result["redundant_ops"][0]["reason"].startswith("unchanged_text")
    assert calls == []


def test_delete_duplicate_keeps_exact_witness_without_judge(setup_case, env):
    calls = scripted_judge(env)
    result = assess(env, setup_case[0], [entry("m1", "Milo"), entry("m2", "Milo")],
                    [{"op": "DELETE", "id": "m1"}], mode="refine")
    assert result["retention_pass"] and result["information_loss"] == 0
    assert result["retention_checks"][0]["witness_ids"] == ["m2"]
    assert calls == []


def test_delete_checks_facts_not_replay_targets(setup_case, env):
    scripted_judge(env, {"Milo in January": [fact("Milo"), fact("January")]})
    result = assess(env, setup_case[0], [entry("m1", "Milo in January"), entry("m2", "Milo")],
                    [{"op": "DELETE", "id": "m1"}])
    assert not result["retention_pass"] and result["information_loss"] == 0.5
    assert result["retention_checks"][0]["missing_fact_indices"] == [1]


@pytest.mark.parametrize("keep_replacement", [True, False])
def test_correction_requires_supported_replacement(setup_case, env, keep_replacement):
    scripted_judge(env, {"The user completed training.": [fact("planned training", "corrected")]})
    memory = [entry("m1", "The user completed training.")]
    if keep_replacement:
        memory.append(entry("m2", "The user planned training."))
    result = assess(env, setup_case[0], memory, [{"op": "DELETE", "id": "m1"}])
    assert result["retention_pass"] is keep_replacement
    assert result["information_loss"] == (0 if keep_replacement else 1)


def test_unsupported_old_claim_can_be_removed(setup_case, env):
    scripted_judge(env, {"UNSUPPORTED": [fact("UNSUPPORTED", "unsupported")]})
    result = assess(env, setup_case[0], [entry("m1", "UNSUPPORTED")], [{"op": "DELETE", "id": "m1"}])
    assert result["retention_pass"] and result["information_loss"] == 0


def test_safe_shorter_merge_is_not_redundant(setup_case, env):
    scripted_judge(env, {"The user's cat is Milo.": [fact("Milo")],
                         "The user enjoys gardening.": [fact("gardening")]})
    result = assess(env, setup_case[0], [entry("m1", "The user's cat is Milo."),
                                      entry("m2", "The user enjoys gardening.")],
                    [{"op": "MERGE", "ids": ["m1", "m2"], "text": "Milo; gardening.", "prov": []}],
                    mode="refine")
    assert result["retention_pass"] and result["redundancy_ratio"] == 0


def test_novel_fact_not_redundant_when_uncovered_by_tests(setup_case, env):
    scripted_judge(env)
    result = assess(env, setup_case[0], [entry("m1", "Milo")],
                    [{"op": "ADD", "text": "gardening", "prov": ["s1:r1"]}])
    assert result["redundancy_ratio"] == 0


def test_duplicate_within_one_action_detected(setup_case, env):
    calls = scripted_judge(env)
    result = assess(env, setup_case[0], [], [{"op": "ADD", "text": "Milo", "prov": ["s1:r1"]},
                                          {"op": "ADD", "text": "Milo", "prov": ["s1:r1"]}])
    assert result["redundancy_ratio"] == 0.5
    assert [op["op_index"] for op in result["redundant_ops"]] == [1]
    assert calls == []


def test_semantic_rewrite_without_saving_is_redundant(setup_case, env):
    scripted_judge(env, {"The cat is Milo.": [fact("Milo")]})
    result = assess(env, setup_case[0], [entry("m1", "Milo")],
                    [{"op": "UPDATE", "id": "m1", "text": "The cat is Milo.", "prov": []}])
    assert result["retention_pass"] and result["redundancy_ratio"] == 1
    assert result["redundant_ops"][0]["semantic"] is True


def test_delete_add_shorter_replacement_is_not_duplicate(setup_case, env):
    scripted_judge(env, {"The user's cat is Milo.": [fact("Milo")]})
    result = assess(env, setup_case[0], [entry("m1", "The user's cat is Milo.")],
                    [{"op": "DELETE", "id": "m1"},
                     {"op": "ADD", "text": "Milo", "prov": ["s1:r1"]}])
    assert result["retention_pass"] and result["redundancy_ratio"] == 0


def test_delete_add_identical_text_is_identity_churn(setup_case, env):
    calls = scripted_judge(env)
    result = assess(env, setup_case[0], [entry("m1", "Milo")],
                    [{"op": "DELETE", "id": "m1"},
                     {"op": "ADD", "text": "Milo", "prov": ["s1:r1"]}])
    assert result["retention_pass"] and result["redundancy_ratio"] == 1
    assert result["redundant_ops"][0]["reason"] == "no_final_text_effect"
    assert calls == []


def test_dummy_exact_update_does_not_dilute_information_loss(setup_case, env):
    scripted_judge(env, {"Milo in January": [fact("Milo"), fact("January")]})
    result = assess(env, setup_case[0], [entry("m1", "Milo in January"), entry("m2", "Milo")],
                    [{"op": "DELETE", "id": "m1"},
                     {"op": "UPDATE", "id": "m2", "text": "Milo", "prov": []}])
    assert result["information_loss"] == 0.5
    assert result["redundancy_ratio"] == 1


def test_coverage_exhausts_complete_batches_until_late_witness(env):
    calls = scripted_judge(env)
    candidates = [{"id": f"m{i}", "text": "x" * 1000 + ("Milo" if i == 2 else "")} for i in range(3)]
    facts = [fact("Milo")]
    payload = {"facts": [{"index": 0, "text": "Milo"}], "entries": [candidates[-1]]}
    env.cfg.judge_input_tokens = env.counter.prompt_count(messages(COVERAGE, payload)) + 10
    witnesses, missing, searched = _coverage(env, facts, candidates)
    assert len(calls) == 3
    assert searched == ["m0", "m1", "m2"] and missing == []
    assert witnesses[0]["ids"] == ["m2"]
    assert all(len(payload["entries"][0]["text"]) >= 1000 for _, payload, _ in calls)


def test_coverage_requires_all_batches_before_reporting_missing(env):
    calls = scripted_judge(env)
    candidates = [{"id": f"m{i}", "text": "x" * 1000} for i in range(3)]
    payload = {"facts": [{"index": 0, "text": "Milo"}], "entries": [candidates[0]]}
    env.cfg.judge_input_tokens = env.counter.prompt_count(messages(COVERAGE, payload)) + 10
    _, missing, searched = _coverage(env, [fact("Milo")], candidates)
    assert len(calls) == 3 and missing == [0] and searched == ["m0", "m1", "m2"]


def test_coverage_rejects_unfit_entry_instead_of_truncating(env):
    scripted_judge(env)
    env.cfg.judge_input_tokens = 100
    with pytest.raises(Unknown, match="complete entry"):
        _coverage(env, [fact("Milo")], [{"id": "m1", "text": "x" * 1000}])


def test_coverage_rejects_fabricated_witness(env):
    env.judge = lambda *_: {"results": [{"index": 0, "covered": True,
                                         "witness_ids": ["missing"], "reason": "bad"}]}
    with pytest.raises(Unknown, match="witnesses"):
        _coverage(env, [fact("Milo")], [{"id": "m1", "text": "Milo"}])


@pytest.mark.parametrize("response", [
    {"facts": []}, {"facts": "invalid"}, {"unexpected": []},
    {"facts": [{**fact("Milo"), "status": "invented"}]},
    {"facts": [{**fact("Milo"), "source_rids": ["unknown"]}]},
    {"facts": [{**fact("Milo"), "source_rids": []}]},
    {"facts": [{**fact("Milo"), "text": ""}]},
])
def test_invalid_inventory_is_unknown(setup_case, env, response):
    env.judge = lambda *_: response
    with pytest.raises(Unknown):
        assess(env, setup_case[0], [entry("m1", "Milo")], [{"op": "DELETE", "id": "m1"}])


@pytest.mark.parametrize("response", [
    {"results": []}, {"results": "invalid"},
    {"results": [{"index": True, "covered": True, "witness_ids": ["m1"], "reason": "bad index"}]},
    {"results": [{"index": 0, "covered": "true", "witness_ids": ["m1"], "reason": "bad bool"}]},
    {"results": [{"index": 0, "covered": False, "witness_ids": ["m1"], "reason": "contradiction"}]},
])
def test_invalid_coverage_is_unknown(env, response):
    env.judge = lambda *_: response
    with pytest.raises(Unknown):
        _coverage(env, [fact("Milo")], [{"id": "m1", "text": "Milo"}])
