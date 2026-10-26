"""scripts/probe_breakdown.py：离线拆解与ceiling诊断（用离线替身，不调用真实模型）。"""
import importlib.util
import json
from pathlib import Path

from admem.common import write_rows

_spec = importlib.util.spec_from_file_location(
    "probe_breakdown", Path(__file__).resolve().parents[2] / "scripts" / "probe_breakdown.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def sample(state, reward, ops, faith, before, after, noop=False, loss_facts=None):
    checks = [{"entry_id": "m1", "facts": [{"text": t} for t in loss_facts], "missing_fact_indices": [0]}] \
        if loss_facts else []
    return {"state_id": state, "status": "ok", "legal": True, "reward": reward, "noop": noop,
            "raw_text": json.dumps({"ops": ops}), "faith_scores": faith, "faith_ratio": sum(faith) / max(1, len(faith)),
            "before_scores": before, "test_scores": after, "information_loss": 0.5 if loss_facts else 0.0,
            "retention_checks": checks, "reward_components": {"gain": 0.0, "faith": reward},
            "generation_seconds": 1.0, "scoring_seconds": 20.0}


def test_breakdown_reports_noop_gap_unfaithful_writes_and_dead_tests():
    add = {"op": "ADD", "text": "Bought bike yesterday on 2023/05/09", "prov": ["s2:r1"]}
    upd = {"op": "UPDATE", "id": "m1", "text": "Rewritten card", "prov": []}
    samples = [sample("a", -1.0, [upd, add], [True, False], [False], [False], loss_facts=["Lives in Denver"]),
               sample("a", 0.2, [add], [True], [False], [False]),
               sample("b", -0.5, [add], [False], [False], [False]),
               {"state_id": "b", "status": "unknown", "error": "JUDGE JSON invalid after two attempts"}]
    report = tool.breakdown(samples)
    assert report["noop_comparison"]["states_mean_below_noop"] == 2
    assert report["tests"]["states_where_no_test_is_ever_correct"] == 2
    assert report["operations"]["faithful_entry_rate_by_op"] == {"UPDATE": 1.0, "ADD": 0.333}
    assert report["examples"]["unfaithful_writes"][0]["text"] == add["text"]
    assert report["examples"]["lost_facts"][0]["missing_facts"] == ["Lives in Denver"]
    assert report["unknown_errors"][0][0].startswith("JUDGE JSON")
    assert report["reward"]["components"]["faith"]["within_state_std"] > 0


def test_ceiling_separates_builder_misses_from_unanswerable_questions(setup_case, env, tmp_path):
    full, path, ctx = setup_case
    q = {"q": "What is my cat's name?", "a": "Milo", "type": "single-session-user",
         "question_date": "2023-04-01", "E": ["s1:r1"]}
    state = {"full_path": str(path), "M": [{"id": "m", "text": "Likes gardening.", "prov": ["s2:r1"], "kind": "card"}],
             "tests": [q]}
    states = tmp_path / "states.jsonl"
    write_rows(states, [{"id": "s0", "state": state}])
    result = tool.ceiling(env, states, {"s0"})
    assert result["accuracy"] == {"before": 0.0, "M+raw(E)": 1.0, "raw(E) only": 1.0}
