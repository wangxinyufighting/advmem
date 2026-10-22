"""跨会话出题的结构预筛，全部离线，不替代真实模型/gate 验证。"""
from types import SimpleNamespace

from conftest import Full, Pack
from admem.audit_prompts import feasible_types, personal_score
from admem.pipeline import attacker_state, _bank_pack


def history(second=None):
    return Full.build({
        "haystack_session_ids": ["first", "second"],
        "haystack_dates": ["2023-01-01", "2023-02-01"],
        "haystack_sessions": [
            [{"role": "user", "content": "I bought two books for $20."}],
            second or [{"role": "user", "content": "I purchased three more books for $30."}],
        ],
    })


def context():
    return {"key": "c1", "split": "val", "question_type": "multi-session",
            "question_date": "2023-03-01"}


def test_multi_session_personal_word_misses_reach_attacker(env, tmp_path):
    full = history()
    assert personal_score(full, full.rounds) == 0  # Old prefilter dropped both statements.
    pack = Pack("p1", full.fingerprint, "first", "session", 0, ["s1:r1"], list(full.rounds))
    env.cfg.attacker_type_filter = True
    env.cfg.seed_min_personal = 1
    state = attacker_state(full, tmp_path / "full.json", context(), [], pack, [], env, 0)
    assert state["qtype"] == "multi-session"
    assert set(state["pack"]["rids"]) == {"s1:r1", "s2:r1"}


def test_multiple_rounds_in_one_session_are_not_multiple_sessions():
    full = SimpleNamespace(rounds={rid: SimpleNamespace(session_id="same", messages=[])
                                  for rid in ["s1:r1", "s1:r2"]})
    pack = SimpleNamespace(seed_rids=["s1:r1"], rids=list(full.rounds))
    assert "multi-session" not in feasible_types(full, pack, min_personal=0)


def test_cross_session_cluster_seed_does_not_need_third_session():
    full = history()
    pack = Pack("p1", full.fingerprint, "cluster", "cluster", 0,
                list(full.rounds), list(full.rounds))
    assert "multi-session" in feasible_types(full, pack, min_personal=1)


def test_compact_view_must_retain_two_actual_sessions(env, tmp_path):
    full = history([{"role": "assistant", "content": "A previous recommendation."}])
    pack = Pack("p1", full.fingerprint, "first", "session", 0, ["s1:r1"], list(full.rounds))
    env.cfg.attacker_view = "compact"
    assert attacker_state(full, tmp_path / "f.json", context(), [], pack, [], env, 0) is None
    env.cfg.attacker_view = "full"
    assert attacker_state(full, tmp_path / "f.json", context(), [], pack, [], env, 0) is not None


def test_skipped_bank_pack_keeps_type_in_diagnostics(env, tmp_path):
    full = history()
    pack = Pack("p1", full.fingerprint, "first", "session", 0, ["s1:r1"], ["s1:r1"])
    env.cfg.attacker_type_filter = True
    info, logs, items = _bank_pack(full, tmp_path / "f.json", context(), pack, 0, [], env, "ATTACKER")
    assert items == [] and not env.control["prompts"]
    assert logs[0]["status"] == "type_infeasible"
    assert logs[0]["qtype"] == info["case_question_type"] == "multi-session"
    assert info["pack_sessions"] == 1


def test_two_session_pack_does_not_bypass_evidence_gate(env):
    full = history()
    pack = Pack("p1", full.fingerprint, "first", "session", 0, ["s1:r1"], list(full.rounds))
    question = {"q": "How much did I spend?", "a": "$20", "type": "multi-session",
                "question_date": "2023-03-01", "E": ["s1:r1"]}
    assert env.gate(full, pack, question, "2023-03-01", "multi-session")["status"] == "rejected"
