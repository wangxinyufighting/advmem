"""Reward-ordering regressions; semantic judge behavior is tested separately.

These tests use real action application, explicit per-question before/after verdicts,
and controlled edit-check verdicts so unrelated model behavior cannot mask a sign bug.
"""
import json

import pytest

from admem.common import Unknown
from admem.pipeline import builder_state


def card(eid, text, rid="s1:r1"):
    return {"id": eid, "text": text, "prov": [rid], "kind": "card"}


def state_for(env, setup_case, memory, n=1, mode="patch"):
    full, path, context = setup_case
    questions = [{"q": f"Recall test {i}", "a": f"Answer {i}",
                  "type": "single-session-user", "question_date": "2023-04-01",
                  "E": ["s1:r1"]} for i in range(n)]
    sources = [] if mode == "refine" else [
        {"rid": "s1:r1", "date": "2023-01-01", "text": "My cat is named Milo."}]
    return builder_state(full, path, context, memory, sources, mode, questions, env,
                         old_ids=[entry["id"] for entry in memory])


def checks(monkeypatch, **overrides):
    result = {"retention_pass": True, "information_loss": 0.0,
              "retention_checks": [], "redundancy_ratio": 0.0, "redundant_ops": []}
    result.update(overrides)
    monkeypatch.setattr("admem.environment.assess_edits", lambda *args: dict(result))


def answers(monkeypatch, env, state, before, after):
    def answer(memory, question):
        idx = state["tests"].index(question)
        scores = before if memory == state["M"] else after
        return {"correct": scores[idx]}
    monkeypatch.setattr(env, "answer", answer)


def update(text="Milo"):
    return json.dumps({"ops": [{"op": "UPDATE", "id": "m1", "text": text, "prov": []}]})


@pytest.mark.parametrize("faith_scores", [[False], [True, False]])
def test_unfaithful_writes_cannot_collect_answer_gain(
        env, setup_case, monkeypatch, faith_scores):
    full, _, _ = setup_case
    state = state_for(env, setup_case, [], n=2)
    checks(monkeypatch)
    answers(monkeypatch, env, state, [False, False], [True, True])
    monkeypatch.setattr(env, "faith_scores", lambda *args: faith_scores)
    env.cfg.size_weight = 0
    action = {"ops": [{"op": "ADD", "text": f"Claim {i}", "prov": ["s1:r1"]}
                      for i in range(len(faith_scores))]}
    result = env.builder_score(full, state, json.dumps(action))
    assert result["gain"] == 1
    assert result["faith"] is False
    assert result["faith_ratio"] == sum(faith_scores) / len(faith_scores)
    assert result["reward"] == pytest.approx(
        -env.cfg.builder_faith_weight * (1 - result["faith_ratio"]))
    assert result["effective_reward"] == result["reward"]


def test_forgetting_penalty_is_never_discounted_by_unfaithfulness(env, setup_case, monkeypatch):
    full, _, _ = setup_case
    state = state_for(env, setup_case, [card("m1", "Original detail")], n=2)
    checks(monkeypatch)
    answers(monkeypatch, env, state, [True, True], [False, True])
    env.cfg.size_weight = 0
    # Two writes allow comparison between all-faithful and half-faithful outcomes.
    action = json.dumps({"ops": [
        {"op": "UPDATE", "id": "m1", "text": "Replacement", "prov": []},
        {"op": "ADD", "text": "Additional detail", "prov": ["s1:r1"]}]})
    monkeypatch.setattr(env, "faith_scores", lambda *args: [True, True])
    faithful = env.builder_score(full, state, action)
    monkeypatch.setattr(env, "faith_scores", lambda *args: [True, False])
    partial = env.builder_score(full, state, action)
    assert faithful["loss"] == partial["loss"] == 0.5
    assert faithful["reward"] == pytest.approx(-0.5 * env.cfg.builder_loss_weight)
    assert partial["reward"] == pytest.approx(
        faithful["reward"] - 0.5 * env.cfg.builder_faith_weight)
    assert partial["reward"] < faithful["reward"]


def test_equal_accuracy_can_hide_forgetting_and_must_not_earn_compression(
        env, setup_case, monkeypatch):
    full, _, _ = setup_case
    state = state_for(env, setup_case, [card("m1", "A long original record with useful detail")],
                      n=2, mode="refine")
    checks(monkeypatch)
    answers(monkeypatch, env, state, [True, False], [False, True])
    monkeypatch.setattr(env, "faith_scores", lambda *args: [True])
    result = env.builder_score(full, state, update())
    assert result["delta"] == 0
    assert result["gain"] == result["loss"] == 0.5
    assert result["after_tokens"] < result["before_tokens"]
    assert not result["safe_compression"]
    assert result["reward"] == pytest.approx(
        0.5 * env.cfg.builder_gain_weight - 0.5 * env.cfg.builder_loss_weight)
    assert result["reward"] < 0


def test_deleting_untested_fact_is_penalized_despite_unchanged_answers(
        env, setup_case, monkeypatch):
    full, _, _ = setup_case
    state = state_for(env, setup_case, [card("m1", "Milo"),
                      card("m2", "The user enjoys gardening.", "s2:r1")], mode="refine")
    checks(monkeypatch, retention_pass=False, information_loss=1.0,
           retention_checks=[{"old_id": "m2", "missing": ["The user enjoys gardening."]}])
    answers(monkeypatch, env, state, [True], [True])
    result = env.builder_score(full, state, json.dumps({"ops": [{"op": "DELETE", "id": "m2"}]}))
    assert result["faith"] is True  # No new assertions; faith cannot protect deletion.
    assert result["loss"] == 0
    assert result["retention_pass"] is False
    assert not result["safe_compression"]
    assert result["reward"] == pytest.approx(-env.cfg.builder_retention_weight)


def test_supported_shortening_beats_legal_noop(env, setup_case, monkeypatch):
    full, _, _ = setup_case
    state = state_for(env, setup_case, [card("m1", "The user's cat is named Milo.")], mode="refine")
    checks(monkeypatch)
    answers(monkeypatch, env, state, [True], [True])
    monkeypatch.setattr(env, "faith_scores", lambda full, changed: [True] * len(changed))
    noop = env.builder_score(full, state, '{"ops":[]}', family="merge_timeline")
    shortened = env.builder_score(full, state, update())
    assert noop["reward"] == noop["effective_reward"] == 0
    assert shortened["safe_compression"]
    assert shortened["reward"] > noop["reward"]


def test_new_fact_is_not_redundant_merely_because_replay_misses_it(env, setup_case, monkeypatch):
    full, _, _ = setup_case
    state = state_for(env, setup_case, [card("m1", "Milo")])
    checks(monkeypatch)
    answers(monkeypatch, env, state, [True], [True])
    monkeypatch.setattr(env, "faith_scores", lambda *args: [True])
    action = {"ops": [{"op": "ADD", "text": "A genuinely new preference", "prov": ["s1:r1"]}]}
    result = env.builder_score(full, state, json.dumps(action), family="noop_repeat")
    assert result["gain"] == 0
    assert result["redundant_ops"] == []
    assert result["reward"] == pytest.approx(-env.cfg.size_weight * result["growth_ratio"])


def test_growth_cost_is_bounded_even_without_source_text(env, setup_case, monkeypatch):
    full, _, _ = setup_case
    state = state_for(env, setup_case, [card("m1", "Milo")], mode="refine")
    checks(monkeypatch)
    answers(monkeypatch, env, state, [True], [True])
    monkeypatch.setattr(env, "faith_scores", lambda *args: [True])
    result = env.builder_score(full, state, update("Milo " * 100))
    assert 0 <= result["growth_ratio"] <= 1
    assert result["reward"] >= -env.cfg.size_weight


def test_edit_check_outage_is_unknown_not_a_training_penalty(env, setup_case, monkeypatch):
    full, _, _ = setup_case
    state = state_for(env, setup_case, [card("m1", "Original detail")])
    monkeypatch.setattr(env, "faith_scores", lambda *args: [True])
    answers(monkeypatch, env, state, [True], [True])

    def unavailable(*args):
        raise Unknown("Retention judge unavailable")

    monkeypatch.setattr("admem.environment.assess_edits", unavailable)
    with pytest.raises(Unknown, match="Retention judge unavailable"):
        env.builder_score(full, state, update())


def test_invalid_action_is_below_all_bounded_legal_penalties(env, setup_case):
    full, _, _ = setup_case
    state = state_for(env, setup_case, [])
    invalid = env.builder_score(full, state, "invalid JSON")
    worst_bound = -(env.cfg.builder_loss_weight + env.cfg.builder_faith_weight
                    + env.cfg.builder_retention_weight + env.cfg.size_weight
                    + env.cfg.builder_redundancy_weight * env.cfg.max_ops)
    assert invalid["reward"] < worst_bound
    assert invalid["effective_reward"] == invalid["reward"]
    assert not invalid["legal"]
