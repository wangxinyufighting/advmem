"""Offline regressions for exact-candidate reuse and scoped answer caching."""
from copy import deepcopy
import json

import pytest

from admem.common import Unknown, read, rows, write_rows
from admem.pipeline import builder_state
from admem.probes import make_suite, run_probe
from admem.store import neutral_round


def verdict():
    return {"legal": True, "faith": True, "accuracy": 1.0,
            "edit_quality_pass": True, "reward": 0.0, "effective_reward": 0.0}


def test_identical_samples_still_generate_eight_times(env, tmp_path, monkeypatch, capsys):
    path = tmp_path / "states.jsonl"
    write_rows(path, make_suite(env, tmp_path / "suite", variants=1)[:1])
    generated, scored = [], []

    def complete(prompt, nonce, **kwargs):
        generated.append((nonce, kwargs["temperature"]))
        return '{"ops":[]}'

    def score(*args):
        scored.append(args)
        return verdict()

    monkeypatch.setattr(env.policy("BUILDER"), "complete", complete)
    monkeypatch.setattr(env, "builder_score", score)
    report = run_probe(path, env, tmp_path / "out", samples=8, temperature=1.0)[0]
    samples = list(rows(tmp_path / "out" / "samples.jsonl"))
    assert len(generated) == len(set(nonce for nonce, _ in generated)) == 8
    assert all(temperature == 1.0 for _, temperature in generated)
    assert len(scored) == 1
    assert len(samples) == len(list(rows(tmp_path / "out" / "raw_samples.jsonl"))) == 8
    assert [s["score_cache_hit"] for s in samples] == [False] + [True] * 7
    assert all(s["generation_seconds"] >= 0 and s["scoring_seconds"] >= 0 for s in samples)
    assert report["score_cache_hits"] == 7
    assert report["unique_raw_mean"] == 1
    assert report["reward_std_mean"] == 0
    performance = read(tmp_path / "out" / "summary.json")["performance"]
    assert performance["generation_attempts"] == performance["generated_samples"] == 8
    assert performance["scoring_attempts"] == performance["scored_samples"] == 1
    assert performance["score_cache_hits"] == 7
    assert "8/8" in capsys.readouterr().err
    assert env._answer_cache is None


def test_score_cache_is_scoped_to_each_state(env, tmp_path, monkeypatch):
    path = tmp_path / "states.jsonl"
    records = make_suite(env, tmp_path / "suite", variants=1)[:2]
    write_rows(path, records)
    scored = []

    def score(full, state, raw):
        scored.append(state["prompt"])
        return verdict()

    monkeypatch.setattr(env, "builder_score", score)
    monkeypatch.setattr(env.policy("BUILDER"), "complete", lambda *a, **kw: '{"ops":[]}')
    run_probe(path, env, tmp_path / "out", samples=2)
    assert scored == [r["prompt"] for r in records]
    assert [s["score_cache_hit"] for s in rows(tmp_path / "out" / "samples.jsonl")] == [False, True] * 2


def test_unknown_score_is_retried_for_identical_raw(env, tmp_path, monkeypatch):
    path = tmp_path / "states.jsonl"
    write_rows(path, make_suite(env, tmp_path / "suite", variants=1)[:1])
    attempts = []

    def score(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise Unknown("Temporary judge outage")
        return verdict()

    monkeypatch.setattr(env, "builder_score", score)
    monkeypatch.setattr(env.policy("BUILDER"), "complete", lambda *a, **kw: '{"ops":[]}')
    report = run_probe(path, env, tmp_path / "out", samples=3)[0]
    samples = list(rows(tmp_path / "out" / "samples.jsonl"))
    assert len(attempts) == 2
    assert [s["status"] for s in samples] == ["unknown", "ok", "ok"]
    assert [s["score_cache_hit"] for s in samples] == [False, False, True]
    assert report["unknown_samples"] == 1
    performance = read(tmp_path / "out" / "summary.json")["performance"]
    assert performance["scoring_attempts"] == 2
    assert performance["scored_samples"] == 1


def test_interrupted_probe_releases_answer_cache(env, tmp_path, monkeypatch):
    path = tmp_path / "states.jsonl"
    write_rows(path, make_suite(env, tmp_path / "suite", variants=1)[:1])
    monkeypatch.setenv("PROBE_FAIL_LIMIT", "1")

    def unavailable(*args):
        raise Unknown("Judge unavailable")

    monkeypatch.setattr(env, "builder_score", unavailable)
    monkeypatch.setattr(env.policy("BUILDER"), "complete", lambda *a, **kw: '{"ops":[]}')
    with pytest.raises(Unknown):
        run_probe(path, env, tmp_path / "out", samples=8)
    assert env._answer_cache is None
    assert len(list(rows(tmp_path / "out" / "samples.jsonl"))) == 1
    assert read(tmp_path / "out" / "interrupted.json")["consecutive_errors"] == 1


def question():
    return {"q": "What is my cat's name?", "a": "Milo", "type": "single-session-user",
            "question_date": "2023-04-01", "E": ["s1:r1"]}


def card(text="Milo"):
    return {"id": "m1", "text": text, "prov": ["s1:r1"], "kind": "card"}


def test_answer_cache_key_isolates_memory_questions_and_flags(env, monkeypatch):
    calls = []
    context = env.context

    def observed(memory, q, all_memory=False):
        calls.append(1)
        return context(memory, q, all_memory)

    monkeypatch.setattr(env, "context", observed)
    memory, q = [card()], question()
    with env.cached_answers():
        first = env.answer(memory, q)
        first["visible_ids"].clear()  # Callers cannot mutate a cached verdict.
        assert env.answer(deepcopy(memory), deepcopy(q))["visible_ids"] == ["m1"]
        assert len(calls) == 1
        assert not env.answer([card("Cedar")], q)["correct"]
        assert not env.answer(memory, dict(q, a="Cedar"))["correct"]
        env.answer(memory, q, all_memory=True)
        env.answer(memory, q, abstention=True)
        env.answer(memory, q, type_hint=False)
        assert len(calls) == 6
    env.answer(memory, q)
    assert len(calls) == 7


def test_cached_baseline_preserves_reward_and_evaluates_new_memories(env, setup_case, monkeypatch):
    full, path, context = setup_case
    state = builder_state(full, path, context, [], [neutral_round(full, "s1:r1")],
                          "stream", [question()], env)
    actions = [json.dumps({"ops": [{"op": "ADD", "text": text, "prov": ["s1:r1"]}]})
               for text in ("My cat is named Milo.", "Milo")]
    calls = []
    original = env.context

    def observed(memory, q, all_memory=False):
        calls.append(deepcopy(memory))
        return original(memory, q, all_memory)

    monkeypatch.setattr(env, "context", observed)
    uncached = [env.builder_score(full, state, raw) for raw in actions]
    assert len(calls) == 4
    calls.clear()
    with env.cached_answers():
        cached = [env.builder_score(full, state, raw) for raw in actions]
    assert cached == uncached
    assert sum(not memory for memory in calls) == 1
    assert len(calls) == 3  # One baseline and each distinct proposed memory.


def test_answer_cache_is_bounded_and_cleans_up_after_failure(env, monkeypatch):
    calls = []
    original = env.context

    def observed(memory, q, all_memory=False):
        calls.append(1)
        return original(memory, q, all_memory)

    monkeypatch.setattr(env, "context", observed)
    with pytest.raises(Unknown):
        with env.cached_answers(max_entries=2):
            for text in ("Milo", "Cedar", "Juniper", "Milo"):
                env.answer([card(text)], question())
            assert len(calls) == 4
            assert len(env._answer_cache[0]) == 2
            raise Unknown("Interrupted sample")
    assert env._answer_cache is None


def test_failed_answer_is_not_cached(env, monkeypatch):
    original = env.grade
    attempts = []

    def grade(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise Unknown("Temporary grading failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(env, "grade", grade)
    with env.cached_answers():
        with pytest.raises(Unknown):
            env.answer([card()], question())
        assert env.answer([card()], question())["correct"]
        assert env.answer([card()], question())["correct"]
    assert len(attempts) == 2
