"""Probe reports use training verdicts and preserve their reward evidence."""
import json

import pytest

from admem.common import Unknown, read, rows, write_rows
from admem.probes import make_suite, run_probe


def verdict(**overrides):
    value = {
        "legal": True, "faith": True, "faith_ratio": 1.0, "faith_scores": [],
        "accuracy": 1.0, "before_accuracy": 1.0, "gain": 0.0, "loss": 0.0,
        "before_scores": [True], "test_scores": [True], "delta": 0.0,
        "before_tokens": 20, "after_tokens": 20, "shrink_ratio": 0.0,
        "growth_ratio": 0.0, "safe_compression": True, "retention_pass": True,
        "information_loss": 0.0, "retention_checks": [], "redundancy_ratio": 0.0,
        "redundant_ops": [], "edit_quality_pass": True,
        "reward": 0.0, "effective_reward": 0.0, "reward_components": {"gain": 0.0},
    }
    value.update(overrides)
    return value


def test_probe_refine_noop_is_neutral_but_not_compression(env, tmp_path, monkeypatch):
    suite = make_suite(env, tmp_path / "suite", variants=1)
    record = next(r for r in suite if r["family"] == "merge_timeline")
    path = tmp_path / "selected.jsonl"
    write_rows(path, [record])
    # Three positional arguments intentionally reject a probe-only family keyword.
    monkeypatch.setattr(env, "builder_score", lambda full, state, raw: verdict())
    monkeypatch.setattr(env.policy("BUILDER"), "complete", lambda *a, **kw: '{"ops":[]}')

    report = run_probe(path, env, tmp_path / "out", samples=2)[0]
    assert report["success_rate_all_samples"] == 1.0
    assert report["compression_success_rate"] == 0.0
    assert report["noop_rate"] == 1.0
    assert report["family_constraint_rate"] == report["edit_quality_rate"] == 1.0
    sample = next(rows(tmp_path / "out" / "samples.jsonl"))
    assert sample["reward"] == sample["effective_reward"] == 0.0
    assert sample["reward_components"] == {"gain": 0.0}
    assert sample["before_scores"] == [True]
    assert sample["retention_checks"] == []


def test_probe_counts_unknowns_and_strict_faith_separately(env, tmp_path, monkeypatch):
    record = make_suite(env, tmp_path / "suite", variants=1)[0]
    path = tmp_path / "selected.jsonl"
    write_rows(path, [record])
    results = iter([
        Unknown("judge unavailable"),
        verdict(faith=False, faith_ratio=0.5, faith_scores=[True, False],
                edit_quality_pass=False, reward=-0.5, effective_reward=-0.5),
        verdict(),
    ])

    def score(full, state, raw):
        result = next(results)
        if isinstance(result, Unknown):
            raise result
        return result

    monkeypatch.setattr(env, "builder_score", score)
    monkeypatch.setattr(env.policy("BUILDER"), "complete", lambda *a, **kw: '{"ops":[]}')
    report = run_probe(path, env, tmp_path / "out", samples=3)[0]
    assert report["unknown_samples"] == 1
    assert report["faith_evaluated_samples"] == 2
    assert report["faith_rate_all_samples"] == pytest.approx(1 / 3)
    assert report["faith_rate_evaluated"] == 0.5
    assert report["faith_ratio_mean"] == 0.75
    assert report["legal_rate"] == pytest.approx(2 / 3)
    assert report["success_rate_all_samples"] == pytest.approx(1 / 3)
    assert report["first_sample_success_rate"] == 0.0
    samples = list(rows(tmp_path / "out" / "samples.jsonl"))
    assert samples[0]["reward"] is None
    assert samples[0]["status"] == "unknown"
    assert samples[1]["faith"] is False
    assert samples[1]["faith_scores"] == [True, False]
    assert "metric_notes" in read(tmp_path / "out" / "summary.json")


def test_probe_family_does_not_require_empty_ops(env, tmp_path, monkeypatch):
    record = next(r for r in make_suite(env, tmp_path / "suite", variants=1)
                  if r["family"] == "noop_repeat")
    path = tmp_path / "selected.jsonl"
    write_rows(path, [record])
    raw = json.dumps({"ops": [{"op": "UPDATE", "id": "m1", "text": "shorter", "prov": []}]})
    monkeypatch.setattr(env.policy("BUILDER"), "complete", lambda *a, **kw: raw)
    monkeypatch.setattr(env, "builder_score", lambda full, state, raw: verdict(
        after_tokens=10, shrink_ratio=0.5, reward=0.1, effective_reward=0.1))
    report = run_probe(path, env, tmp_path / "out", samples=1)[0]
    assert report["family_constraint_rate"] == 1.0
    assert report["success_rate_all_samples"] == 1.0
    assert report["noop_rate"] == 0.0


def test_suite_original_schedule_does_not_claim_completed_training(env, tmp_path):
    record = next(r for r in make_suite(env, tmp_path / "suite", variants=1)
                  if r["family"] == "update_preserve_old")
    text = record["state"]["M"][0]["text"]
    assert "initial weekly training schedule" in text
    assert "the user trained" not in text


def test_probe_limit_counts_reward_bearing_states(env, tmp_path, monkeypatch):
    suite = make_suite(env, tmp_path / "suite", variants=1)
    empty, eligible = suite[:2]
    empty["state"]["tests"] = []
    path = tmp_path / "real_windows.jsonl"
    write_rows(path, [empty, eligible])
    scored = []

    def score(full, state, raw):
        scored.append(state["case_key"])
        return verdict()

    monkeypatch.setattr(env, "builder_score", score)
    monkeypatch.setattr(env.policy("BUILDER"), "complete", lambda *a, **kw: '{"ops":[]}')
    report = run_probe(path, env, tmp_path / "out", samples=1, limit=1)
    assert scored == [eligible["state"]["case_key"]]
    assert report[0]["states"] == 1
    assert read(tmp_path / "out" / "summary.json")["skipped_empty_test_states"] == 1
