"""Builder奖励v3：忠实条目按比例给分、漏存扣分（NOOP也扣）、faith权重可降、事实清单限30条；Builder提示词的ADD/id契约。"""
import json

import pytest

import admem.environment as envmod
from admem import edit_checks, prompts
from admem import rewards as rv2
from admem.common import Config
from test_reward_prompt_v2 import bstate, builder_case, contains_answer, q

FACTS = ["beagle Biscuit", "Biscuit adopted yesterday"]


def world(env, monkeypatch, version="v3"):
    """测试题按“记忆里有没有答案”判对错；faith看文本里是否有UNSUPPORTED；覆盖率按关键词找见证条目。"""
    env.cfg.builder_reward = version
    env.cfg.builder_faith_weight = 1.0 if version == "v3" else 2.0
    monkeypatch.setattr(env, "answer", contains_answer)
    monkeypatch.setattr(env, "faith_scores", lambda full, changed: ["UNSUPPORTED" not in e["text"] for e in changed])
    monkeypatch.setattr(envmod, "assess_edits", lambda *a, **k: {
        "retention_pass": True, "information_loss": 0.0, "redundant_ops": [], "redundancy_ratio": 0.0,
        "retention_checks": []})

    def coverage(env_, state, proposed, changed):
        written = {e["id"] for e in changed}
        witnesses = {i: [e["id"] for e in proposed if e["id"] in written and fact.split()[-1] in e["text"]]
                     for i, fact in enumerate(FACTS)}
        return {"facts": len(FACTS), "new": len(FACTS), "witnesses": {i: w for i, w in witnesses.items() if w}}

    monkeypatch.setattr(envmod, "source_coverage_v3", coverage)
    monkeypatch.setattr(envmod, "source_coverage", lambda env_, state, proposed, changed: {
        "facts": 2, "new": 2, "stored": len(coverage(env_, state, proposed, changed)["witnesses"]),
        "ratio": len(coverage(env_, state, proposed, changed)["witnesses"]) / 2})


def ops(*items):
    return json.dumps({"ops": [{"op": "ADD", "text": text, "prov": ["s2:r1"]} for text in items]})


GOOD = "The user adopted a beagle named Biscuit yesterday."
BAD = "UNSUPPORTED: the user also bought a parrot."
TESTS = [q("What is my dog's name?", ("s2:r1",), a="Biscuit")]


def score(env, completion, tests=TESTS):
    full = builder_case()
    return env.builder_score(full, bstate(full, tests), completion)


def test_noop_pays_for_new_facts_it_leaves_unstored(env, monkeypatch):
    world(env, monkeypatch, "v3")
    noop = score(env, '{"ops": []}')
    assert noop["reward"] == pytest.approx(-env.cfg.builder_omission_weight)
    assert noop["reward_components"]["omission"] == pytest.approx(-0.5)
    world(env, monkeypatch, "v2")
    assert score(env, '{"ops": []}')["reward"] == 0.0  # v2：NOOP恒为0


def test_partly_faithful_write_beats_noop_in_v3_but_not_in_v2(env, monkeypatch):
    world(env, monkeypatch, "v2")
    v2_mixed, v2_noop = score(env, ops(GOOD, BAD))["reward"], score(env, '{"ops": []}')["reward"]
    assert v2_mixed < v2_noop  # v2：一条不忠实就清零gain/覆盖并扣2×0.5
    world(env, monkeypatch, "v3")
    mixed, noop = score(env, ops(GOOD, BAD)), score(env, '{"ops": []}')
    good, bad = score(env, ops(GOOD)), score(env, ops(BAD))
    assert good["reward"] > mixed["reward"] > noop["reward"] > bad["reward"]
    assert mixed["reward_components"]["gain"] == pytest.approx(1.0 * 1.0 * 0.5)  # gain×faith_ratio
    assert mixed["coverage"]["stored"] == 2 and mixed["reward_components"]["omission"] == 0.0


def test_only_faithful_entries_count_as_storing_a_fact(env, monkeypatch):
    world(env, monkeypatch, "v3")
    result = score(env, ops("UNSUPPORTED: Biscuit is a beagle adopted yesterday."))
    assert result["coverage"]["stored_any"] == 2 and result["coverage"]["stored"] == 0
    assert result["reward_components"]["coverage"] == 0.0
    assert result["reward_components"]["omission"] == pytest.approx(-0.5)


def test_invalid_output_stays_below_every_legal_v3_reward(env, monkeypatch):
    world(env, monkeypatch, "v3")
    cfg = env.cfg
    worst_legal = -(cfg.builder_loss_weight + cfg.builder_faith_weight + cfg.builder_retention_weight
                    + cfg.size_weight + cfg.builder_redundancy_weight * cfg.builder_redundancy_cap
                    + cfg.builder_omission_weight)
    invalid = score(env, '{"ops": [{"op": "ADD", "id": "m9", "text": "x", "prov": ["s2:r1"]}]}')
    assert invalid["legal"] is False and invalid["reward"] == pytest.approx(worst_legal - 1.0)
    assert invalid["reward_version"] == "v3"
    assert rv2.invalid_threshold(cfg, "builder") == pytest.approx(invalid["reward"] + 0.5)


def test_v3_inventory_is_capped(env):
    seen = []

    def judge(system, payload, key):
        seen.append(system)
        return {"facts": [{"text": "fact", "status": "preserve", "source_rids": ["s1:r1"], "reason": "ok"}]}

    env.judge = judge
    full = builder_case()
    entry = {"id": "m1", "text": "Cat Milo", "prov": ["s1:r1"]}
    env.cfg.builder_reward = "v2"
    edit_checks._inventory(env, full, entry, [], "retention")
    env.cfg.builder_reward = "v3"
    edit_checks._inventory(env, full, entry, [], "retention")
    assert seen == [edit_checks.INVENTORY, edit_checks.INVENTORY_V3]
    assert "at most 30 facts" in edit_checks.INVENTORY_V3 and "at most 30" not in edit_checks.INVENTORY


def test_builder_prompt_states_that_add_has_no_id():
    assert "ADD has no id field" in prompts.BUILDER_V2
    assert '{"op": "ADD", "text": "<new fact>", "prov": ["<rid from allowed_source_rids>"]}' in prompts.BUILDER_V2


def test_v3_config_validates():
    cfg = Config.load(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "configs" / "rl_hidden_4b_v3.json"))
    assert cfg.builder_reward == "v3" and cfg.builder_faith_weight == 1.0 and cfg.salvage
    with pytest.raises(ValueError):
        Config(builder_reward="v4").validate()
    with pytest.raises(ValueError):
        Config(builder_reward="v3", builder_omission_weight=3.0).validate()
