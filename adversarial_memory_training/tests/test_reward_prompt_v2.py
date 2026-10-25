"""v2奖励（attacker/builder）、v2提示词与其他修复的离线测试。"""
import json
from pathlib import Path

import pytest

from conftest import Full, Pack
import admem.environment as envmod
from admem import rewards as rv2
from admem.audit_prompts import ATTACKER_SCORING_NOTE, prompt_hashes
from admem.common import Config, InvalidAction, read, role_models, self_judging
from admem.pipeline import attacker_state, builder_state, run_case, stream_tests, Collector
from admem.prompts import BUILDER, BUILDER_V2, SCREEN
from admem.store import apply, round_dates

ROOT = Path(__file__).resolve().parents[2]
DATE = "2023-04-01"


def q(text="What is my cat's name?", E=("s1:r1",), qtype="single-session-user", a="Milo"):
    return {"q": text, "a": a, "type": qtype, "question_date": DATE, "E": list(E)}


def pair(user, assistant="Noted."):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def history(*users):
    return Full.build({"haystack_session_ids": [f"src{i}" for i in range(len(users))],
                       "haystack_dates": [f"2023/0{i + 1}/01 (Sun) 10:00" for i in range(len(users))],
                       "haystack_sessions": [pair(u) for u in users]})


def ctx(qtype="single-session-user"):
    return {"key": "c0000", "split": "train", "question_type": qtype, "question_date": DATE}


def attacker_setup(env, tmp_path, full=None, accepted=()):
    env.cfg.attacker_reward = "v2"
    full = full or history("My cat is named Milo.", "I enjoy gardening.")
    pack = Pack("p", full.fingerprint, "src0", "session", 0, ["s1:r1"], list(full.rounds))
    return full, attacker_state(full, tmp_path / "f.json", ctx(), [], pack, list(accepted), env, 0)


# ================================================================ attacker v2

def test_rejected_items_score_below_an_empty_list_and_invalid_below_everything(env, tmp_path, monkeypatch):
    full, state = attacker_setup(env, tmp_path)
    monkeypatch.setattr(env, "gate", lambda full, pack, item, date, qtype: {
        "status": "rejected", "reason": "closed_book", "item": item})
    four = json.dumps({"items": [q(f"Question {i}?") for i in range(4)]})
    assert env.attacker_score(full, state, four)["reward"] < 0 == env.attacker_score(
        full, state, json.dumps({"items": []}))["reward"]
    monkeypatch.setattr(env, "gate", lambda full, pack, item, date, qtype: {
        "status": "rejected", "reason": "Question schema mismatch", "item": item})
    worst_legal = env.attacker_score(full, state, four)["reward"]
    assert env.attacker_score(full, state, "not json")["reward"] < worst_legal == -env.cfg.attacker_reject_penalty


def test_rejection_ladder_follows_gate_order():
    order = [{"reason": "Question schema mismatch"},
             {"reason": "oracle_or_type", "oracle": {"type_valid": False}},
             {"reason": "oracle_or_type", "oracle": {"answerable": False}},
             {"reason": "unsupported_answer"}, {"reason": "scope_or_completeness"}, {"reason": "closed_book"}]
    values = [rv2.rejection_reward_v2(g, 0.1) for g in order]
    assert values == sorted(values) and all(v < 0 for v in values)


def test_padding_evidence_cannot_raise_the_weight(env, tmp_path, monkeypatch):
    full, state = attacker_setup(env, tmp_path)
    state["evidence_usage"] = {"s1:r1": 3}
    accept = lambda full, pack, item, date, qtype: {"status": "accepted", "item": item}  # noqa: E731
    monkeypatch.setattr(env, "gate", accept)
    lean = env.attacker_score(full, state, json.dumps({"items": [q()]}))["items"][0]
    # 审计判定只有s1:r1必要：塞进一个没用过的round不会把novelty拉回，还会按必要比例打折。
    monkeypatch.setattr(env, "evidence_audit", lambda full, item, prior: {
        "necessary_rids": ["s1:r1"], "personal": True, "duplicate_of": None, "reason": "stub"})
    padded = env.attacker_score(full, state, json.dumps({"items": [q(E=("s1:r1", "s2:r1"))]}))["items"][0]
    assert lean["weight"] == pytest.approx(0.25)
    assert padded["weight"] == pytest.approx(0.125) and padded["necessity"] == 0.5


def test_personal_flag_comes_from_the_audit_not_a_word_list(env, tmp_path):
    full = history("I bought a road bike for $900.", "Explain gear ratios.")  # 旧词表把第一句判成0
    full, state = attacker_setup(env, tmp_path, full)
    item = env.attacker_score(full, state, json.dumps({"items": [q(a="Milo")]}))["items"][0]
    assert item["status"] == "accepted" and item["audit"]["personal"] and item["weight"] == 1.0


def test_semantic_duplicates_and_repairs_earn_less(env, tmp_path, monkeypatch):
    full, state = attacker_setup(env, tmp_path, accepted=[q("What did I name my cat?")])
    assert state["prior_questions"] == [{"q": "What did I name my cat?", "E": ["s1:r1"]}]
    monkeypatch.setattr(env, "evidence_audit", lambda full, item, prior: {
        "necessary_rids": item["E"], "personal": True, "duplicate_of": 0 if prior else None, "reason": "stub"})
    dup = env.attacker_score(full, state, json.dumps({"items": [q()]}))
    assert dup["items"][0]["status"] == "duplicate" and dup["reward"] < 0
    state["prior_questions"], state["evidence_usage"] = [], {}
    monkeypatch.setattr(env, "gate", lambda full, pack, item, date, qtype: {
        "status": "accepted", "item": item, "repaired": True})
    repaired = env.attacker_score(full, state, json.dumps({"items": [q()]}))["items"][0]
    assert repaired["weight"] == env.cfg.attacker_repair_credit


def test_empty_memory_skips_defect_but_real_memory_uses_it(env, tmp_path, monkeypatch):
    full, state = attacker_setup(env, tmp_path)
    monkeypatch.setattr(env, "defect", lambda *a: pytest.fail("defect must not run against empty M"))
    item = env.attacker_score(full, state, json.dumps({"items": [q()]}))["items"][0]
    assert item["defect"]["kind"] == "skipped_empty_memory" and item["reward"] == 1.0
    state["M"] = [{"id": "m1", "text": "The user's cat is named Milo.", "prov": ["s1:r1"], "kind": "card"}]
    monkeypatch.setattr(env, "defect", lambda *a: {"kind": "none"})
    assert env.attacker_score(full, state, json.dumps({"items": [q()]}))["items"][0]["reward"] == 0.5


def test_scoring_note_is_only_in_v2_prompts(env, tmp_path):
    full = history("My cat is named Milo.", "I enjoy gardening.")
    pack = Pack("p", full.fingerprint, "src0", "session", 0, ["s1:r1"], list(full.rounds))
    v1 = attacker_state(full, tmp_path / "f.json", ctx(), [], pack, [], env, 0)
    env.cfg.attacker_reward = "v2"
    v2 = attacker_state(full, tmp_path / "f.json", ctx(), [], pack, [], env, 0)
    assert ATTACKER_SCORING_NOTE not in v1["prompt"][0]["content"] and "prior_questions" not in v1
    assert ATTACKER_SCORING_NOTE in v2["prompt"][0]["content"]


# ================================================================ builder v2

def builder_case():
    return history("My cat is named Milo.", "I adopted a beagle named Biscuit yesterday.")


def builder_stub(env, monkeypatch, answer):
    env.cfg.builder_reward = "v2"
    monkeypatch.setattr(env, "answer", answer)
    monkeypatch.setattr(env, "faith_scores", lambda full, changed: [True] * len(changed))
    monkeypatch.setattr(envmod, "assess_edits", lambda *a, **k: {
        "retention_pass": True, "information_loss": 0.0, "redundant_ops": [], "redundancy_ratio": 0.0,
        "retention_checks": []})
    monkeypatch.setattr(envmod, "source_coverage", lambda *a, **k: {"facts": 0, "new": 0, "stored": 0, "ratio": None})


def bstate(full, tests, memory=None, x=None):
    x = x or [{"rid": "s2:r1", "date": full.sessions["src1"].date,
               "text": "user: I adopted a beagle named Biscuit yesterday.", "span": [0, 10]}]
    memory = memory if memory is not None else [
        {"id": f"m{i}", "text": f"Old fact {i}.", "prov": ["s1:r1"], "kind": "card"} for i in range(11)]
    return {"M": memory, "old_ids": [e["id"] for e in memory], "x": x, "mode": "stream", "tests": tests,
            "entry_limit": 600}


ADD = json.dumps({"ops": [{"op": "ADD", "text": "The user adopted a beagle named Biscuit.", "prov": ["s2:r1"]}]})


def contains_answer(memory, question, **kw):
    return {"correct": any(question["a"] in e["text"] for e in memory), "visible_ids": [], "answer": ""}


def test_gain_is_not_diluted_by_replay_tests(env, monkeypatch):
    full = builder_case()
    builder_stub(env, monkeypatch, contains_answer)
    new = q("What is my dog's name?", ("s2:r1",), a="Biscuit")
    replay = [q(f"Old {i}?", ("s1:r1",), a=f"Old fact {i}") for i in range(11)]
    one = env.builder_score(full, bstate(full, [new]), ADD)
    twelve = env.builder_score(full, bstate(full, [new] + replay), ADD)
    assert one["gain"] == twelve["gain"] == 1.0 and twelve["target_tests"] == 1
    assert twelve["reward"] == pytest.approx(one["reward"], abs=0.02)
    noop = env.builder_score(full, bstate(full, [new] + replay), json.dumps({"ops": []}))
    assert noop["reward"] == 0.0


def test_lookahead_tests_credit_the_window_that_holds_part_of_the_evidence(env, monkeypatch, tmp_path):
    full = Full.build({"haystack_session_ids": ["A", "B", "C"],
                       "haystack_dates": ["2023/05/01 (Mon) 10:00", "2023/05/09 (Tue) 10:00", "2023/06/02 (Fri) 10:00"],
                       "haystack_sessions": [pair("I love jazz."), pair("I bought a road bike for $900."),
                                             pair("I bought a bike helmet for $120.")]})
    bank = [q("How much have I spent on cycling gear?", ("s2:r1", "s3:r1"), "multi-session", "$1,020")]
    window = [{"rid": "s2:r1", "date": "2023/05/09", "text": "user: I bought a road bike for $900.", "span": [0, 5]}]
    env.cfg.builder_reward = "v1"
    assert stream_tests(bank, window, {"s1:r1", "s2:r1"}, set(), env) == []
    env.cfg.builder_reward = "v2"
    tests = stream_tests(bank, window, {"s1:r1", "s2:r1"}, set(), env)
    assert [t["context_rids"] for t in tests] == [["s3:r1"]]

    def both_purchases(memory, question, **kw):  # 只有两笔都在记忆里才答对
        text = " ".join(e["text"] for e in memory)
        return {"correct": "$900" in text and "$120" in text, "visible_ids": [], "answer": ""}

    builder_stub(env, monkeypatch, both_purchases)
    state = bstate(full, tests, memory=[], x=window)
    stored = json.dumps({"ops": [{"op": "ADD", "text": "The user bought a road bike for $900.", "prov": ["s2:r1"]}]})
    assert env.builder_score(full, state, stored)["gain"] == 1.0  # s3原文作为context补进来
    skipped = json.dumps({"ops": [{"op": "ADD", "text": "The user likes bikes.", "prov": ["s2:r1"]}]})
    assert env.builder_score(full, state, skipped)["gain"] == 0.0


def test_lookahead_tests_enter_collected_states(env, tmp_path, monkeypatch):
    full = Full.build({"haystack_session_ids": ["A", "B"], "haystack_dates": ["2023-05-01", "2023-05-09"],
                       "haystack_sessions": [pair("I bought a road bike for $900."), pair("I bought a helmet for $120.")]})
    path = tmp_path / "full.json"
    full.save(path)
    bank = [q("How much did I spend on cycling gear?", ("s1:r1", "s2:r1"), "multi-session", "$1,020")]
    env.cfg.builder_reward = "v2"
    run_case(full, path, ctx("multi-session"), env, tmp_path / "run", mode="build", bank=bank,
             collect=Collector(tmp_path / "collected"))
    states = [read(p)["state"] for p in (tmp_path / "collected" / "train" / "builder").glob("*.json")]
    first = [s for s in states if s["x"][0]["rid"] == "s1:r1"][0]
    assert first["tests"][0]["context_rids"] == ["s2:r1"] and first["coverage_scorable"]


def test_retrieval_displacement_flips_count_half(env, monkeypatch):
    full = builder_case()

    def displaced(memory, question, k=None, **kw):  # 新卡片把旧答案挤出top-k，2k下仍能答对
        added = any("Biscuit" in e["text"] for e in memory)
        return {"correct": (not added) or k is not None, "visible_ids": [], "answer": ""}

    builder_stub(env, monkeypatch, displaced)
    old = q("Old 0?", ("s2:r1",), a="Old fact 0")
    result = env.builder_score(full, bstate(full, [old]), ADD)
    assert result["test_deltas"] == [-0.5] and result["loss"] == 0.5 and result["confirmed_flips"] == 1


def test_coverage_rewards_storing_untested_facts_and_test_free_states_are_scorable(env, monkeypatch):
    full = builder_case()
    builder_stub(env, monkeypatch, contains_answer)
    monkeypatch.setattr(envmod, "source_coverage", lambda env, state, proposed, changed: {
        "facts": 2, "new": 2, "stored": sum("Biscuit" in e["text"] for e in changed), "ratio":
        sum("Biscuit" in e["text"] for e in changed) / 2})
    result = env.builder_score(full, bstate(full, []), ADD)
    assert result["reward_components"]["coverage"] == pytest.approx(0.5 * env.cfg.builder_coverage_weight)
    env.cfg.builder_coverage_weight = 0
    with pytest.raises(Exception, match="no reward is defined"):
        env.builder_score(full, bstate(full, []), ADD)


def test_source_coverage_counts_only_new_facts(env):
    from admem.edit_checks import COVERAGE, SOURCE_INVENTORY, source_coverage

    def judge(system, payload, key):
        if system == SOURCE_INVENTORY:
            return {"facts": [{"text": "cat is Milo", "source_rids": ["s2:r1"]},
                              {"text": "dog is Biscuit", "source_rids": ["s2:r1"]}]}
        assert system == COVERAGE
        texts = " ".join(e["text"] for e in payload["entries"])
        return {"results": [{"index": f["index"], "covered": f["text"].split()[-1] in texts,
                             "witness_ids": [payload["entries"][0]["id"]] if f["text"].split()[-1] in texts else [],
                             "reason": "stub"} for f in payload["facts"]]}

    env.judge = judge
    state = {"x": [{"rid": "s2:r1", "date": "d", "text": "t"}], "old_ids": ["m1"],
             "M": [{"id": "m1", "text": "Cat Milo", "prov": ["s1:r1"]}]}
    proposed = state["M"] + [{"id": "m2", "text": "Dog Biscuit", "prov": ["s2:r1"]}]
    result = source_coverage(env, state, proposed, [proposed[1]])
    assert result == {"facts": 2, "new": 1, "stored": 1, "ratio": 1.0}


def test_redundancy_ignores_duplicates_the_builder_could_not_see(env, monkeypatch):
    from admem import edit_checks
    full = builder_case()
    memory = [{"id": "seen", "text": "The user's cat is Milo.", "prov": ["s1:r1"], "kind": "card"},
              {"id": "hidden", "text": "The user adopted a beagle named Biscuit.", "prov": ["s2:r1"], "kind": "card"}]
    state = {**bstate(full, [], memory), "old_ids": ["seen"]}
    action = json.loads(ADD)
    proposed, changed = apply(memory, action, mode="stream", visible_ids=["seen"], source_ids=["s2:r1"],
                              full_ids=full.rounds, counter=env.counter, limit=600)
    monkeypatch.setattr(edit_checks, "_inventory", lambda *a: [])
    everywhere = edit_checks.assess_edits(env, full, state, action, proposed, changed)
    visible = edit_checks.assess_edits(env, full, state, action, proposed, changed, scope_ids={"seen"})
    assert everywhere["redundant_ops"] and not visible["redundant_ops"]


def test_invalid_reward_shaping_and_std_floor():
    cfg = Config(builder_reward="v2")
    assert rv2.builder_invalid_reward_v2(cfg) < -(2 + 2 + 2 + 0.2 + 0.1 * 10)
    rewards = [0.081, 0.0, 0.075, rv2.builder_invalid_reward_v2(cfg), 0.080, 0.0, 0.079, 0.0]
    shaped = rv2.shape_group(rewards, [True, True, True, False, True, True, True, True])
    adv, _ = rv2.normalized_advantages(shaped)
    assert shaped[3] < 0 and adv[0] - adv[1] > 0.5  # 合法样本之间的差别没有被非法样本压扁
    small = [0.073, 0.067, 0.063, 0.059]
    assert max(rv2.normalized_advantages(small)[0]) > 1
    assert max(rv2.normalized_advantages(small, std_floor=0.1)[0]) < 0.15


# ================================================================ prompts

def test_v2_builder_prompt_catalog_and_payload(env, tmp_path):
    full = builder_case()
    memory = [{"id": f"m{i}", "text": f"Fact {i} about {'dogs' if i == 0 else 'tea'}.", "prov": ["s1:r1"],
               "kind": "card"} for i in range(5)]
    x = [{"rid": "s2:r1", "date": "d", "text": "user: I adopted a beagle.", "span": [0, 5]}]
    v1 = builder_state(full, tmp_path / "f.json", ctx("temporal-reasoning"), [], x, "stream", [], env)
    assert v1["prompt"][0]["content"].endswith(BUILDER) and json.loads(v1["prompt"][1]["content"])["write_required"]
    env.cfg.builder_prompt, env.cfg.builder_reward = "v2", "v2"
    v2 = builder_state(full, tmp_path / "f.json", ctx("temporal-reasoning"), memory, x, "stream", [], env,
                       old_ids=["m1"])
    system, payload = v2["prompt"][0]["content"], json.loads(v2["prompt"][1]["content"])
    assert system.endswith(BUILDER_V2) and "TEMPORAL MEMORY CHECK" not in system
    assert "write_required" not in payload
    assert "m1" not in [c["id"] for c in payload["stored_catalog"]]  # 已在M_old里全文给出
    assert v2["catalog_ids"] == [c["id"] for c in payload["stored_catalog"]]
    assert len(BUILDER_V2) < len(BUILDER) / 1.4


def test_cards_carry_source_dates_and_the_reader_sees_them(env):
    full = builder_case()
    action = json.loads(ADD)
    plain, _ = apply([], action, mode="stream", visible_ids=[], source_ids=["s2:r1"], full_ids=full.rounds,
                     counter=env.counter, limit=600)
    dated, _ = apply([], action, mode="stream", visible_ids=[], source_ids=["s2:r1"], full_ids=full.rounds,
                     counter=env.counter, limit=600, round_dates=round_dates(full))
    assert "dates" not in plain[0] and dated[0]["dates"] == ["2023/02/01 (Sun) 10:00"]
    assert env.render_entry(dated[0]) == dated[0]["text"]
    env.cfg.builder_prompt = "v2"
    assert env.render_entry(dated[0]).startswith("[Source date: 2023/02/01 (Sun) 10:00] ")
    raw = {"id": "r", "text": "Recorded on x. y", "prov": ["s1:r1"], "kind": "raw", "dates": ["x"]}
    assert env.render_entry(raw) == raw["text"]


def test_initial_noop_is_allowed_only_without_the_v1_rule(env):
    full = builder_case()
    kwargs = dict(mode="stream", visible_ids=[], source_ids=["s2:r1"], full_ids=full.rounds,
                  counter=env.counter, limit=600)
    with pytest.raises(InvalidAction):
        apply([], {"ops": []}, **kwargs)
    assert apply([], {"ops": []}, forbid_initial_noop=False, **kwargs) == ([], [])


def test_prompt_hashes_record_the_screen_prompt_actually_used():
    import hashlib
    assert prompt_hashes()["gate_screen"] == hashlib.sha256(SCREEN.encode()).hexdigest()


# ================================================================ other issues

@pytest.mark.parametrize("aware,hidden", [("type_aware.json", "type_hidden.json"),
                                          ("type_aware_4b.json", "type_hidden_4b.json"),
                                          ("type_aware_4b_nogate.json", "type_hidden_4b_nogate.json"),
                                          ("type_aware_4b_nogate_v2.json", "hidden_4b_nogate_v2.json")])
def test_hint_ablation_configs_differ_only_in_hint_mode(aware, hidden):
    a, h = (json.loads((ROOT / "configs" / n).read_text()) for n in (aware, hidden))
    assert {k for k in set(a) | set(h) if a.get(k) != h.get(k)} == {"hint_mode"}


def test_self_judging_is_detected(monkeypatch):
    for role in ("BUILDER", "ATTACKER", "DEFENDER", "JUDGE"):
        monkeypatch.delenv(role + "_MODEL", raising=False)
        monkeypatch.delenv(role + "_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "qwen")
    monkeypatch.setenv("LLM_BASE_URL", "http://h/v1")
    assert "BUILDER=JUDGE" in self_judging()
    monkeypatch.setenv("JUDGE_MODEL", "big-judge")
    monkeypatch.setenv("DEFENDER_MODEL", "big-reader")
    assert self_judging() == [] and role_models()["JUDGE"][0] == "big-judge"


def test_verl_neutral_policy_returns_zero_and_logs(tmp_path, monkeypatch):
    from admem import verl_reward
    from admem.common import Unknown
    monkeypatch.setenv("ADMEM_REWARD_UNKNOWN_LOG", str(tmp_path / "log.jsonl"))
    with pytest.raises(Unknown):
        verl_reward._unknown_reward(Unknown("judge down"), {"state_path": "s"})
    monkeypatch.setenv("ADMEM_REWARD_ON_UNKNOWN", "neutral")
    assert verl_reward._unknown_reward(Unknown("judge down"), {"state_path": "s"}) == 0.0
    assert "judge down" in (tmp_path / "log.jsonl").read_text()
