"""judge/reader格式不合时重问；v3事实清单只清点能被问到的信息（可为空）。"""
import pytest

from admem import edit_checks
from admem.common import Unknown
from admem.environment import boolean
from test_reward_prompt_v2 import builder_case


def test_schema_failure_is_retried_with_error_note(env, monkeypatch):
    calls = []
    replies = iter([{"correct": "yes"}, {"correct": True}])

    def judge(system, payload, key):
        calls.append((key, payload.get("previous_reply_error")))
        return next(replies)

    monkeypatch.setattr(env, "judge", judge)
    assert env.judge_checked("SYS", {"q": 1}, "grade:k", lambda v: boolean(v, "correct")) is True
    assert calls[0] == ("grade:k", None)  # 第一次请求不变：已有缓存照常命中
    assert calls[1][0] == "grade:k:schema-retry1" and "must be boolean" in calls[1][1]
    assert env.schema_retry_successes == 1


def test_schema_retries_are_bounded_and_can_be_disabled(env, monkeypatch):
    calls = []
    monkeypatch.setattr(env, "judge", lambda s, p, k: calls.append(k) or {"correct": None})
    with pytest.raises(Unknown, match="must be boolean"):
        env.judge_checked("SYS", {}, "k", lambda v: boolean(v, "correct"))
    assert calls == ["k", "k:schema-retry1", "k:schema-retry2"]
    calls.clear()
    monkeypatch.setenv("JUDGE_SCHEMA_RETRIES", "0")
    with pytest.raises(Unknown):
        env.judge_checked("SYS", {}, "k", lambda v: boolean(v, "correct"))
    assert calls == ["k"]


def test_transport_failure_is_not_retried_as_schema_error(env, monkeypatch):
    calls = []

    def judge(system, payload, key):
        calls.append(key)
        raise Unknown("JUDGE transport/service failure")

    monkeypatch.setattr(env, "judge", judge)
    with pytest.raises(Unknown, match="transport"):
        env.judge_checked("SYS", {}, "k", lambda v: v)
    assert calls == ["k"]


def test_reader_without_answer_string_is_asked_again(env, monkeypatch):
    remote, seen = env.policy("DEFENDER"), []
    replies = iter([{"reply": "Milo"}, {"answer": "Milo"}])

    def json_call(prompt, nonce, prompt_tokens=None):
        seen.append((nonce, prompt[-1]["content"]))
        return next(replies)

    monkeypatch.setattr(remote, "json", json_call)
    q = {"q": "What is my cat's name?", "a": "Milo", "type": "single-session-user",
         "question_date": "2023-04-01", "E": ["s1:r1"]}
    assert env.reader(q, "", "closed:k") == "Milo"
    assert seen[1][0].endswith(":schema-retry1") and "no answer string" in seen[1][1]


def test_v3_inventory_counts_only_answerable_information():
    text = edit_checks.INVENTORY_V3
    assert "never a\ndull moment" in text and "conversation process itself" in text
    assert "entities and names" in text and "numbers, quantities" in text
    assert "at most 30 facts" in text and "at most 60 facts" in edit_checks.INVENTORY_V3_RETENTION
    assert "Never omit a supported fact because it seems unimportant" in edit_checks.INVENTORY  # v1/v2不变


def test_v3_inventory_may_be_empty_but_v2_may_not(env, monkeypatch):
    full = builder_case()
    entry = {"id": "m1", "text": "Orlando's parks: never a dull moment!", "prov": ["s1:r1"]}
    monkeypatch.setattr(env, "judge", lambda system, payload, key: {"facts": []})
    env.cfg.builder_reward = "v3"
    assert edit_checks._inventory(env, full, entry, [], "retention") == []
    env.cfg.builder_reward = "v2"
    with pytest.raises(Unknown, match="no facts"):
        edit_checks._inventory(env, full, entry, [], "retention")
