"""bank/run的调用量：同一pack的题并发过gate（结果与顺序不变）、按题型换teacher出题、
gate记录oracle具体失败项、run --collect默认不为SFT示范重答测试题。"""
import json
import time

import pytest

from conftest import Pack
from admem import pipeline
from admem.common import read
from admem.pipeline import Collector, _bank_pack, run_case


def pack_for(full):
    return Pack("p1", full.fingerprint, "seed", "session", 0, ["s1:r1"], list(full.rounds))


def question(text="What is my cat's name?"):
    return {"q": text, "a": "Milo", "type": "single-session-user",
            "question_date": "2023-04-01", "E": ["s1:r1"]}


def test_parallel_gate_keeps_item_order_and_dedup(setup_case, env, tmp_path, monkeypatch):
    full, path, context = setup_case
    items = [question("What is my cat's name?"), question("Which name did I give my cat?"),
             question("What is my cat's name?")]  # 第3题与第1题重复：照常判定但不重复入库
    monkeypatch.setattr(env.policy("ATTACKER"), "complete", lambda *a, **kw: json.dumps({"items": items}))
    delays = {items[0]["q"]: 0.15, items[1]["q"]: 0.0}

    def gate(full, pack, item, date, qtype):
        time.sleep(delays.get(item["q"], 0.05))  # 第1题最慢：完成顺序与原顺序不同
        return {"status": "accepted", "item": item}

    monkeypatch.setattr(env, "gate", gate)
    env.cfg.require_reader_answerable = False
    info, logs, added = _bank_pack(full, path, context, pack_for(full), 0, [], env, "ATTACKER",
                                   trace_path=tmp_path / "t.json")
    assert [row["item"]["q"] for row in logs] == [q["q"] for q in items]
    assert added == items[:2] and info["policy_role"] == "ATTACKER"
    trace = read(tmp_path / "t.json")
    assert trace["stage"] == "complete" and trace["added_items"] == items[:2]


def test_teacher_role_generates_only_the_listed_types(setup_case, env, monkeypatch):
    full, path, context = setup_case
    env.cfg.type_policy, env.cfg.attacker_type_filter = "official", False
    used = []
    for role in ("ATTACKER", "TEACHER"):
        monkeypatch.setattr(env.policy(role), "complete",
                            lambda *a, _r=role, **kw: used.append(_r) or '{"items": []}')
    ms = {**context, "question_type": "multi-session"}
    info, _, _ = _bank_pack(full, path, ms, pack_for(full), 0, [], env, "ATTACKER",
                            teacher_role="TEACHER", teacher_types=("multi-session",))
    assert info["policy_role"] == "TEACHER" and used == ["TEACHER"]
    info, _, _ = _bank_pack(full, path, context, pack_for(full), 1, [], env, "ATTACKER",
                            teacher_role="TEACHER", teacher_types=("multi-session",))
    assert info["policy_role"] == "ATTACKER" and used == ["TEACHER", "ATTACKER"]


def test_gate_reports_which_oracle_check_failed(setup_case, env, monkeypatch):
    full, path, context = setup_case
    oracle = {"answerable": True, "answer": "Milo", "user_relevant": True, "type_valid": False,
              "no_answer_leak": "n/a"}  # 已判失败后，其余字段的非布尔值不再算服务失败
    monkeypatch.setattr(env, "judge", lambda system, payload, key: oracle)
    result = env.gate(full, pack_for(full), question(), "2023-04-01", "single-session-user")
    assert result["status"] == "rejected" and result["reason"] == "oracle_or_type"
    assert result["failed_checks"] == ["type_valid"]


def test_collect_skips_demo_verification_unless_requested(setup_case, env, tmp_path, monkeypatch):
    full, path, context = setup_case
    calls = []
    monkeypatch.setattr(pipeline, "all_pass", lambda *a, **kw: calls.append(1) or True)
    bank = [question()]
    run_case(full, path, context, env, tmp_path / "off", mode="build", bank=bank,
             collect=Collector(tmp_path / "c_off"), verify_demos=False)
    assert calls == []
    states = [read(p) for p in (tmp_path / "c_off" / "train" / "builder").glob("*.json")]
    with_tests = [s for s in states if s["state"]["tests"]]
    assert with_tests and all(s.get("completion") is None for s in with_tests)
    run_case(full, path, context, env, tmp_path / "on", mode="build", bank=bank,
             collect=Collector(tmp_path / "c_on"), verify_demos=True)
    assert calls  # 打开时照旧逐题验证
