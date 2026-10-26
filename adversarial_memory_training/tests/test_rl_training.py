"""verl启动参数、奖励诊断字段、组内优势、验证指标、早停与rl_loop编排的离线测试（不需要verl/torch/GPU）。"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from conftest import Pack
from admem import rewards as rv2
from admem.common import Config, read, write, write_rows
from admem.data import prepare
from admem.pipeline import Collector, generate_bank, run_case
from admem.validation import (BUILDER_GUARDS, EarlyStopper, attacker_metrics, build_stats, builder_metrics,
                              paired_bootstrap)
from admem.verl_launch import LaunchOptions, main as launch_main, verl_env, verl_overrides

ROOT = Path(__file__).resolve().parents[2]
V2 = str(ROOT / "configs" / "rl_hidden_4b_v2.json")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ================================================================ advantages

def test_group_advantages_shape_invalid_samples_per_group():
    cfg = Config.load(V2)
    threshold = rv2.invalid_threshold(cfg, "builder")
    invalid = rv2.builder_invalid_reward_v2(cfg)
    assert invalid < threshold < -(cfg.builder_loss_weight + cfg.builder_faith_weight + cfg.builder_retention_weight
                                   + cfg.size_weight + cfg.builder_redundancy_weight * cfg.builder_redundancy_cap)
    scores = [0.08, 0.0, invalid, 0.07, 1.0, 1.0, 1.0, 1.0, 0.5]
    index = ["a", "a", "a", "a", "b", "b", "b", "b", "c"]
    adv = rv2.group_advantages(scores, index, threshold, std_floor=0.1)
    assert adv[2] < min(adv[0], adv[1], adv[3]) and adv[0] - adv[1] > 0.5
    assert adv[4:8] == [0.0] * 4 and adv[8] == 0.0  # 零方差组、单样本组优势为0
    assert rv2.invalid_threshold(Config(), "attacker") is None  # v1 attacker无法区分非法与最差合法
    assert rv2.invalid_threshold(Config(attacker_reward="v2"), "attacker") == pytest.approx(-0.15)


# ================================================================ verl launch

def test_launch_overrides_use_current_reward_keys_and_custom_estimator(tmp_path):
    o = LaunchOptions(role="builder", config=V2, train="t.parquet", model="/m", out=str(tmp_path / "o"), steps=40,
                      log_file=str(tmp_path / "log.jsonl"))
    args = verl_overrides(o)
    assert f"reward.custom_reward_function.path={ROOT / 'admem' / 'verl_reward.py'}" in args
    assert not any(a.startswith("custom_reward_function.") for a in args)
    assert "algorithm.adv_estimator=admem_grpo" in args and "trainer.total_training_steps=40" in args
    cfg = json.loads(Path(V2).read_text())
    assert f"data.max_prompt_length={cfg['builder_input_tokens'] + 256}" in args
    assert "data.val_files=t.parquet" in args and "trainer.logger=['console','file']" in args
    legacy = verl_overrides(LaunchOptions(role="attacker", config=V2, train="t", model="m", out="o",
                                          reward_keys="legacy"))
    assert any(a.startswith("custom_reward_function.path=") for a in legacy)
    assert f"data.max_prompt_length={cfg['attacker_input_tokens'] + 256}" in legacy
    env = verl_env(o, base={"VERL_USE_EXTERNAL_MODULES": "other.mod"})
    assert env["VERL_USE_EXTERNAL_MODULES"] == "other.mod,admem.verl_adv"
    assert env["ADMEM_ROLE"] == "builder" and env["ADMEM_CONFIG"].endswith("rl_hidden_4b_v2.json")
    assert env["PYTHONPATH"].startswith(str(ROOT)) and env["VERL_FILE_LOGGER_PATH"].endswith("log.jsonl")


def test_launch_dry_run_passes_extra_overrides(capsys):
    assert launch_main(["--role", "builder", "--config", V2, "--train", "t", "--model", "m", "--out", "o",
                        "--dry-run", "--", "trainer.nnodes=2"]) == 0
    printed = capsys.readouterr().out
    assert "verl.trainer.main_ppo" in printed and printed.strip().endswith("trainer.nnodes=2")


# ================================================================ reward diagnostics

def test_verl_reward_returns_score_and_fixed_diagnostic_keys(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from admem import verl_reward
    write(tmp_path / "s.json", {"state": {"split": "train", "role": "builder", "full_path": "f", "full_hash": "h",
                                          "environment_fingerprint": "c"}})
    details = iter([{"effective_reward": 0.3, "legal": True, "gain": 1.0, "noop": False,
                     "reward_components": {"gain": 1.0, "faith": -0.7}},
                    {"effective_reward": -8.2, "legal": False}])
    env = NS(cfg=NS(fingerprint=lambda: "c"),
             memory_module=NS(FullMemory=NS(load=lambda path: NS(fingerprint="h"))),
             score=lambda full, state, completion: next(details))
    monkeypatch.setattr(verl_reward, "environment", lambda path: env)
    extra = {"split": "train", "state_path": str(tmp_path / "s.json"), "config": "c.json"}
    good = verl_reward.compute_score("admem_builder", "{}", "", extra, reward_router_address=None)
    bad = verl_reward.compute_score("admem_builder", "x", "", extra)
    assert good["score"] == 0.3 and good["component_faith"] == -0.7 and good["legal"] == 1.0
    assert bad["score"] == -8.2 and bad["legal"] == 0.0 and set(good) == set(bad)


# ================================================================ data collection additions

def test_bank_collect_saves_attacker_states_and_build_records_ops(setup_case, env, tmp_path):
    full, path, ctx = setup_case
    generate_bank(full, path, ctx, env, tmp_path / "bank", collect=Collector(tmp_path / "collected"))
    states = [read(p) for p in (tmp_path / "collected" / "train" / "attacker").glob("*.json")]
    assert states and states[0]["state"]["role"] == "attacker" and states[0].get("completion")
    run_case(full, path, ctx, env, tmp_path / "run", mode="build")
    stats = build_stats(tmp_path / "run")
    assert stats["windows"] == stats["legal"] > 0 and stats["noop"] == 0


# ================================================================ validation metrics

def prepared_val(case, env, tmp_path):
    values = []
    for i in range(2):
        c = copy.deepcopy(case)
        c["question_id"] = f"q{i}"
        c["haystack_sessions"][1][0]["content"] += str(i)
        values.append(c)
    source = tmp_path / "data.json"
    source.write_text(json.dumps(values))
    prepare(source, tmp_path / "prepared", env.memory_module, [0, 2, 0])
    bank = {"q": "What is my cat's name?", "a": "Milo", "type": "single-session-user",
            "question_date": "2023-04-01", "E": ["s1:r1"]}
    for key in ("c0000", "c0001"):
        write(tmp_path / "val_bank" / key / "bank.json", [bank])
    return tmp_path / "prepared"


def test_builder_metrics_score_bank_and_official_questions(case, env, tmp_path):
    prepared = prepared_val(case, env, tmp_path)
    env.clients["BUILDER_EVAL"] = env.clients["BUILDER"]
    env.cfg.faith_gate = True
    metrics = builder_metrics(env, prepared, "val", tmp_path / "eval", val_bank=tmp_path / "val_bank")
    assert metrics["primary_name"] == "bank_accuracy" and metrics["primary"] == 1.0
    assert metrics["official_accuracy"] == 0.0 and metrics["official_questions"] == 2  # 参考答案是canary
    assert set(metrics["per_question"]) == {f"bank:c000{i}:{k}" for i in range(2)
                                            for k in [next(iter(metrics["per_question"])).rsplit(":", 1)[1]]}
    assert metrics["guards"]["legal_rate"] == 1.0 and metrics["guards"]["faith_rate"] == 1.0
    assert metrics["guards"]["noop_rate"] == 0.0 and metrics["guards"]["mean_m_tokens"] > 0
    full = builder_metrics(env, prepared, "val", tmp_path / "eval_full", val_bank=tmp_path / "val_bank",
                           reference="full")
    assert full["primary"] == 1.0 and full["windows"] == 0


def test_attacker_metrics_count_defects_against_reference_memory(case, env, tmp_path, monkeypatch):
    prepared = prepared_val(case, env, tmp_path)
    env.clients["ATTACKER_EVAL"] = env.clients["ATTACKER"]
    for key in ("c0000", "c0001"):
        write(tmp_path / "ref" / key / "M_build.json",
              [{"id": "m1", "text": "The user enjoys gardening.", "prov": ["s2:r1"], "kind": "card"}])
    metrics = attacker_metrics(env, prepared, "val", tmp_path / "eval", memory_run=tmp_path / "ref", packs_per_case=2)
    assert metrics["primary"] == 1.0 and metrics["guards"]["accept_rate"] == 1.0
    assert metrics["guards"]["invalid_rate"] == 0.0 and len(metrics["per_question"]) == metrics["counts"]["sent"]


# ================================================================ early stopping

def metrics(per_question, **guards):
    base = {"legal_rate": 1.0, "faith_rate": 0.8, "noop_rate": 0.0, "fallback_rate": 0.1, "mean_m_tokens": 1000}
    return {"primary": sum(per_question.values()) / len(per_question), "per_question": per_question,
            "guards": {**base, **guards}}


def answers(n_right, n=40):
    return {f"q{i}": float(i < n_right) for i in range(n)}


def test_paired_bootstrap_only_compares_shared_questions():
    prob, diff, paired = paired_bootstrap({"a": 1, "b": 1, "c": 0}, {"a": 0, "b": 1, "x": 1}, seed=1)
    assert paired == 2 and diff == 0.5 and prob > 0.5


def test_stopper_selects_significant_improvements_and_stops_on_patience():
    stopper = EarlyStopper(BUILDER_GUARDS, patience=2, confidence=0.9)
    assert stopper.update("s0", 0, metrics(answers(10)))["decision"] == "baseline"
    assert stopper.update("s20", 20, metrics(answers(20)))["improved"]       # +25个百分点，显著
    noisy = stopper.update("s40", 40, metrics(answers(21)))                  # +1题，不显著
    assert not noisy["improved"] and noisy["decision"] == "continue"
    final = stopper.update("s60", 60, metrics(answers(19)))
    assert final["decision"] == "stop" and stopper.state["best"]["tag"] == "s20"
    assert "no significant improvement" in stopper.state["reason"]


def test_stopper_guards_veto_and_stop_relative_to_baseline():
    stopper = EarlyStopper(BUILDER_GUARDS, patience=5, guard_patience=2, max_kl=0.5)
    stopper.update("s0", 0, metrics(answers(10)))
    collapsed = stopper.update("s20", 20, metrics(answers(30), noop_rate=0.4))  # 更高分但NOOP塌缩
    assert not collapsed["improved"] and "noop_rate" in collapsed["violations"][0]
    assert stopper.state["best"]["tag"] == "s0" and collapsed["decision"] == "continue"
    kl = stopper.update("s40", 40, {**metrics(answers(30)), "train": {"kl_loss": 0.9}})
    assert kl["decision"] == "stop" and "kl_loss" in stopper.state["reason"]


# ================================================================ rl_loop orchestration

class FakeLoop:
    """把rl_loop.Loop的外部步骤换成假的：训练/合并/验证只记录调用并返回预设指标。"""

    def __init__(self, module, args, sequence):
        self.loop = module.Loop(args)
        self.calls = []
        sequence = iter(sequence)
        self.loop.train_chunk = lambda target: self.calls.append(("train", target)) or str(Path(args.out) / "x.jsonl")
        self.loop.merge = lambda target: self.calls.append(("merge", target)) or Path(args.out) / f"m{target}"
        self.loop.validate = lambda model, tag: self.calls.append(("validate", tag)) or next(sequence)


def loop_args(module, tmp_path, *extra):
    return module.parser().parse_args(["--role", "builder", "--config", V2, "--train", "t", "--model", "base",
                                       "--prepared", "p", "--val-bank", "b", "--out", str(tmp_path / "rl"),
                                       "--eval-every", "10", "--max-steps", "100", "--patience", "2", *extra])


def test_rl_loop_stops_early_keeps_best_and_resumes(tmp_path):
    module = load_script("rl_loop")
    args = loop_args(module, tmp_path)
    run = FakeLoop(module, args, [metrics(answers(10)), metrics(answers(25)), metrics(answers(24)),
                                  metrics(answers(25))])
    summary = run.loop.run()
    assert [c for c in run.calls if c[0] == "train"] == [("train", 10), ("train", 20), ("train", 30)]
    assert summary["stopped_early"] and summary["best"]["tag"] == "step_00010" and summary["steps_trained"] == 30
    state = read(tmp_path / "rl" / "loop_state.json")
    assert state["step"] == 30 and state["stopper"]["stopped"]
    again = FakeLoop(module, args, [])  # 续跑：已停止，不再训练
    assert again.loop.run()["best"]["tag"] == "step_00010" and not again.calls


def test_rl_loop_reaches_max_steps_without_stopping(tmp_path):
    module = load_script("rl_loop")
    args = loop_args(module, tmp_path, "--max-steps", "20", "--patience", "5")
    summary = FakeLoop(module, args, [metrics(answers(5)), metrics(answers(15)), metrics(answers(30))]).loop.run()
    assert not summary["stopped_early"] and summary["reason"] == "reached --max-steps 20"
    assert summary["best"]["tag"] == "step_00020"


def test_train_metrics_read_kl_and_reward_from_verl_file_logger(tmp_path):
    module = load_script("rl_loop")
    log = tmp_path / "log.jsonl"
    write_rows(log, [{"step": i, "data": {"actor/kl_loss": 0.1 * i, "critic/score/mean": 0.5}} for i in range(1, 7)])
    values = module.train_metrics(log)
    assert values["kl_loss"] == pytest.approx(0.4) and values["train_reward"] == 0.5
