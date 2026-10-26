"""交替训练：记忆池、reader可答性有效性、建记忆的缓存复用、scripts/alternate.py编排。全部离线。"""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from conftest import Pack
from admem.common import Config, read, write
from admem.pipeline import _bank_pack, attacker_state, generate_bank, pool_index, run_case

ROOT = Path(__file__).resolve().parents[2]
V2 = str(ROOT / "configs" / "rl_hidden_4b_v2.json")


def question(text="What is my cat's name?"):
    return {"q": text, "a": "Milo", "type": "single-session-user", "question_date": "2023-04-01", "E": ["s1:r1"]}


def card(text):
    return [{"id": "m1", "text": text, "prov": ["s2:r1"], "kind": "card"}]


# ================================================================ memory pool

def test_pool_index_respects_latest_share_and_is_deterministic():
    cfg = Config()
    picks = Counter(pool_index(cfg, "c0000", pi, 3, 0.5) for pi in range(2000))
    assert 0.45 < picks[0] / 2000 < 0.55 and picks[1] > 300 and picks[2] > 300
    assert [pool_index(cfg, "c0000", pi, 3, 0.5) for pi in range(20)] == [
        pool_index(cfg, "c0000", pi, 3, 0.5) for pi in range(20)]
    assert {pool_index(cfg, "k", pi, 1, 0.5) for pi in range(50)} == {0}
    assert {pool_index(cfg, "k", pi, 3, 1.0) for pi in range(50)} == {0}


def test_bank_uses_the_memory_pool_and_states_keep_the_chosen_memory(setup_case, env, tmp_path, monkeypatch):
    full, path, ctx = setup_case
    packs = [Pack(f"p{i}", full.fingerprint, "seed", "session", 0, ["s1:r1"], list(full.rounds)) for i in range(12)]
    monkeypatch.setattr("admem.pipeline.make_pool", lambda *a, **k: packs)
    latest, older = card("Latest memory card."), card("Older memory card.")
    from admem.pipeline import Collector
    generate_bank(full, path, ctx, env, tmp_path / "bank", memory=latest, memory_pool=[("old", older)],
                  latest_share=0.5, collect=Collector(tmp_path / "collected"))
    logs = read(tmp_path / "bank" / "bank_log.json")
    used = {row["memory_pool_index"] for row in logs}
    assert used == {0, 1}
    manifest = read(tmp_path / "bank" / "bank_manifest.json")
    assert manifest["memory_pool"][0]["source"] == "old" and manifest["memory_latest_share"] == 0.5
    states = [read(p)["state"]["M"] for p in (tmp_path / "collected" / "train" / "attacker").glob("*.json")]
    assert {s[0]["text"] for s in states} == {"Latest memory card.", "Older memory card."}
    with pytest.raises(ValueError):
        generate_bank(full, path, ctx, env, tmp_path / "b2", memory_pool=[("old", older)])


# ================================================================ reader-answerable validity

def test_unanswerable_questions_are_rejected_by_reward_and_bank(setup_case, env, tmp_path, monkeypatch):
    full, path, ctx = setup_case
    env.cfg.attacker_reward, env.cfg.require_reader_answerable = "v2", True
    pack = Pack("p", full.fingerprint, "seed", "session", 0, ["s1:r1"], list(full.rounds))
    state = attacker_state(full, path, ctx, [], pack, [], env, 0)
    good = env.attacker_score(full, state, json.dumps({"items": [question()]}))
    assert good["items"][0]["status"] == "accepted" and good["reward"] > 0
    monkeypatch.setattr(env, "reader_answerable", lambda full, q: False)
    bad = env.attacker_score(full, state, json.dumps({"items": [question()]}))
    assert bad["items"][0]["reason"] == "reader_unanswerable" and -env.cfg.attacker_reject_penalty <= bad["reward"] < 0
    _, logs, added = _bank_pack(full, path, ctx, pack, 0, [], env, "ATTACKER")
    assert logs[0]["status"] == "rejected" and logs[0]["reason"] == "reader_unanswerable" and added == []


def test_unresolved_with_memory_is_valid_only_if_raw_evidence_suffices(setup_case, env, monkeypatch):
    full, _, _ = setup_case
    env.cfg.require_reader_answerable = True
    monkeypatch.setattr(env, "defect", lambda full, memory, q: {"kind": "unresolved"})
    monkeypatch.setattr(env, "reader_answerable", lambda full, q: True)
    defect, valid = env.validity_defect(full, card("x"), question())
    assert valid and defect["kind"] == "retrieval_composition_candidate"
    monkeypatch.setattr(env, "reader_answerable", lambda full, q: False)
    assert env.validity_defect(full, card("x"), question())[1] is False
    monkeypatch.setattr(env, "defect", lambda full, memory, q: {"kind": "missing"})
    assert env.validity_defect(full, card("x"), question()) == ({"kind": "missing"}, True)


def test_reader_answerable_uses_only_the_raw_evidence(setup_case, env):
    full, _, _ = setup_case
    assert env.reader_answerable(full, question()) is True
    assert env.reader_answerable(full, {**question(), "E": ["s2:r1"]}) is False


def test_rl_configs_enable_strict_validity():
    assert Config.load(V2).require_reader_answerable


# ================================================================ cached rebuild

def test_rebuilding_with_a_new_bank_reuses_builder_nonces(setup_case, env, tmp_path):
    full, path, ctx = setup_case
    run_case(full, path, ctx, env, tmp_path / "a", mode="build", bank=[])
    run_case(full, path, ctx, env, tmp_path / "b", mode="build", bank=[question()])
    nonces = [n for role, _, n in env.control["prompts"] if role == "BUILDER"]
    half = len(nonces) // 2
    assert half and nonces[:half] == nonces[half:]


# ================================================================ alternate.py

def load_alternate():
    spec = importlib.util.spec_from_file_location("alternate", ROOT / "scripts" / "alternate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fake:
    """替换外部动作：命令只记录；rl_loop阶段按脚本写summary.json。"""

    def __init__(self, module, args, outcomes, fail_on=None):
        self.alt = module.Alternation(args)
        self.commands, self.outcomes, self.fail_on = [], list(outcomes), fail_on
        self.alt.serve = lambda model, name: self.commands.append(("serve", model)) or None
        self.alt.stop = lambda process: None
        self.alt.run_cmd = self.run_cmd

    def run_cmd(self, command, env=None):
        command = [str(c) for c in command]
        if self.fail_on and self.fail_on in " ".join(command):
            raise RuntimeError("simulated failure")
        self.commands.append(command)
        if command[1].endswith("rl_loop.py"):
            out, model = command[command.index("--out") + 1], command[command.index("--model") + 1]
            improved = self.outcomes.pop(0)
            best = {"tag": "step_00020" if improved else "step_00000",
                    "model_path": f"{out}/merged/step_00020" if improved else model}
            write(Path(out) / "summary.json", {"best": best, "reason": "test", "stopped_early": True})


def alt_args(module, tmp_path, *extra):
    (tmp_path / "val_bank").mkdir(exist_ok=True)
    return module.parser().parse_args(["--config", V2, "--prepared", "P", "--builder", "B0", "--attacker", "A0",
                                       "--val-bank", str(tmp_path / "val_bank"), "--out", str(tmp_path / "alt"),
                                       "--rounds", "4", *extra])


def flag(command, name):
    return command[command.index(name) + 1] if name in command else None


def test_alternation_runs_rounds_with_pool_cumulative_bank_and_stops(tmp_path):
    module = load_alternate()
    fake = Fake(module, alt_args(module, tmp_path), [True, True, False, False])
    summary = fake.alt.run()
    cmds = [c for c in fake.commands if c[0] != "serve"]
    banks = [c for c in cmds if "bank" in c and "admem.cli" in c]
    rl = [c for c in cmds if c[1].endswith("rl_loop.py")]
    assert [flag(c, "--role") for c in rl] == ["attacker", "builder", "attacker", "builder"]
    r1, r2 = tmp_path / "alt" / "round_1", tmp_path / "alt" / "round_2"
    assert "--memory-pool" not in banks[0] and flag(banks[2], "--memory-pool") == str(r1 / "build_train")
    assert flag(banks[3], "--base-bank") == str(r1 / "bank") and "--defects-only" in banks[3]
    assert flag(rl[2], "--model") == f"{r1}/rl_attacker/merged/step_00020"   # A1
    assert flag(rl[3], "--model") == f"{r1}/rl_builder/merged/step_00020"    # B1
    assert flag(rl[1], "--val-bank") == str(tmp_path / "val_bank") and flag(rl[0], "--memory-run") == str(r1 / "build_val")
    assert summary["stopped_early"] and summary["rounds"] == 2 and "neither" in summary["reason"]
    assert summary["builder"].endswith("round_1/rl_builder/merged/step_00020") and summary["bank"] == str(r2 / "bank")


def test_builder_first_skips_the_first_attacker_phase(tmp_path):
    module = load_alternate()
    fake = Fake(module, alt_args(module, tmp_path, "--start", "builder", "--rounds", "1"), [True])
    fake.alt.run()
    rl = [c for c in fake.commands if c[0] != "serve" and c[1].endswith("rl_loop.py")]
    assert [flag(c, "--role") for c in rl] == ["builder"]


def test_alternation_resumes_after_a_failure_without_repeating_steps(tmp_path):
    module = load_alternate()
    args = alt_args(module, tmp_path, "--rounds", "1")
    with pytest.raises(RuntimeError):
        Fake(module, args, [True], fail_on="rl_loop.py --role builder").alt.run()
    resumed = Fake(module, args, [True])
    resumed.alt.run()
    ran = [c for c in resumed.commands if c[0] != "serve"]
    assert len(ran) == 1 and flag(ran[0], "--role") == "builder"
    assert read(tmp_path / "alt" / "alternate_state.json")["done"]["round_1"][-1] == "builder_phase"
