"""Offline checks for the real-history CLI runner; no model/API calls."""
import json
import importlib.util
from collections import Counter
from pathlib import Path
import sys

import pytest

from admem.common import Config, read, write, write_rows
from admem.audit_prompts import TYPES
_spec = importlib.util.spec_from_file_location(
    "probe_longmemeval_runner", Path(__file__).resolve().parents[2] / "scripts" / "probe_longmemeval.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def state_row(key="c0000", identity="s0", tests=None):
    state = {"case_key": key, "role": "builder", "mode": "stream", "split": "val",
             "full_hash": "full-" + key, "environment_fingerprint": "config-hash",
             "prompt": [{"role": "user", "content": "history only"}],
             "tests": [{"q": "generated question"}] if tests is None else tests}
    return {"id": identity, "state": state, "prompt": state["prompt"]}


def test_filter_before_limit_and_round_robin_are_deterministic():
    records = [state_row(identity="empty", tests=[])]
    records += [state_row(key, f"{key}-{i}") for key in ("c0000", "c0001") for i in range(5)]
    first, counts = runner.select_states(records, ["c0000", "c0001"], 4, 3, "config-hash")
    second, _ = runner.select_states(reversed(records), ["c0000", "c0001"], 4, 3, "config-hash")
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert [r["state"]["case_key"] for r in first] == ["c0000", "c0001", "c0000", "c0001"]
    assert counts["c0000"] == {"all": 6, "available": 5, "selected": 2}
    assert counts["c0001"] == {"all": 5, "available": 5, "selected": 2}


@pytest.mark.parametrize("change", [
    {"mode": "refine"}, {"environment_fingerprint": "stale"},
    {"full_hash": "wrong"}, {"split": "test"},
])
def test_reject_incompatible_collected_states(change):
    row = state_row()
    row["state"].update(change)
    with pytest.raises(ValueError):
        runner.select_states([row], ["c0000"], 1, 0, "config-hash",
                             full_hashes={"c0000": "full-c0000"}, split="val")


def test_case_without_reward_states_is_not_silently_omitted():
    with pytest.raises(ValueError, match="No reward-bearing stream states for c0001"):
        runner.select_states([state_row()], ["c0000", "c0001"], 4, 0, "config-hash")


def test_cli_uses_current_interpreter_and_repo_root(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    runner._run_cli("probe", tmp_path / "config.json", "--out", tmp_path / "out")
    args, kwargs = calls[0]
    assert args[0][:4] == [sys.executable, "-m", "admem.cli", "probe"]
    assert kwargs == {"cwd": runner.ROOT, "check": True}


def prepared_inputs(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    write(config, {"project": str(runner.ROOT), "tokenizer": "test/tokenizer", "embedding": "none"})
    prepared = tmp_path / "prepared"
    write(prepared / "manifest.json", {"cases": [{"key": "c0000", "split": "val",
                                                "question_type": "single-session-user"}]})
    write(prepared / "cases" / "c0000" / "context.json", {
        "key": "c0000", "split": "val", "question_type": "single-session-user", "full_hash": "full-c0000"})
    write(prepared / "cases" / "c0000" / "full.json", {
        "fingerprint": "full-c0000", "sessions": [{"session_id": "s1"}]})
    monkeypatch.setenv("LLM_MODEL", "offline-model")
    for role in ("BUILDER", "ATTACKER", "DEFENDER", "JUDGE"):
        monkeypatch.delenv(role + "_MODEL", raising=False)
        monkeypatch.delenv(role + "_BASE_URL", raising=False)
        monkeypatch.delenv(role + "_EXTRA_BODY", raising=False)
    monkeypatch.delenv("LLM_EXTRA_BODY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    return runner.parser().parse_args([
        "--config", str(config), "--prepared", str(prepared),
        "--data", str(tmp_path / "unused.json"), "--out", str(tmp_path / "out"),
        "--case-limit", "1", "--state-limit", "2", "--samples", "2",
    ])


def test_preflight_preserves_existing_prepared_and_checks_local_models(tmp_path, monkeypatch):
    args = prepared_inputs(tmp_path, monkeypatch)
    cfg, roles = runner.preflight(args)
    assert cfg.embedding == "none"
    assert roles["BUILDER"] == "offline-model"
    assert not args.data.exists()  # Existing preparation does not require raw source again.
    write(args.config, {"project": str(runner.ROOT), "tokenizer": str(tmp_path / "missing-model")})
    with pytest.raises(ValueError, match="Local tokenizer path does not exist"):
        runner.preflight(args)


def test_preflight_rejects_partial_prepared_existing_out_and_missing_role(tmp_path, monkeypatch):
    args = prepared_inputs(tmp_path, monkeypatch)
    Path(args.out).mkdir()
    with pytest.raises(ValueError, match="Output already exists"):
        runner.preflight(args)
    Path(args.out).rmdir()
    monkeypatch.delenv("LLM_MODEL")
    with pytest.raises(ValueError, match="BUILDER_MODEL"):
        runner.preflight(args)
    monkeypatch.setenv("LLM_MODEL", "offline-model")
    (Path(args.prepared) / "manifest.json").unlink()
    with pytest.raises(ValueError, match="no manifest.json"):
        runner.preflight(args)


def test_runner_calls_existing_stages_with_identical_case_keys(tmp_path, monkeypatch):
    args = prepared_inputs(tmp_path, monkeypatch)
    fingerprint = Config.load(args.config).fingerprint()
    calls = []

    def fake_cli(command, config, *arguments):
        calls.append((command, [str(a) for a in arguments]))
        arguments = [str(a) for a in arguments]
        out = Path(arguments[arguments.index("--out") + 1])
        if command in {"bank", "run"}:
            write(out / "summary.json", [{"key": "c0000", "status": "ok"}])
            filename = "bank.json" if command == "bank" else "M_final.json"
            write(out / "c0000" / filename, [{"q": "generated"}] if command == "bank" else [])
        elif command == "export":
            records = [state_row(identity="empty", tests=[]), state_row(identity="valid")]
            for row in records:
                row["state"]["environment_fingerprint"] = fingerprint
            write_rows(out, records)
        elif command == "probe":
            selected = [json.loads(line) for line in Path(arguments[arguments.index("--states") + 1]).read_text().splitlines()]
            assert len(selected) == 1 and selected[0]["state"]["tests"]
            assert selected[0]["family"] == "single-session-user"
            assert selected[0]["state"]["mode"] == "stream"
            assert "--limit" not in arguments
            write(out / "summary.json", {"groups": [{"family": "single-session-user"}]})
        else:
            pytest.fail(f"Unexpected stage: {command}")

    monkeypatch.setattr(runner, "_run_cli", fake_cli)
    result = runner.run(args)
    assert [c[0] for c in calls] == ["bank", "run", "export", "probe"]
    for _, arguments in calls[:2]:
        assert arguments[arguments.index("--keys") + 1] == "c0000"
        assert arguments[arguments.index("--split") + 1] == "val"
    assert calls[1][1][calls[1][1].index("--mode") + 1] == "build"
    assert result["nature"] == "real_history_stream/generated_bank"
    assert result["state_counts"]["c0000"] == {"all": 2, "available": 1, "selected": 1}
    assert read(Path(args.out) / "selected_cases.json")[0]["sessions"] == 1


def test_runner_stops_on_empty_bank_before_build(tmp_path, monkeypatch):
    args = prepared_inputs(tmp_path, monkeypatch)
    calls = []

    def fake_cli(command, config, *arguments):
        calls.append(command)
        arguments = list(map(str, arguments))
        out = Path(arguments[arguments.index("--out") + 1])
        write(out / "summary.json", [{"key": "c0000", "status": "ok"}])
        write(out / "c0000" / "bank.json", [])

    monkeypatch.setattr(runner, "_run_cli", fake_cli)
    with pytest.raises(ValueError, match="No accepted generated questions"):
        runner.run(args)
    assert calls == ["bank"]


def test_wrong_prepare_sizes_fail_before_any_cli_call(tmp_path, monkeypatch):
    args = prepared_inputs(tmp_path, monkeypatch)
    args.prepared = str(tmp_path / "new_prepared")
    write(args.data, [{"question_id": "dummy"}])
    monkeypatch.setattr(runner, "_run_cli", lambda *a: pytest.fail("Unexpected CLI call"))
    with pytest.raises(ValueError, match="input contains 1 cases"):
        runner.run(args)
    assert not Path(args.prepared).exists()


def typed_inputs(tmp_path, monkeypatch, *, counts=None, extra=()):
    """Three cases per actual question type, including abstention strata."""
    original = prepared_inputs(tmp_path, monkeypatch)
    prepared = Path(original.prepared)
    catalog = []
    for question_type in TYPES:
        for index in range((counts or {}).get(question_type, 3)):
            key = f"c{len(catalog):04d}"
            catalog.append({"key": key, "split": "val", "question_type": question_type,
                            "stratum": "abstention" if index == 0 else question_type})
            write(prepared / "cases" / key / "context.json", {
                "key": key, "split": "val", "question_type": question_type,
                "full_hash": "full-" + key})
            write(prepared / "cases" / key / "full.json", {
                "fingerprint": "full-" + key, "sessions": [{"session_id": "s1"}]})
    write(prepared / "manifest.json", {"cases": catalog})
    return runner.parser().parse_args([
        "--config", str(original.config), "--prepared", str(prepared),
        "--out", str(original.out), *extra,
    ])


def test_default_selects_two_cases_for_each_of_six_actual_question_types(tmp_path, monkeypatch):
    args = typed_inputs(tmp_path, monkeypatch)
    runner.preflight(args)
    selected = runner.selected_cases(args)
    assert len(selected) == 12  # No leftover default case-limit=1 truncation.
    assert Counter(row["question_type"] for row in selected) == {typ: 2 for typ in TYPES}
    assert len({row["key"] for row in selected}) == 12
    assert args.states_per_case == 16
    assert args.state_limit is None
    # Every type's first case was marked abstention. Grouping by seven strata
    # would select a different count and fail the six-type quota above.
    manifest = {r["key"]: r for r in read(Path(args.prepared) / "manifest.json")["cases"]}
    assert any(manifest[r["key"]]["stratum"] == "abstention" for r in selected)


def test_explicit_per_type_and_type_subset(tmp_path, monkeypatch):
    args = typed_inputs(tmp_path, monkeypatch, extra=[
        "--per-type", "3", "--question-types", "knowledge-update", "temporal-reasoning",
    ])
    runner.preflight(args)
    selected = runner.selected_cases(args)
    assert Counter(row["question_type"] for row in selected) == {
        "knowledge-update": 3, "temporal-reasoning": 3,
    }


def test_type_shortage_stops_before_any_bank_call(tmp_path, monkeypatch):
    args = typed_inputs(tmp_path, monkeypatch, counts={"temporal-reasoning": 1})
    monkeypatch.setattr(runner, "_run_cli", lambda *a: pytest.fail("No CLI stage should run"))
    with pytest.raises(ValueError, match="temporal-reasoning"):
        runner.run(args)
    assert not Path(args.out).exists()


def test_explicit_case_limit_keeps_single_case_smoke_test(tmp_path, monkeypatch):
    args = typed_inputs(tmp_path, monkeypatch, extra=["--case-limit", "1"])
    runner.preflight(args)
    assert len(runner.selected_cases(args)) == 1


def test_explicit_keys_do_not_activate_implicit_per_type_quota(tmp_path, monkeypatch):
    args = typed_inputs(tmp_path, monkeypatch, extra=["--keys", "c0001"])
    runner.preflight(args)
    assert [r["key"] for r in runner.selected_cases(args)] == ["c0001"]


@pytest.mark.parametrize("options", [
    ["--per-type", "2", "--case-limit", "1"],
    ["--states-per-case", "16", "--state-limit", "16"],
])
def test_case_and_state_budget_flags_are_mutually_exclusive(options):
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["--out", "unused", *options])


def test_state_cap_is_per_case_without_borrowing_unused_capacity():
    records = [state_row("c0000", f"a{i}") for i in range(2)]
    records += [state_row("c0001", f"b{i}") for i in range(40)]
    selected, counts = runner.select_states(
        records, ["c0000", "c0001"], 32, 0, "config-hash", per_case_limit=16)
    assert len(selected) == 18
    assert counts["c0000"]["selected"] == 2
    assert counts["c0001"]["selected"] == 16


def test_default_run_preserves_all_type_case_quotas_and_per_case_state_budgets(tmp_path, monkeypatch):
    args = typed_inputs(tmp_path, monkeypatch)
    fingerprint = Config.load(args.config).fingerprint()
    bank_keys = []
    cases = {r["key"]: r for r in read(Path(args.prepared) / "manifest.json")["cases"]}

    def fake_cli(command, config, *arguments):
        arguments = list(map(str, arguments))
        out = Path(arguments[arguments.index("--out") + 1])
        if command in {"bank", "run"}:
            keys = []
            for token in arguments[arguments.index("--keys") + 1:]:
                if token.startswith("--"):
                    break
                keys.append(token)
            assert len(keys) == 12
            assert Counter(cases[key]["question_type"] for key in keys) == {typ: 2 for typ in TYPES}
            if command == "bank":
                bank_keys.extend(keys)
            else:
                assert keys == bank_keys
            write(out / "summary.json", [{"key": key, "status": "ok"} for key in keys])
            filename = "bank.json" if command == "bank" else "M_final.json"
            for key in keys:
                write(out / key / filename, [{"q": "generated"}] if command == "bank" else [])
        elif command == "export":
            exported = []
            for key in bank_keys:
                for index in range(20):
                    row = state_row(key, f"{key}-{index}")
                    row["state"]["environment_fingerprint"] = fingerprint
                    exported.append(row)
            write_rows(out, exported)
        elif command == "probe":
            states = [json.loads(line) for line in Path(arguments[arguments.index("--states") + 1]).read_text().splitlines()]
            assert len(states) == 192
            assert Counter(r["state"]["case_key"] for r in states) == {key: 16 for key in bank_keys}
            assert Counter(r["family"] for r in states) == {typ: 32 for typ in TYPES}
            assert all(r["state"]["mode"] == "stream" for r in states)
            write(out / "summary.json", {"groups": [{"family": typ, "states": 32} for typ in TYPES]})
        else:
            pytest.fail(f"Unexpected CLI command: {command}")

    monkeypatch.setattr(runner, "_run_cli", fake_cli)
    result = runner.run(args)
    assert len(result["selected_cases"]) == 12
    assert sum(c["selected"] for c in result["state_counts"].values()) == 192
    assert {r["family"] for r in result["probe"]["groups"]} == set(TYPES)
