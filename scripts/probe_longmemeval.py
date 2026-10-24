#!/usr/bin/env python3
"""Probe Builder rewards on natural stream states from real LongMemEval history.

Relative paths are resolved from the repository root. Official questions are
isolated by ``prepare`` and are never used as reward tests. This runner invokes
the existing CLI; it does not create synthetic probes or bootstrap mutations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admem.audit_prompts import TYPES
from admem.common import Config, read, rows, write, write_rows
from admem.data import contexts


def _path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/type_aware_4b_nogate.json")
    p.add_argument("--data", default="data/longmemeval/longmemeval_s.json")
    p.add_argument("--prepared", default="data/prepared_longmemeval")
    p.add_argument("--sizes", nargs=3, type=int, default=[300, 50, 150])
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    cases = p.add_mutually_exclusive_group()
    cases.add_argument("--per-type", type=int, help="Cases per question type (default: 2 unless --keys/--case-limit is set)")
    cases.add_argument("--case-limit", type=int, help="Total cases; disables the default per-type quota")
    states = p.add_mutually_exclusive_group()
    states.add_argument("--states-per-case", type=int, help="Maximum probe states per case (default: 16)")
    states.add_argument("--state-limit", type=int, help="Global probe state budget instead of a per-case cap")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0, help="State selection seed; not the model sampling seed")
    p.add_argument("--keys", nargs="+")
    p.add_argument("--question-types", nargs="+", choices=TYPES, help="Case types to include (default: all six)")
    return p


def preflight(args):
    """Validate local inputs and role settings before starting any model request."""
    if args.per_type is None and args.case_limit is None and not args.keys:
        args.per_type = 2
    if args.states_per_case is None and args.state_limit is None:
        args.states_per_case = 16
    for name in ("per_type", "case_limit", "states_per_case", "state_limit", "samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be finite and nonnegative")
    if any(n < 0 for n in args.sizes):
        raise ValueError("--sizes must contain three nonnegative counts")
    for name in ("config", "data", "prepared", "out"):
        setattr(args, name, _path(getattr(args, name)))
    if args.out.exists():
        raise ValueError(f"Output already exists; choose a new --out: {args.out}")
    manifest = args.prepared / "manifest.json"
    if args.prepared.exists() and not manifest.is_file():
        raise ValueError(f"Prepared directory has no manifest.json; use a new --prepared or repair it: {args.prepared}")
    if not manifest.exists() and not args.data.is_file():
        raise ValueError(f"LongMemEval input does not exist: {args.data}")
    cfg = Config.load(args.config)
    cfg.project = str(_path(cfg.project))
    for filename in ("memory.py", "packs.py", "retrieve.py", "llm.py"):
        if not (Path(cfg.project) / filename).is_file():
            raise ValueError(f"config.project is missing {filename}: {cfg.project}")
    models = [("tokenizer", cfg.tokenizer)]
    if cfg.embedding == "local":
        models.append(("embed_model", cfg.embed_model or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")))
    if cfg.reranker:
        models.append(("reranker", cfg.reranker))
    for name, value in models:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing model setting: {name}")
        if (Path(value).is_absolute() or value.startswith(("./", "../", "~"))) and not _path(value).exists():
            raise ValueError(f"Local {name} path does not exist: {value}")
    roles = {}
    for role in ("BUILDER", "ATTACKER", "DEFENDER", "JUDGE"):
        model = os.getenv(role + "_MODEL", os.getenv("LLM_MODEL", ""))
        if not model.strip():
            raise ValueError(f"Set {role}_MODEL or the shared LLM_MODEL")
        url = os.getenv(role + "_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{role}_BASE_URL or LLM_BASE_URL must be an HTTP(S) URL")
        try:
            extra = json.loads(os.getenv(role + "_EXTRA_BODY", os.getenv("LLM_EXTRA_BODY", "{}")))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid JSON in {role}_EXTRA_BODY or LLM_EXTRA_BODY") from exc
        if not isinstance(extra, dict):
            raise ValueError(f"{role}_EXTRA_BODY or LLM_EXTRA_BODY must be a JSON object")
        roles[role] = model
    if cfg.embedding == "api" and not (cfg.embed_model or os.getenv("EMBED_MODEL")):
        raise ValueError("API embeddings require config.embed_model or EMBED_MODEL")
    return cfg, roles


def _run_cli(command, config, *arguments):
    subprocess.run([sys.executable, "-m", "admem.cli", command, "--config", str(config),
                    *map(str, arguments)], cwd=ROOT, check=True)


@contextmanager
def timed_stage(out, timings, name):
    """记录阶段耗时；进行中或失败的阶段也可从输出目录定位。"""
    started = time.perf_counter()
    entry = {"status": "running"}
    timings[name] = entry
    write(out / "stage_timings.json", timings)
    print(f"[stage] {name}: starting", flush=True)
    try:
        yield
    except BaseException:
        entry["status"] = "failed"
        raise
    else:
        entry["status"] = "ok"
    finally:
        entry["seconds"] = time.perf_counter() - started
        write(out / "stage_timings.json", timings)
        print(f"[stage] {name}: {entry['status']} ({entry['seconds']:.1f}s)", flush=True)


def selected_cases(args):
    selected, counts = [], Counter()
    required_types = args.question_types or TYPES
    # contexts(per_type=...) groups seven prepare strata, including abstention.
    # This runner deliberately applies its quota to the six actual question types.
    for context, full_path in contexts(args.prepared, args.split, args.case_limit,
                                       args.keys, args.question_types):
        qtype = context["question_type"]
        if qtype not in required_types:
            raise ValueError(f"Unexpected question type in prepared context: {qtype}")
        if args.per_type is not None and counts[qtype] >= args.per_type:
            continue
        full = read(full_path)
        if full.get("fingerprint") != context.get("full_hash"):
            raise ValueError(f"Prepared full/context hash mismatch: {context['key']}")
        selected.append({"key": context["key"], "question_type": context["question_type"],
                         "full_hash": context["full_hash"], "split": context["split"],
                         "sessions": len(full.get("sessions", full.get("history", {}).get("haystack_sessions", [])))})
        counts[qtype] += 1
    if args.per_type is not None:
        shortages = [f"{qtype}: {counts[qtype]}/{args.per_type}" for qtype in required_types
                     if counts[qtype] < args.per_type]
        if shortages:
            raise ValueError(f"Insufficient cases in split={args.split} for --per-type: " + ", ".join(shortages) +
                             "; check --keys/--question-types or choose a split with enough cases")
    if not selected:
        raise ValueError("No cases match --split/--keys/--question-types in the existing prepared split")
    if args.keys and set(args.keys) - {r["key"] for r in selected}:
        raise ValueError("Some --keys are absent from the selected split/type or exceed the case quota; adjust these flags")
    if args.state_limit is not None and args.state_limit < len(selected):
        raise ValueError("--state-limit must be at least the number of selected cases")
    return selected


def check_stage(folder, keys, *, bank=False):
    report = {r["key"]: r for r in read(Path(folder) / "summary.json")}
    for key in keys:
        if report.get(key, {}).get("status") != "ok":
            raise ValueError(f"{'bank' if bank else 'build'} did not complete for {key}; inspect {folder}/summary.json")
        path = Path(folder) / key / ("bank.json" if bank else "M_final.json")
        if not path.is_file():
            raise ValueError(f"Missing completed stage output: {path}")
        if bank and not read(path):
            raise ValueError(f"No accepted generated questions for {key}; inspect {folder}/{key}/bank_log.json")


def select_states(records, case_keys, limit, seed, fingerprint, *, full_hashes=None, split=None,
                  per_case_limit=None):
    """Filter before limiting, then sample reproducibly in case round-robin order."""
    grouped = {key: [] for key in case_keys}
    counts = {key: {"all": 0, "available": 0, "selected": 0} for key in case_keys}
    for row in records:
        state = row["state"]
        key = state["case_key"]
        if key not in grouped:
            raise ValueError(f"Unexpected case in collected states: {key}")
        if state["role"] != "builder" or state["mode"] != "stream":
            raise ValueError("Runner accepts only natural Builder stream states")
        if split is not None and state["split"] != split:
            raise ValueError("Collected state split differs from the selected split")
        if full_hashes is not None and state.get("full_hash") != full_hashes[key]:
            raise ValueError(f"Collected full-memory hash differs for {key}; regenerate these states")
        if row["prompt"] != state["prompt"] or state.get("environment_fingerprint") != fingerprint:
            raise ValueError("Collected state prompt/config differs; regenerate with the same code and config")
        counts[key]["all"] += 1
        if state.get("tests"):
            grouped[key].append(row)
            counts[key]["available"] += 1
    missing = [key for key in case_keys if not grouped[key]]
    if missing:
        raise ValueError("No reward-bearing stream states for " + ", ".join(missing) +
                         "; inspect generated bank evidence coverage and build outputs")
    rng = random.Random(seed)
    for group in grouped.values():
        group.sort(key=lambda row: row["id"])
        rng.shuffle(group)
        if per_case_limit is not None:
            del group[per_case_limit:]
    selected = []
    for index in range(max(map(len, grouped.values()))):
        for key in case_keys:
            if index < len(grouped[key]) and len(selected) < limit:
                selected.append(grouped[key][index])
                counts[key]["selected"] += 1
        if len(selected) >= limit:
            break
    return selected, counts


def run(args):
    cfg, roles = preflight(args)
    if not (args.prepared / "manifest.json").exists():
        # Count first so a wrong --sizes does not leave a partially prepared directory.
        from admem.data import iter_cases
        total = sum(1 for _ in iter_cases(args.data))
        if total != sum(args.sizes):
            raise ValueError(f"--sizes sum={sum(args.sizes)} but input contains {total} cases; choose matching explicit sizes")
        _run_cli("prepare", args.config, "--data", args.data, "--sizes", *args.sizes, "--out", args.prepared)
    cases = selected_cases(args)
    keys = [r["key"] for r in cases]
    types_by_key = {r["key"]: r["question_type"] for r in cases}
    type_counts = dict(Counter(types_by_key.values()))
    state_limit = args.state_limit if args.state_limit is not None else args.states_per_case * len(cases)
    args.out.mkdir(parents=True)
    write(args.out / "selected_cases.json", cases)
    print(json.dumps({"nature": "real_history_stream/generated_bank", "roles": roles,
                      "case_keys": keys, "case_counts_by_type": type_counts,
                      "split": args.split}, ensure_ascii=False), flush=True)
    common = ["--prepared", args.prepared, "--split", args.split, "--keys", *keys]
    bank, build, collect = (args.out / name for name in ("bank", "build", "collected"))
    timings = {}
    with timed_stage(args.out, timings, "bank"):
        _run_cli("bank", args.config, *common, "--out", bank)
        check_stage(bank, keys, bank=True)
    with timed_stage(args.out, timings, "build"):
        _run_cli("run", args.config, *common, "--mode", "build", "--builder-role", "BUILDER",
                 "--bank", bank, "--collect", collect, "--out", build)
        check_stage(build, keys)
    exported, states = args.out / "states_all.jsonl", args.out / "states.jsonl"
    with timed_stage(args.out, timings, "export"):
        _run_cli("export", args.config, "--states", collect, "--role", "builder", "--split", args.split, "--out", exported)
    selected, counts = select_states(rows(exported), keys, state_limit, args.seed, cfg.fingerprint(),
                                    full_hashes={r["key"]: r["full_hash"] for r in cases}, split=args.split,
                                    per_case_limit=args.states_per_case)
    # The family field controls reporting only; every state still uses the same reward.
    selected = [{**row, "family": types_by_key[row["state"]["case_key"]]} for row in selected]
    write_rows(states, selected)
    write(args.out / "state_selection.json", {"seed": args.seed, "limit": state_limit,
          "states_per_case": args.states_per_case,
          "per_case": counts, "selected_state_ids": [r["id"] for r in selected]})
    probe = args.out / "probe"
    with timed_stage(args.out, timings, "probe"):
        _run_cli("probe", args.config, "--states", states, "--samples", args.samples,
                 "--temperature", args.temperature, "--builder-role", "BUILDER", "--out", probe)
    summary = {"nature": "real_history_stream/generated_bank", "split": args.split,
               "selected_cases": cases, "case_counts_by_type": type_counts,
               "states_per_case": args.states_per_case, "state_limit": state_limit,
               "state_counts": counts, "samples_per_state": args.samples,
               "state_selection_seed": args.seed, "requested_temperature": args.temperature,
               "config": str(args.config), "environment_fingerprint": cfg.fingerprint(),
               "states": str(states), "probe_summary": str(probe / "summary.json"),
               "stage_timings": timings,
               "probe": read(probe / "summary.json")}
    write(args.out / "summary.json", summary)
    print(f"Real-history reward probe complete: {args.out / 'summary.json'}", flush=True)
    return summary


def main(argv=None):
    p = parser()
    try:
        run(p.parse_args(argv))
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(1, f"Real-history probe stopped: {exc}\n")


if __name__ == "__main__":
    main()
