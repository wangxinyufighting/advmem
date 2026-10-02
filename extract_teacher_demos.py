#!/usr/bin/env python3
"""Extract one high-quality Teacher demonstration per probe state.

Inputs are JSONL files produced by ``probe-suite`` and ``probe``.
The output is JSONL with the fields ``prompt`` and ``completion`` plus
traceability metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_FAMILIES = {"merge_timeline", "dedup_refine", "noop_repeat"}


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"Expected an object in {path}:{line_no}")
            rows.append(value)
    return rows


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 10**9) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, help="probe-suite probes.jsonl")
    parser.add_argument("--samples", required=True, help="probe samples.jsonl")
    parser.add_argument("--out", required=True, help="output demonstrations JSONL")
    parser.add_argument(
        "--families",
        nargs="+",
        default=sorted(DEFAULT_FAMILIES),
        help="families to extract (default: merge_timeline dedup_refine noop_repeat)",
    )
    args = parser.parse_args()
    families = set(args.families)

    states = read_jsonl(args.states)
    samples = read_jsonl(args.samples)

    state_by_id: dict[str, dict[str, Any]] = {}
    for row in states:
        state_id = row.get("state_id")
        family = row.get("family")
        if state_id is None:
            raise SystemExit("Each state row must contain state_id")
        if family in families:
            state_by_id[str(state_id)] = row

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for sample in samples:
        family = sample.get("family")
        state_id = str(sample.get("state_id", ""))
        if family not in families or state_id not in state_by_id:
            continue

        # These are the same gates used by the probe summary.  Requiring all
        # of them prevents an answer-correct but family-invalid patch from
        # becoming an SFT target.
        if not all(
            sample.get(name) is True
            for name in ("success", "legal", "faith", "family_constraint_pass")
        ):
            skipped += 1
            continue

        completion = sample.get("raw_text")
        if not isinstance(completion, str) or not completion.strip():
            skipped += 1
            continue

        reward = as_float(sample.get("effective_reward"))
        if reward is None:
            skipped += 1
            continue

        candidates[(state_id, str(family))].append(sample)

    selected: list[dict[str, Any]] = []
    missing: list[tuple[str, str]] = []
    for state_id, state in state_by_id.items():
        family = str(state.get("family"))
        key = (state_id, family)
        options = candidates.get(key, [])
        if not options:
            missing.append(key)
            continue

        # Prefer reward; use shorter output and then lower sample index as
        # deterministic tie-breakers.
        chosen = max(
            options,
            key=lambda row: (
                as_float(row.get("effective_reward"), float("-inf")),
                -as_int(row.get("after_tokens")),
                -as_int(row.get("sample_idx")),
            ),
        )
        selected.append(
            {
                "family": family,
                "state_id": state_id,
                "prompt": state.get("prompt"),
                "completion": chosen["raw_text"],
                "effective_reward": as_float(chosen.get("effective_reward")),
                "reward": as_float(chosen.get("reward")),
                "raw_hash": chosen.get("raw_hash"),
                "sample_idx": chosen.get("sample_idx"),
            }
        )

    selected.sort(key=lambda row: (row["family"], row["state_id"]))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(selected)} demonstrations to {output}")
    print(f"skipped unsuccessful/invalid samples: {skipped}")
    if missing:
        print("missing successful candidate for:")
        for state_id, family in missing:
            print(f"  {family} {state_id}")


if __name__ == "__main__":
    main()
