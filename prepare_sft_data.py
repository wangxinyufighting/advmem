#!/usr/bin/env python3
"""Prepare prompt/completion SFT data from admem probe outputs.

The input files are the JSONL files produced by ``probe-suite`` and ``probe``.
The script:

* keeps only successful, legal, faithful, family-valid candidates;
* maps probe-suite states to sample state_ids, including older files where the
  state rows do not contain an explicit state_id;
* selects one best completion per state (highest effective_reward);
* removes duplicate completions;
* splits whole states into train/validation sets; and
* saves rejected candidates separately for later preference training.

The positive output is deliberately in simple ``prompt``/``completion`` JSONL
format so it can be adapted to the project's existing trainer.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_FAMILIES = {
    "append_event",
    "update_preserve_old",
    "merge_timeline",
    "dedup_refine",
    "assistant_list",
    "preference_context",
    "temporal_anchor",
    "noop_repeat",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"expected an object in {path}:{line_no}")
            rows.append(row)
    return rows


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: Any, default: int = 10**9) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def explicit_ids(row: dict[str, Any]) -> list[str]:
    """Find IDs when a probe-suite version stores them in state rows."""
    result: list[str] = []
    for key in ("state_id", "id"):
        if row.get(key) is not None:
            result.append(str(row[key]))
    nested = row.get("state")
    if isinstance(nested, dict):
        for key in ("state_id", "id"):
            if nested.get(key) is not None:
                result.append(str(nested[key]))
    return result


def completion_json(raw: Any) -> tuple[str | None, str | None]:
    """Validate and normalize an operation JSON completion."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty_completion"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None, "completion_not_json"
    if not isinstance(value, dict) or not isinstance(value.get("ops"), list):
        return None, "completion_missing_ops"
    # Compact, deterministic JSON is easier for a trainer to learn.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")), None


def build_state_map(
    state_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    families: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (sample state_id, family) to the corresponding state row.

    Current probe-suite files can omit state_id from probes.jsonl.  The probe
    runner emits samples in state order, so first-seen sample IDs are matched
    to state rows in order within each family.
    """
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        family = str(row.get("family", ""))
        if family in families:
            rows_by_family[family].append(row)

    ids_by_family: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        family = str(sample.get("family", ""))
        state_id = sample.get("state_id")
        if family not in families or state_id is None:
            continue
        state_id = str(state_id)
        if state_id not in seen[family]:
            seen[family].add(state_id)
            ids_by_family[family].append(state_id)

    state_map: dict[tuple[str, str], dict[str, Any]] = {}
    for family in families:
        rows = rows_by_family.get(family, [])
        ids = ids_by_family.get(family, [])
        if len(rows) != len(ids):
            raise SystemExit(
                f"{family}: probes have {len(rows)} states but samples have "
                f"{len(ids)} distinct state_ids; check that the files match"
            )

        id_set = set(ids)
        used: set[str] = set()
        unresolved: list[dict[str, Any]] = []
        for row in rows:
            matches = [value for value in explicit_ids(row) if value in id_set]
            if matches:
                state_map[(matches[0], family)] = row
                used.add(matches[0])
            else:
                unresolved.append(row)

        remaining = [state_id for state_id in ids if state_id not in used]
        if len(unresolved) != len(remaining):
            raise SystemExit(
                f"{family}: cannot align state rows with sample state_ids"
            )
        for state_id, row in zip(remaining, unresolved):
            state_map[(state_id, family)] = row
    return state_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="families to include; default is every family present in --states",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    if not 0.0 <= args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1")

    state_rows = read_jsonl(args.states)
    samples = read_jsonl(args.samples)
    if args.families is None:
        families = {
            str(row.get("family"))
            for row in state_rows
            if row.get("family") is not None
        }
    else:
        families = set(args.families)
    if not families:
        raise SystemExit("no families found in --states")
    state_map = build_state_map(state_rows, samples, families)

    positives: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()

    for sample in samples:
        family = str(sample.get("family", ""))
        state_id = sample.get("state_id")
        if family not in families or state_id is None:
            continue
        state_id = str(state_id)
        key = (state_id, family)
        state = state_map.get(key)
        if state is None:
            continue

        completion, completion_error = completion_json(sample.get("raw_text"))
        gates = ("success", "legal", "faith", "family_constraint_pass")
        failed_gates = [gate for gate in gates if sample.get(gate) is not True]
        if completion_error:
            failed_gates.append(completion_error)

        if failed_gates:
            reason = ",".join(failed_gates)
            reject_reasons[reason] += 1
            rejected.append(
                {
                    "family": family,
                    "state_id": state_id,
                    "prompt": state.get("prompt"),
                    "completion": sample.get("raw_text"),
                    "reason": reason,
                    "faith_reason": sample.get("faith_reason"),
                    "family_constraint_reason": sample.get("family_constraint_reason"),
                    "sample_idx": sample.get("sample_idx"),
                }
            )
            continue

        candidate = {
            "family": family,
            "state_id": state_id,
            "prompt": state.get("prompt"),
            "completion": completion,
            "effective_reward": number(sample.get("effective_reward")),
            "reward": number(sample.get("reward")),
            "raw_hash": sample.get("raw_hash"),
            "sample_idx": sample.get("sample_idx"),
            "after_tokens": sample.get("after_tokens"),
        }
        score = (
            candidate["effective_reward"],
            -integer(candidate["after_tokens"]),
            -integer(candidate["sample_idx"]),
        )
        previous = positives.get(key)
        if previous is None or score > previous["_score"]:
            candidate["_score"] = score
            positives[key] = candidate

    records = list(positives.values())
    for record in records:
        record.pop("_score", None)
        record.pop("after_tokens", None)

    # Split by complete states, stratified by family.  No state can appear in
    # both train and validation, even when it had multiple Teacher samples.
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["family"]].append(record)

    rng = random.Random(args.seed)
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        rng.shuffle(rows)
        if len(rows) <= 1 or args.val_ratio == 0:
            val_count = 0
        else:
            val_count = max(1, round(len(rows) * args.val_ratio))
            val_count = min(val_count, len(rows) - 1)
        valid.extend(rows[:val_count])
        train.extend(rows[val_count:])

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    write_jsonl(args.out_dir / "rejected.jsonl", rejected)

    stats = {
        "states_in_probe_file": len(state_map),
        "positive_records": len(records),
        "train_records": len(train),
        "valid_records": len(valid),
        "rejected_records": len(rejected),
        "families": sorted(families),
        "positive_by_family": dict(Counter(row["family"] for row in records)),
        "train_by_family": dict(Counter(row["family"] for row in train)),
        "valid_by_family": dict(Counter(row["family"] for row in valid)),
        "reject_reasons": dict(reject_reasons),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
