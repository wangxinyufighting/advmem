#!/usr/bin/env python3
"""Split official LongMemEval cases and extract real Builder transitions.

This script never puts the final LongMemEval question or answer into the
Builder prompt.  They are written to a separate labels file for offline
evaluation of Teacher candidates.

Splits must agree with ``admem.cli prepare``.  Pass ``--manifest`` (the prepared
``manifest.json``) to reuse its split for every question_id; this is required when
the outputs are mixed with admem prepared/bank/probe artifacts.  Without a
manifest, cases with identical histories are hashed as one group, so variants
sharing a history (e.g. ``_abs`` versions) can never land in different splits.

``--select-sessions evidence`` (default, as before) uses the official
``answer_session_ids``/``has_answer`` labels to choose which sessions become
transitions; the report marks this as label-informed.  ``--select-sessions all``
emits every session and uses no labels for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_cases(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        # Accept a wrapper such as {"data": [...]} when present.
        for key in ("data", "cases", "instances"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise SystemExit("LongMemEval input must be a JSON list of objects")
    return value


def history_group(case: dict[str, Any]) -> str:
    """Digest of the label-free history (dates + role/content), shared by identical histories."""
    sessions = [clean_session(s) for s in case.get("haystack_sessions", []) or []]
    value = [list(case.get("haystack_dates", []) or []), sessions]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def manifest_splits(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = {}
    for row in manifest.get("cases", []):
        split = {"val": "valid"}.get(row["split"], row["split"])
        splits[str(row["question_id"])] = split
    if not splits:
        raise SystemExit(f"{path} lists no cases")
    return splits


def split_name(question_id: str, seed: int, train_ratio: float, valid_ratio: float) -> str:
    """Hash split.  Callers pass the group's anchor question_id so shared histories stay together."""
    digest = hashlib.sha256(f"{seed}\0{question_id}".encode()).hexdigest()
    fraction = int(digest[:12], 16) / float(16**12)
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + valid_ratio:
        return "valid"
    return "test"


def clean_session(session: Any) -> list[dict[str, str]]:
    if not isinstance(session, list):
        return []
    turns: list[dict[str, str]] = []
    for turn in session:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            # Deliberately drop has_answer and every other evaluation label.
            turns.append({"role": role, "content": content})
    return turns


def evidence_indices(case: dict[str, Any], sessions: list[Any], session_ids: list[Any]) -> list[int]:
    by_id = {str(value): idx for idx, value in enumerate(session_ids)}
    selected: set[int] = set()
    for value in case.get("answer_session_ids", []) or []:
        if str(value) in by_id:
            selected.add(by_id[str(value)])

    # Fallback for files where answer_session_ids is absent but turn labels are
    # present.  Labels are used offline only and never enter the Builder prompt.
    if not selected:
        for idx, session in enumerate(sessions):
            if isinstance(session, list) and any(
                isinstance(turn, dict) and turn.get("has_answer") is True
                for turn in session
            ):
                selected.add(idx)
    return sorted(selected)


def render_prompt(
    question_type: str,
    question_date: Any,
    history_before: list[dict[str, Any]],
    incoming_session: dict[str, Any],
    question_meta: bool = True,
) -> str:
    history_text = json.dumps(history_before, ensure_ascii=False)
    incoming_text = json.dumps(incoming_session, ensure_ascii=False)
    meta = ("Question type: " + str(question_type) + "\n"
            "Question date: " + str(question_date) + "\n\n") if question_meta else ""
    return (
        "You are the memory Builder.\n\n"
        "Store only durable information from the new session. "
        "Do not answer a future question, do not write an answer rubric, "
        "and do not invent facts. Preserve provenance using the supplied "
        "session and turn identifiers. Return only a JSON object of the "
        'form {"ops": [...]}.\n\n'
        + meta +
        "Existing chronological session context:\n" + history_text + "\n\n"
        "New session to process:\n" + incoming_text + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--manifest", type=Path,
                        help="admem prepared manifest.json; reuse its split for every question_id")
    parser.add_argument("--select-sessions", choices=["evidence", "all"], default="evidence",
                        help="evidence: official evidence sessions only (label-informed); all: every session")
    parser.add_argument("--no-question-meta", action="store_true",
                        help="omit the official question type/date lines from the Builder prompt")
    parser.add_argument("--max-history-sessions", type=int, default=0,
                        help="keep only the last K previous sessions in history_before (0 = all)")
    args = parser.parse_args()
    if args.max_history_sessions < 0:
        raise SystemExit("--max-history-sessions must be >= 0")
    if args.train_ratio < 0 or args.valid_ratio < 0 or args.train_ratio + args.valid_ratio >= 1:
        raise SystemExit("train-ratio and valid-ratio must be nonnegative and sum to less than 1")

    cases = read_cases(args.input)
    assigned = manifest_splits(args.manifest) if args.manifest else None
    groups: dict[str, set[str]] = {}
    # 同一份历史用组内字典序最小的question_id做hash锚点：单例组的划分与旧版逐qid hash完全相同。
    anchors: dict[str, str] = {}
    for case in cases:
        qid = str(case.get("question_id", ""))
        if qid:
            group = history_group(case)
            anchors[group] = min(anchors.get(group, qid), qid)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    builder_cases: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    labels: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    skipped: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()

    for case in cases:
        question_id = str(case.get("question_id", ""))
        if not question_id:
            skipped.append({"reason": "missing_question_id"})
            continue
        if question_id in seen_question_ids:
            skipped.append({"question_id": question_id, "reason": "duplicate_question_id"})
            continue
        seen_question_ids.add(question_id)
        group = history_group(case)
        if assigned is not None:
            if question_id not in assigned:
                raise SystemExit(f"{question_id} is not in {args.manifest}; prepare and split the same data")
            split = assigned[question_id]
        else:
            split = split_name(anchors[group], args.seed, args.train_ratio, args.valid_ratio)
        groups.setdefault(group, set()).add(split)
        splits[split].append(case)

        question_type = str(case.get("question_type", ""))
        question_date = case.get("question_date")
        session_ids = list(case.get("haystack_session_ids", []) or [])
        dates = list(case.get("haystack_dates", []) or [])
        sessions = list(case.get("haystack_sessions", []) or [])
        selected = (evidence_indices(case, sessions, session_ids) if args.select_sessions == "evidence"
                    else list(range(len(sessions))))

        labels[split].append({
            "case_id": question_id,
            "question_id": question_id,
            "question_type": question_type,
            "question": case.get("question"),
            "answer": case.get("answer"),
            "question_date": question_date,
            "answer_session_ids": case.get("answer_session_ids", []),
        })

        # Abstention cases and cases without evidence are retained in the raw
        # split for final evaluation but do not create a positive memory-write
        # target.
        if (args.select_sessions == "evidence" and question_id.endswith("_abs")) or not selected:
            continue

        clean_sessions = [clean_session(session) for session in sessions]
        for idx in selected:
            if idx >= len(clean_sessions):
                continue
            session_id = str(session_ids[idx]) if idx < len(session_ids) else f"session_{idx}"
            session_date = dates[idx] if idx < len(dates) else None
            first = max(0, idx - args.max_history_sessions) if args.max_history_sessions else 0
            history_before = [
                {
                    "session_id": str(session_ids[j]) if j < len(session_ids) else f"session_{j}",
                    "date": dates[j] if j < len(dates) else None,
                    "turns": clean_sessions[j],
                }
                for j in range(first, idx)
            ]
            incoming = {
                "session_id": session_id,
                "date": session_date,
                "turns": clean_sessions[idx],
            }
            builder_cases[split].append({
                "case_id": f"{question_id}::session::{session_id}",
                "question_id": question_id,
                "question_type": question_type,
                "question_date": question_date,
                "session_id": session_id,
                "session_date": session_date,
                "prompt": render_prompt(question_type, question_date, history_before, incoming,
                                        question_meta=not args.no_question_meta),
                "state": {
                    "question_type": question_type,
                    "question_date": question_date,
                    "history_before": history_before,
                    "incoming_session": incoming,
                },
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (args.out_dir / f"{split}.json").write_text(
            json.dumps(splits[split], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for name, rows in (("builder", builder_cases[split]), ("labels", labels[split])):
            with (args.out_dir / f"{split}_{name}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.out_dir / "skipped.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skipped),
        encoding="utf-8",
    )
    report = {
        "input_cases": len(cases),
        "unique_question_ids": len(seen_question_ids),
        "split_cases": {name: len(rows) for name, rows in splits.items()},
        "builder_cases": {name: len(rows) for name, rows in builder_cases.items()},
        "label_cases": {name: len(rows) for name, rows in labels.items()},
        "skipped": len(skipped),
        "question_types": dict(Counter(str(row.get("question_type", "")) for row in cases)),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "split_source": "manifest" if args.manifest else "history_group_hash",
        "history_groups": len(groups),
        "groups_spanning_splits": sum(len(s) > 1 for s in groups.values()),
        "session_selection": args.select_sessions,
        "label_informed_selection": args.select_sessions == "evidence",
        "question_meta_in_prompt": not args.no_question_meta,
        "max_history_sessions": args.max_history_sessions,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
