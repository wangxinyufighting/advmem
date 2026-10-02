#!/usr/bin/env python3
"""Split official LongMemEval cases and extract real Builder transitions.

This script never puts the final LongMemEval question or answer into the
Builder prompt.  They are written to a separate labels file for offline
evaluation of Teacher candidates.  Splitting happens first by question_id, so
all transitions derived from one LongMemEval question stay in one split.
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


def split_name(question_id: str, seed: int, train_ratio: float, valid_ratio: float) -> str:
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
) -> str:
    history_text = json.dumps(history_before, ensure_ascii=False)
    incoming_text = json.dumps(incoming_session, ensure_ascii=False)
    return (
        "You are the memory Builder.\n\n"
        "Store only durable information from the new session. "
        "Do not answer a future question, do not write an answer rubric, "
        "and do not invent facts. Preserve provenance using the supplied "
        "session and turn identifiers. Return only a JSON object of the "
        'form {"ops": [...]}.\n\n'
        "Question type: " + str(question_type) + "\n"
        "Question date: " + str(question_date) + "\n\n"
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
    args = parser.parse_args()
    if args.train_ratio < 0 or args.valid_ratio < 0 or args.train_ratio + args.valid_ratio >= 1:
        raise SystemExit("train-ratio and valid-ratio must be nonnegative and sum to less than 1")

    cases = read_cases(args.input)
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
        split = split_name(question_id, args.seed, args.train_ratio, args.valid_ratio)
        splits[split].append(case)

        question_type = str(case.get("question_type", ""))
        question_date = case.get("question_date")
        session_ids = list(case.get("haystack_session_ids", []) or [])
        dates = list(case.get("haystack_dates", []) or [])
        sessions = list(case.get("haystack_sessions", []) or [])
        selected = evidence_indices(case, sessions, session_ids)

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
        if question_id.endswith("_abs") or not selected:
            continue

        clean_sessions = [clean_session(session) for session in sessions]
        for idx in selected:
            if idx >= len(clean_sessions):
                continue
            session_id = str(session_ids[idx]) if idx < len(session_ids) else f"session_{idx}"
            session_date = dates[idx] if idx < len(dates) else None
            history_before = [
                {
                    "session_id": str(session_ids[j]) if j < len(session_ids) else f"session_{j}",
                    "date": dates[j] if j < len(dates) else None,
                    "turns": clean_sessions[j],
                }
                for j in range(idx)
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
                "prompt": render_prompt(question_type, question_date, history_before, incoming),
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
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
