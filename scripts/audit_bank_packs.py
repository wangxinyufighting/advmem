#!/usr/bin/env python3
"""Offline joint inspection of bank pack inputs, raw outputs and gate logs.

Recorded traces are authoritative snapshots. Legacy API-cache matches are only
unverified candidates, even when a single matching request remains. This script
never reconstructs prompts or invokes a model/tokenizer/embedding service.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from admem.audit_prompts import TYPES
from admem.common import digest, read, write
from memory import FullMemory


def bucket(row):
    return row.get("reason", "rejected") if row.get("status") == "rejected" else row.get("status", "unknown")


def select_packs(logs, per_status=2, seed=0):
    groups = defaultdict(set)
    for row in logs:
        groups[bucket(row)].add(row["pack"])
    chosen, rng = set(), random.Random(seed)
    for status in sorted(groups):
        candidates = sorted(groups[status])
        rng.shuffle(candidates)
        chosen.update(candidates[:per_status])
    return sorted(chosen)


def payload_from(messages):
    if not isinstance(messages, list):
        return None
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        try:
            value = json.loads(message.get("content", ""))
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("rounds"), list):
            return value
    return None


def compatible_payload(payload, full, context, rows):
    """A necessary case match, never proof of the original pack identity."""
    requested_types = {row.get("qtype") for row in rows} - {None}
    requested_types = requested_types or {context["question_type"]}
    if not payload or not payload["rounds"] or payload.get("type") not in requested_types:
        return False
    if context.get("question_date") and payload.get("question_date") != context["question_date"]:
        return False
    seeds = {row.get("seed") for row in rows} - {None}
    actual_seeds = {r.get("rid", "").split(":")[0] for r in payload["rounds"] if r.get("is_seed")}
    if seeds and actual_seeds != seeds:
        return False
    for shown in payload["rounds"]:
        rid = shown.get("rid")
        if rid not in full.rounds or not shown.get("messages"):
            return False
        original = full.rounds[rid]
        if shown.get("session_date") != full.sessions[original.session_id].date:
            return False
        # Exact ordered message subset, allowing compact's assistant elision.
        remaining = iter(original.messages)
        for message in shown["messages"]:
            if not any(m.get("role") == message.get("role") and m.get("content") == message.get("content")
                       for m in remaining):
                return False
    return True


def output_matches(raw, rows):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return any(r.get("status") == "invalid" for r in rows)
    items = value.get("items") if isinstance(value, dict) else None
    if not isinstance(items, list):
        return any(r.get("status") == "invalid" for r in rows)
    if any(r.get("status") == "empty" for r in rows) and items != []:
        return False
    logged = [r["item"] for r in rows if isinstance(r.get("item"), dict)]
    # Gate may repair E. It does not change q/a/type/date.
    fields = ("q", "a", "type", "question_date")
    return all(any(isinstance(q, dict) and all(q.get(k) == item.get(k) for k in fields)
                   for q in items) for item in logged)


def cache_requests(cache):
    if cache is None:
        return
    for path in sorted(Path(cache).glob("*.json")):
        try:
            entry = read(path)
            request = entry["request"]
            payload = payload_from(request.get("messages"))
            if payload is None:
                continue
            choice = entry["response"]["choices"][0]
            raw = choice["message"]["content"]
            if choice.get("finish_reason") == "length" and isinstance(raw, str):
                raw += "\n[TRUNCATED_OUTPUT]"
            yield {"cache_file": str(path), "request_body": request, "payload": payload, "raw": raw}
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            continue


def source_comparison(payload, full):
    """Show what was visible alongside original/adjacent history, without judging it."""
    if not payload:
        return []
    visible = {r.get("rid") for r in payload["rounds"]}
    result = []
    for shown in payload["rounds"]:
        rid = shown.get("rid")
        if rid not in full.rounds:
            continue
        original = full.rounds[rid]
        session = full.sessions[original.session_id]
        position = session.rids.index(rid)
        adjacent = session.rids[max(0, position - 1):position] + session.rids[position + 1:position + 2]
        unseen = [r for r in adjacent if r not in visible]
        messages = [{"role": m["role"], "content": m["content"]} for m in original.messages]
        result.append({"rid": rid, "session": rid.split(":")[0], "is_seed": shown.get("is_seed", False),
                       "visible_messages": shown.get("messages", []), "original_messages": messages,
                       "content_hidden": shown.get("messages") != messages,
                       "adjacent_unseen": [{"rid": r, "messages": full.rounds[r].messages} for r in unseen]})
    return result


def recorded_traces(case, pi, rows, full):
    referenced = {(case / r["trace_file"]).resolve() for r in rows if r.get("trace_file")}
    paths = set(referenced)
    paths.update(path.resolve() for path in (case / "pack_traces").glob(f"pack_{pi:05d}_attempt_*.json"))
    result = []
    for path in sorted(paths):
        # Path.is_relative_to and dict union both require Python 3.9.
        try:
            path.resolve().relative_to(case.resolve())
        except ValueError:
            continue
        if not path.is_file():
            continue
        trace = read(path)
        if trace.get("case_key") != case.name or trace.get("pack_index") != pi:
            continue
        if trace.get("prompt_hash") and digest(trace.get("prompt")) != trace["prompt_hash"]:
            raise ValueError(f"Trace prompt hash mismatch: {path}")
        request = trace.get("request_body")
        messages = request.get("messages") if isinstance(request, dict) else trace.get("prompt")
        payload = payload_from(messages)
        same_full = trace.get("full_hash") == full.fingerprint
        result.append({"trace_file": str(path), "snapshot": trace, "payload": payload,
                       "linked_to_bank_log": path.resolve() in referenced,
                       "input_source": "effective_request" if isinstance(request, dict) else "policy_prompt_snapshot",
                       "full_hash_matches": same_full,
                       "source_comparison": source_comparison(payload, full) if same_full else [],
                       "source_comparison_note": None if same_full else "Prepared history differs; source comparison omitted."})
    linked = [entry for entry in result if entry["linked_to_bank_log"]]
    if not linked:
        return result  # Unlinked snapshots remain explicitly unverified.
    def same_retry(entry, reference):
        a, b = entry["snapshot"], reference["snapshot"]
        fields = ("nonce", "prompt_hash", "full_hash", "environment_fingerprint")
        return (all(a.get(k) is not None and a.get(k) == b.get(k) for k in fields)
                and a.get("attempt", -1) <= b.get("attempt", -1))
    return [entry for entry in result if entry["linked_to_bank_log"]
            or any(same_retry(entry, reference) for reference in linked)]


def build_audits(bank_root, prepared, out, *, cache=None, question_types=None, keys=None,
                 per_status=2, seed=0, packs=None):
    bank_root, prepared, out = map(Path, (bank_root, prepared, out))
    if per_status < 1:
        raise ValueError("--per-status must be positive")
    if not bank_root.is_dir():
        raise ValueError(f"Bank directory does not exist: {bank_root}")
    if out.exists():
        raise ValueError(f"Output already exists; use a new --out: {out}")
    pending, found_keys = [], set()
    for case in sorted(p for p in bank_root.iterdir() if p.is_dir()):
        if keys and case.name not in keys:
            continue
        context_path = prepared / "cases" / case.name / "context.json"
        if not context_path.is_file():
            continue
        context = read(context_path)
        if question_types and context["question_type"] not in question_types:
            continue
        log_path = case / "bank_log.json"
        if not log_path.exists():
            log_path = case / "bank_log.partial.json"
        if not log_path.exists():
            continue
        logs = read(log_path)
        full = FullMemory.load(prepared / "cases" / case.name / "full.json")
        chosen = select_packs(logs, per_status, seed) if packs is None else sorted(set(packs))
        found_keys.add(case.name)
        for pi in chosen:
            rows = [r for r in logs if r["pack"] == pi]
            if not rows:
                raise ValueError(f"No log rows for {case.name} pack {pi}")
            traces = recorded_traces(case, pi, rows, full)
            no_send = all(r.get("status") == "type_infeasible" for r in rows)
            provenance = ("not_sent" if no_send else
                          "trace" if any(t["linked_to_bank_log"] for t in traces) else
                          "trace_candidates_unverified" if traces else "unavailable")
            pending.append({"case_key": case.name, "pack": pi, "question_type": context["question_type"],
                            "statuses": sorted({r.get("status", "unknown") for r in rows}),
                            "reasons": sorted({bucket(r) for r in rows}), "bank_log": rows,
                            "input_provenance": provenance, "traces": traces, "cache_candidates": [],
                            "_full": full, "_context": context})
    if keys and set(keys) - found_keys:
        raise ValueError("No matching bank logs for keys: " + ", ".join(sorted(set(keys) - found_keys)))
    if not pending:
        raise ValueError("No packs match the requested case/type filters")
    missing = [p for p in pending if p["input_provenance"] == "unavailable"]
    for candidate in cache_requests(cache) if missing else []:
        for pack in missing:
            if (compatible_payload(candidate["payload"], pack["_full"], pack["_context"], pack["bank_log"])
                    and output_matches(candidate["raw"], pack["bank_log"])):
                pack["cache_candidates"].append({**candidate,
                    "source_comparison": source_comparison(candidate["payload"], pack["_full"])})
                pack["input_provenance"] = "cache_candidates_unverified"
    index = {"schema": 1, "bank": str(bank_root), "selection_seed": seed, "per_status": per_status,
             "cache_status": "not_requested" if cache is None else "available" if Path(cache).is_dir() else "missing_directory",
             "packs": [], "notes": [
                 "Sampling is stratified by pack status/rejection reason; pack IDs are deduplicated.",
                 "Trace snapshots retain actual prior-question context; effective_request includes request overrides.",
                 "Legacy cache matches are unverified candidates, even if exactly one remains.",
                 "Unlinked trace files are unverified; linked traces include only matching earlier retries.",
                 "No prompt regeneration, official questions, model calls or automatic path-quality judgments are used."]}
    # Publish the complete report directory at once; a failed export must not
    # leave an output folder that prevents the same command from being retried.
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + ".tmp-", dir=str(out.parent)) as staging:
        staging = Path(staging)
        for pack in pending:
            pack.pop("_full")
            pack.pop("_context")
            pack["manual_review"] = {"cross_session_relation": None, "missing_context": None,
                                     "hidden_content_needed": None, "supported_question_possible": None,
                                     "gate_decision_reasonable": None, "notes": ""}
            report_file = Path(pack["case_key"]) / f"pack_{pack['pack']:05d}.json"
            write(staging / report_file, pack)
            entry = {k: pack[k] for k in ("case_key", "pack", "statuses", "reasons", "input_provenance")}
            entry["report_file"] = str(report_file)
            index["packs"].append(entry)
        write(staging / "index.json", index)
        staging.rename(out)
    return index


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", help="Optional legacy API cache, normally .cache/admem/api")
    p.add_argument("--question-types", nargs="+", choices=TYPES)
    p.add_argument("--keys", nargs="+")
    p.add_argument("--packs", nargs="+", type=int, help="Explicit zero-based pack indices; overrides sampling")
    p.add_argument("--per-status", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    try:
        result = build_audits(args.bank, args.prepared, args.out, cache=args.cache,
                              question_types=args.question_types, keys=args.keys, packs=args.packs,
                              per_status=args.per_status, seed=args.seed)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, f"Pack audit stopped: {exc}\n")
    print(json.dumps({"selected_packs": len(result["packs"]), "index": str(Path(args.out) / "index.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
