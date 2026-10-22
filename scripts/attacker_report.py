"""对比若干 `admem.cli bank` 输出目录的 attacker 质量指标。

只读 bank 日志、失败记录及 prepared 的 context/full；不读 private_eval.jsonl。
用法：python scripts/attacker_report.py --prepared data/prepared_longmemeval runs/bank_old runs/bank_new
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from admem.audit_prompts import TYPES, personal_score  # noqa: E402
from memory import FullMemory  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))] if xs else None


def load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def case_diagnostics(key, qtype, logs, bank, *, included, failure=None, max_examples=3):
    """Pack counts and question rejections use separate denominators."""
    pack_statuses = defaultdict(set)
    packs = {}
    rejected, oracle_flags, support, screen = Counter(), Counter(), Counter(), Counter()
    examples, seen_reasons = [], set()
    for row in logs:
        status = row.get("status", "unknown")
        packs.setdefault(row["pack"], row)
        pack_statuses[status].add(row["pack"])
        if status == "rejected":
            rejected[row.get("reason", "unknown")] += 1
            oracle_flags.update(k for k, value in row.get("oracle", {}).items() if value is False)
            if row.get("support", {}).get("correct") is False:
                support[row["support"].get("reason", "unspecified")] += 1
            if row.get("screen", {}).get("stable") is False:
                screen[row["screen"].get("reason", "unspecified")] += 1
        reason = row.get("reason", status)
        if (status in {"rejected", "invalid", "transport_error"}
                and reason not in seen_reasons and len(examples) < max_examples):
            seen_reasons.add(reason)
            item = row.get("item") or {}
            examples.append({"pack": row["pack"], "status": status, "reason": str(reason)[:400],
                             "question": str(item.get("q", ""))[:400], "evidence": item.get("E", []),
                             "oracle_false_flags": [k for k, v in row.get("oracle", {}).items() if v is False],
                             "support_reason": str(row.get("support", {}).get("reason", ""))[:400],
                             "screen_reason": str(row.get("screen", {}).get("reason", ""))[:400]})
    return {"key": key, "question_type": qtype, "included_in_metrics": included,
            "status": "complete" if included else "failed_or_partial",
            "failure": failure, "bank_questions": len(bank),
            "packs": len(packs),
            "pack_status_counts": {s: len(ids) for s, ids in sorted(pack_statuses.items())},
            "pack_session_counts": dict(Counter(r["pack_sessions"] for r in packs.values()
                                                if "pack_sessions" in r)),
            "visible_session_counts": dict(Counter(r["visible_sessions"] for r in packs.values()
                                                   if "visible_sessions" in r)),
            "questions_proposed": sum(r.get("status") in {"accepted", "rejected"} for r in logs),
            "gate_accepted_questions": sum(r.get("status") == "accepted" for r in logs),
            "rejection_reasons": dict(rejected), "oracle_false_flags": dict(oracle_flags),
            "support_reasons_top10": [{"reason": str(r)[:400], "count": n} for r, n in support.most_common(10)],
            "screen_reasons_top10": [{"reason": str(r)[:400], "count": n} for r, n in screen.most_common(10)],
            "examples": examples}


def summarize(root: Path, prepared: Path, question_types=None, max_examples=3) -> dict:
    stats = Counter()
    by_type = defaultdict(Counter)
    tokens, gini_like = [], []
    details, failed = [], 0
    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        complete = all((case / name).is_file() for name in ("bank_log.json", "bank.json"))
        log_path = case / ("bank_log.json" if (case / "bank_log.json").is_file() else "bank_log.partial.json")
        bank_path = case / ("bank.json" if (case / "bank.json").is_file() else "bank.partial.json")
        failure = load(case / "failed.json", None)
        if not log_path.exists() and not bank_path.exists() and failure is None:
            continue
        logs, bank = load(log_path, []), load(bank_path, [])
        context = load(prepared / "cases" / case.name / "context.json", {})
        known_types = {r.get("case_question_type") or r.get("qtype") for r in logs} - {None}
        known_types.update(q["type"] for q in bank)
        qtype = context.get("question_type", next(iter(known_types)) if len(known_types) == 1 else "?")
        if question_types and qtype not in question_types:
            continue
        included = complete and failure is None
        details.append(case_diagnostics(case.name, qtype, logs, bank, included=included,
                                        failure=failure, max_examples=max_examples))
        if not included:
            failed += 1
            continue  # Partial banks are diagnostic evidence, not completed-run metrics.
        full = FullMemory.load(prepared / "cases" / case.name / "full.json")
        packs = {}
        for row in logs:
            packs.setdefault(row["pack"], row)
            t = row.get("qtype") or row.get("case_question_type") or qtype
            status = row["status"]
            if status in {"type_infeasible", "invalid", "empty", "transport_error"}:
                by_type[t][status] += 1
            else:
                by_type[t]["accepted" if status == "accepted" else "rejected"] += 1
        stats["cases"] += 1
        stats["packs"] += len(packs)
        tokens += [r["prompt_tokens"] for r in packs.values() if "prompt_tokens" in r]
        rids, esets, sessions = Counter(), Counter(), Counter()
        for q in bank:
            stats["bank"] += 1
            esets[tuple(sorted(q["E"]))] += 1
            for rid in set(q["E"]):
                rids[rid] += 1
            for sn in {rid.split(":")[0] for rid in q["E"]}:
                sessions[sn] += 1
            if q["type"] != "single-session-assistant" and personal_score(full, q["E"]) == 0:
                stats["impersonal"] += 1
        stats["same_E_repeat"] += sum(n - 1 for n in esets.values())
        stats["distinct_evidence_rounds"] += len(rids)
        if sessions:
            gini_like.append(max(sessions.values()) / sum(sessions.values()))
    accepted = sum(c["accepted"] for c in by_type.values())
    asked = accepted + sum(c["rejected"] for c in by_type.values())
    infeasible = sum(c["type_infeasible"] for c in by_type.values())
    invalid = sum(c["invalid"] for c in by_type.values())
    transport_errors = sum(c["transport_error"] for c in by_type.values())
    bank = stats["bank"] or 1
    return {
        "cases": stats["cases"], "failed_cases": failed,
        "packs": stats["packs"], "type_infeasible_packs": infeasible,
        "transport_error_packs": transport_errors,
        "invalid_output_rate": round(invalid / max(1, stats["packs"] - infeasible), 3),
        "bank_per_feasible_pack": round(stats["bank"] / max(1, stats["packs"] - infeasible), 3),
        "prompt_tokens_p50/p90/max": [pct(tokens, .5), pct(tokens, .9), max(tokens, default=None)],
        "questions_proposed": asked, "gate_accept_rate": round(accepted / asked, 3) if asked else None,
        "bank_questions": stats["bank"],
        "bank_per_pack": round(stats["bank"] / max(1, stats["packs"]), 3),
        "distinct_rounds_per_question": round(stats["distinct_evidence_rounds"] / bank, 3),
        "same_E_repeat_rate": round(stats["same_E_repeat"] / bank, 3),
        "top_session_share_mean": round(statistics.mean(gini_like), 3) if gini_like else None,
        "impersonal_rate": round(stats["impersonal"] / bank, 3),
        "by_type": {t: dict(c) for t, c in sorted(by_type.items())},
        "case_details": details,
        "metric_notes": {
            "pack_status_counts": "Distinct packs per status; a pack can contain both accepted and rejected questions, so status counts may overlap.",
            "partial_cases": "Failed/incomplete cases appear in case_details only and are excluded from quality metrics.",
            "bank_questions": "Deduplicated saved questions; gate_accepted_questions can be larger.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", help="把完整结果另存为JSON")
    ap.add_argument("--question-types", nargs="+", choices=TYPES, help="仅分析指定 case 题型；按 context.json 筛选")
    ap.add_argument("--max-examples", type=int, default=3, help="每个 case 最多展示几条失败示例（默认3）")
    a = ap.parse_args()
    if a.max_examples < 0:
        ap.error("--max-examples must be nonnegative")
    results = {run: summarize(Path(run), Path(a.prepared), a.question_types, a.max_examples) for run in a.runs}
    if a.json:
        Path(a.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["cases", "failed_cases", "packs", "type_infeasible_packs", "transport_error_packs", "prompt_tokens_p50/p90/max", "invalid_output_rate",
            "gate_accept_rate", "bank_per_feasible_pack", "distinct_rounds_per_question",
            "same_E_repeat_rate", "top_session_share_mean", "impersonal_rate"]
    print("| metric | " + " | ".join(Path(r).name for r in a.runs) + " |")
    print("|---" * (len(a.runs) + 1) + "|")
    for c in cols:
        print(f"| {c} | " + " | ".join(str(results[r][c]) for r in a.runs) + " |")
    for r in a.runs:
        print(f"\n{Path(r).name} by_type: {json.dumps(results[r]['by_type'], ensure_ascii=False)}")
        for detail in results[r]["case_details"]:
            print(json.dumps(detail, ensure_ascii=False))


if __name__ == "__main__":
    main()
