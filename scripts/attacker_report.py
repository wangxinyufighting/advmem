"""对比若干 `admem.cli bank` 输出目录的 attacker 质量指标。

只读 bank_log.json / bank.json 和 prepared 的 full.json；不读 private_eval.jsonl。
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
from admem.audit_prompts import personal_score  # noqa: E402
from memory import FullMemory  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))] if xs else None


def summarize(root: Path, prepared: Path) -> dict:
    stats = Counter()
    by_type = defaultdict(Counter)
    tokens, gini_like = [], []
    for case in sorted(p for p in root.iterdir() if (p / "bank_log.json").exists()):
        logs = json.loads((case / "bank_log.json").read_text(encoding="utf-8"))
        bank = json.loads((case / "bank.json").read_text(encoding="utf-8"))
        full = FullMemory.load(prepared / "cases" / case.name / "full.json")
        packs = {}
        for row in logs:
            packs.setdefault(row["pack"], row)
            t = row.get("qtype", "?")
            status = row["status"]
            if status in {"type_infeasible", "invalid", "empty"}:
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
    bank = stats["bank"] or 1
    return {
        "cases": stats["cases"], "packs": stats["packs"],
        "prompt_tokens_p50/p90/max": [pct(tokens, .5), pct(tokens, .9), max(tokens, default=None)],
        "questions_proposed": asked, "gate_accept_rate": round(accepted / asked, 3) if asked else None,
        "bank_questions": stats["bank"],
        "bank_per_pack": round(stats["bank"] / max(1, stats["packs"]), 3),
        "distinct_rounds_per_question": round(stats["distinct_evidence_rounds"] / bank, 3),
        "same_E_repeat_rate": round(stats["same_E_repeat"] / bank, 3),
        "top_session_share_mean": round(statistics.mean(gini_like), 3) if gini_like else None,
        "impersonal_rate": round(stats["impersonal"] / bank, 3),
        "by_type": {t: dict(c) for t, c in sorted(by_type.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for run in a.runs:
        print(run)
        print(json.dumps(summarize(Path(run), Path(a.prepared)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
