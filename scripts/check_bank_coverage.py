"""检查 bank 题库覆盖度：每个 case 的 Q 覆盖了多少 seed session、多少题型。

用法（从 repo 根目录运行）:
  python scripts/check_bank_coverage.py --bank outputs/bank --prepared data/prepared_longmemeval
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admem.common import connect, read


def report(case_dir: Path, full) -> str:
    bank = read(case_dir / "bank.json")
    cited_sessions = Counter()
    cited_rounds = set()
    types = Counter()
    for q in bank:
        types[q["type"]] += 1
        for rid in q["E"]:
            cited_rounds.add(rid)
            cited_sessions[full.rounds[rid].session_id] += 1
    total_sessions, total_rounds = len(full.sessions), len(full.rounds)
    uncited = [s for s in full.chronological_ids if s not in cited_sessions]
    lines = [
        f"== {case_dir.name} ==",
        f"  questions: {len(bank)}   types: {dict(types)}",
        f"  sessions cited: {len(cited_sessions)}/{total_sessions} "
        f"({len(cited_sessions) / max(1, total_sessions) * 100:.0f}%)",
        f"  rounds cited:   {len(cited_rounds)}/{total_rounds} "
        f"({len(cited_rounds) / max(1, total_rounds) * 100:.0f}%)",
        f"  top cited sessions: {cited_sessions.most_common(5)}",
        f"  uncited sessions: {len(uncited)}  {uncited[:8]}{'...' if len(uncited) > 8 else ''}",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", required=True)
    p.add_argument("--prepared", required=True)
    a = p.parse_args()
    memory, *_ = connect(".")
    bank_dir, prepared = Path(a.bank), Path(a.prepared)
    for case_dir in sorted(bank_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        bank_path = case_dir / "bank.json"
        full_path = prepared / "cases" / case_dir.name / "full.json"
        if not bank_path.is_file() or not full_path.is_file():
            print(f"== {case_dir.name} ==  missing bank.json or full.json, skip")
            continue
        print(report(case_dir, memory.FullMemory.load(full_path)))


if __name__ == "__main__":
    main()
