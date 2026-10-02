"""批量评测 Full memory → 检索 → reader；复用原项目 answer/grade，不改变判分prompt。"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from agents import answer, grade
from common import normalize_answer
from metrics import Gold
from packs import TYPES

from _benchmark import (BudgetExhausted, Runtime, attempt, bool_stats, checkpoint, common_args, error_text,
                        init_run, jsonl, load_full, mean, print_selection, save, selected_data,
                        selection, table)


def evidence_metrics(full, gold, rids):
    sessions = {full.rounds[r].session_id for r in rids}
    def score(actual, target):
        if gold.abstention or not target:
            return {"all": None, "recall": None}
        return {"all": target <= actual, "recall": len(target & actual) / len(target)}
    s, r = score(sessions, gold.sessions), score(set(rids), gold.rounds)
    return {"all_sessions": s["all"], "session_recall": s["recall"],
            "all_rounds": r["all"], "round_recall": r["recall"],
            "gold_session_ids": sorted(gold.sessions), "gold_round_ids": sorted(gold.rounds)}


def run_case(args, meta, case, rt, path):
    state = checkpoint(path, meta, args, rt.run_id)
    if state.get("status") in {"ok", "retrieval_only"}:
        return state
    state.update(question=case["question"], reference=str(case["answer"]),
                 question_date=case["question_date"])
    stage = "prepare"
    try:
        full = load_full(args, meta, case)
        gold = Gold.from_case(case, full)
        if "retrieval" not in state:
            stage = "retrieval"
            n = attempt(state, stage, path)
            scope = f"{meta['question_id']}:retrieval:{n}"
            index = rt.index(full)
            planner = rt.client("PLANNER", scope) if args.steps else None
            hits, trace = index.retrieve(gold.question, gold.date, args.k, planner=planner,
                                         steps=args.steps, expand=args.expand, max_chars=args.context_chars)
            context, visible = index.context(hits, args.context_chars)
            rids = sorted({r for h in visible for r in index.docs[h.id].prov})
            state["retrieval"] = {"hits": [asdict(h) for h in hits],
                                  "visible_ids": [h.id for h in visible], "visible_rids": rids,
                                  "context": context, "context_chars": len(context), "trace": trace,
                                  "evidence": evidence_metrics(full, gold, rids)}
            save(path, state)
            (path.parent / "retrieved_context.txt").write_text(context, encoding="utf-8")
        if args.retrieval_only:
            state["status"] = "retrieval_only"
        else:
            if "hypothesis" not in state:
                stage = "reader"
                n = attempt(state, stage, path)
                reader = rt.client("DEFENDER", f"{meta['question_id']}:reader:{n}")
                # Reader 只读问题、日期、实际检索上下文；reference 不传入。
                state["hypothesis"] = answer(reader, gold.question, gold.date, state["retrieval"]["context"])
                state["normalized_exact_match"] = normalize_answer(state["hypothesis"]) == normalize_answer(gold.answer)
                save(path, state)
            if "judge" not in state:
                stage = "judge"
                n = attempt(state, stage, path)
                judge = rt.client("JUDGE", f"{meta['question_id']}:judge:{n}")
                result = grade(judge, gold.question, gold.answer, state["hypothesis"], gold.qtype, gold.abstention)
                if type(result.get("correct")) is not bool:
                    raise ValueError("judge.correct 必须为布尔值")
                state["judge"] = result
            state["status"] = "ok"
        state.pop("error", None)
        state.pop("error_stage", None)
    except BudgetExhausted:
        state["status"] = "pending"
        save(path, state)
        raise
    except Exception as exc:
        # KeyboardInterrupt 不在 Exception 内；Ctrl-C 后保留已完成的检索/回答。
        state.update(status="error", error_stage=stage, error=error_text(exc))
    save(path, state)
    return state


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["question_type"]].append(row)
        groups["ALL_MICRO"].append(row)
    out = []
    for name, rows in sorted(groups.items()):
        stats = bool_stats([r.get("judge", {}).get("correct") for r in rows])
        retrieved = [r for r in rows if "retrieval" in r]
        es = [r["retrieval"]["evidence"] for r in retrieved]
        complete_e = [r for r in rows if r.get("retrieval", {}).get("evidence", {}).get("all_rounds") is True]
        conditional = bool_stats([r.get("judge", {}).get("correct") for r in complete_e])
        out.append({"group": name, "selected": stats["selected"], "judged": stats["completed"],
                    "correct": stats["successes"], "incorrect": stats["failures"], "unknown": stats["unknown"],
                    "error_cases": sum(r.get("status") == "error" for r in rows),
                    "pending_cases": sum(r.get("status") == "pending" for r in rows),
                    "answered": sum("hypothesis" in r for r in rows),
                    "accuracy_completed": stats["rate_completed"],
                    "accuracy_lower_all_selected": stats["lower_all_selected"],
                    "accuracy_upper_all_selected": stats["upper_all_selected"],
                    "retrieval_completed": len(retrieved),
                    "evidence_labeled_cases": sum(e["all_rounds"] is not None for e in es),
                    "all_rounds_retrieved_rate": mean(e["all_rounds"] for e in es),
                    "mean_round_recall": mean(e["round_recall"] for e in es),
                    "all_sessions_retrieved_rate": mean(e["all_sessions"] for e in es),
                    "mean_context_chars": mean(r["retrieval"]["context_chars"] for r in retrieved),
                    "judged_with_full_evidence": conditional["completed"],
                    "accuracy_given_full_evidence": conditional["rate_completed"]})
    for name, include_abs in [("ANSWERABLE_MACRO", False), ("ALL_TYPES_MACRO", True)]:
        typed = [r for r in out if r["group"] in [*TYPES, *(["abstention"] if include_abs else [])]]
        if typed:
            out.append({"group": name, "included_types": len(typed),
                        "types_with_judgments": sum(r["judged"] > 0 for r in typed),
                        "accuracy_completed": mean(r["accuracy_completed"] for r in typed),
                        "accuracy_lower_all_selected": mean(r["accuracy_lower_all_selected"] for r in typed),
                        "accuracy_upper_all_selected": mean(r["accuracy_upper_all_selected"] for r in typed)})
    return out


def flush(args, records, rt):
    out = Path(args.out)
    groups = summarize(records)
    save(out / "summary.json", {"judge_protocol": "existing agents.grade; NOT official evaluation execution",
                               "groups": groups, "current_process_api": rt.stats()})
    table(out / "summary.csv", groups)
    jsonl(out / "cases.jsonl", records)
    # 一行一个 question_id；中断/恢复也不产生重复预测。未知判分不会删掉有效回答。
    predictions = [{"question_id": r["question_id"], "hypothesis": r["hypothesis"]}
                   for r in records if "hypothesis" in r]
    jsonl(out / "predictions.jsonl", predictions)
    compact = [{"case_index": r["case_index"], "question_id": r["question_id"],
                "question_type": r["question_type"], "status": r.get("status"),
                "correct": r.get("judge", {}).get("correct"),
                "round_recall": r.get("retrieval", {}).get("evidence", {}).get("round_recall"),
                "error_stage": r.get("error_stage"), "error": r.get("error")}
               for r in records]
    table(out / "cases.csv", compact)


def main(argv=None):
    p = common_args(__doc__)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--expand", type=int, default=1)
    p.add_argument("--context-chars", type=int, default=80000)
    p.add_argument("--retrieval-only", action="store_true")
    args = p.parse_args(argv)
    if args.k < 1 or min(args.steps, args.expand, args.context_chars) < 0:
        p.error("k必须为正；steps/expand/context-chars不能为负")
    rows = selection(args)
    print_selection(rows)
    if args.list:
        return
    run_id = init_run(args, rows)
    rt = Runtime(args, run_id)
    records = [{**r, "status": "pending"} for r in rows]
    try:
        for i, (meta, case) in enumerate(selected_data(args, rows)):
            path = Path(args.out) / "cases" / f"{meta['case_index']:04d}" / "case_report.json"
            print(f"[{i + 1}/{len(rows)}] reader {meta['question_id']} {meta['question_type']}", flush=True)
            try:
                records[i] = run_case(args, meta, case, rt, path)
            except (KeyboardInterrupt, BudgetExhausted):
                if path.exists():
                    from common import read_json
                    records[i] = read_json(path)
                raise
            flush(args, records, rt)
            r = records[i]
            print(f"  status={r['status']} correct={r.get('judge', {}).get('correct')} "
                  f"error={r.get('error', '')}", flush=True)
    except (KeyboardInterrupt, BudgetExhausted) as exc:
        flush(args, records, rt)
        print("已保存完成阶段；中断或API预算耗尽。原命令加 --resume 继续；预算上限为 MAX_API_CALLS。", flush=True)
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 2)
    flush(args, records, rt)
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    if any(r.get("status") == "error" for r in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
