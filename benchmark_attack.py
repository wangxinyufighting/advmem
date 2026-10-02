"""批量评测 全种子池 → attacker → gate；目标只在事后 evidence 指标中使用。"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from agents import Attacker, gate
from common import digest, read_json
from metrics import Gold
from packs import Pack, Sampler, check_pack, choose_type

from _benchmark import (BudgetExhausted, Runtime, attempt, bool_stats, checkpoint, common_args, error_text,
                        init_run, jsonl, load_full, mean, print_selection, read_lines, save,
                        selected_data, selection, table)

STAGES = ("raw", "accepted_original", "accepted")
COVERAGE = ("any_question_all_rounds", "question_E_union_all_rounds",
            "any_question_all_sessions", "question_E_union_all_sessions")


def evidence(item, full):
    """raw允许gate尚未通过，但不允许伪造/越界/空E来获得覆盖。"""
    if not isinstance(item, dict) or not isinstance(item.get("q"), str) or not item["q"].strip():
        return None
    ids = item.get("E")
    if (not isinstance(ids, list) or not ids or any(not isinstance(r, str) for r in ids)
            or len(set(ids)) != len(ids) or not set(ids) <= full.rounds.keys()):
        return None
    return set(ids)


def inspect_pool(packs, full):
    ids, seeds = set(), set()
    session_seeds = set()
    for p in packs:
        check_pack(p, full)
        if p.pack_id in ids or p.seed_id in seeds:
            raise ValueError("全种子池含重复pack/seed；请使用 benchmark_packs 输出的一次完整遍历")
        ids.add(p.pack_id)
        seeds.add(p.seed_id)
        if p.kind == "session":
            if p.seed_id not in full.sessions:
                raise ValueError("未知session种子")
            if not set(full.sessions[p.seed_id].rids) <= set(p.rids):
                raise ValueError("session种子原文未完整进入pack")
            session_seeds.add(p.seed_id)
    if session_seeds != full.sessions.keys():
        raise ValueError("输入不是全种子池：未包含每个session种子。请复用 benchmark_packs 的 cases/.../packs.jsonl")


def run_pack(args, p, full, rt, index, qtype, path, meta, pack_index):
    state = checkpoint(path, meta, args, rt.run_id)
    if "input_hash" in state and state["input_hash"] != digest(p.to_dict()):
        raise ValueError("pack checkpoint 与当前原文pack不同")
    state.update(input_hash=digest(p.to_dict()), pack_id=p.pack_id, pack_index=pack_index,
                 seed_id=p.seed_id, qtype=qtype)
    if state.get("status") == "ok":
        return state
    try:
        if args.pack_chars and len(full.render(p.rids, {})) > args.pack_chars:
            raise ValueError("pack超过当前字符预算；拒绝静默截断，请在新目录用适当预算运行")
        if state.get("generation", {}).get("status") != "ok":
            n = attempt(state, "generate", path)
            model = rt.client("ATTACKER", f"{full.fingerprint}:{p.pack_id}:generate:{n}")
            # 没有 q/a/gold/type 标签，只有指定题型、日期与原文pack。
            items = Attacker(model, full, {}).generate(p, qtype, meta["question_date"],
                                                      args.questions_per_pack, nonce=f"{args.seed}:{pack_index}")
            state["generation"] = {"status": "ok", "items": items}
            state["results"] = [None] * len(items)
            save(path, state)
        items = state["generation"]["items"]
        for j, original in enumerate(items):
            existing = state["results"][j]
            if existing is not None and existing.get("status") in {"accepted", "rejected"}:
                continue
            n = attempt(state, f"gate:{j}", path)
            scope = f"{full.fingerprint}:{p.pack_id}:gate:{j}:{n}"
            try:
                result = gate(original, p, full, index, rt.client("JUDGE", scope),
                              rt.client("DEFENDER", scope), meta["question_date"], qtype,
                              args.gate, args.gate_chars)
                if result.get("status") not in {"accepted", "rejected", "error"}:
                    raise ValueError("gate返回未知状态")
            except BudgetExhausted:
                raise
            except Exception as exc:
                result = {"generated": original, "item": original, "status": "error", "reason": error_text(exc)}
            state["results"][j] = {"pack_index": pack_index, "pack_id": p.pack_id,
                                  "question_index": j, **result}
            save(path, state)
        state["status"] = "error" if any(r["status"] == "error" for r in state["results"]) else "ok"
        state.pop("error", None)
    except BudgetExhausted:
        state["status"] = "pending"
        save(path, state)
        raise
    except Exception as exc:
        state.update(status="error", error=error_text(exc))
        if state.get("generation", {}).get("status") != "ok":
            state["generation"] = {"status": "error", "error": error_text(exc)}
    save(path, state)
    return state


def prefix_metrics(packs, states, full, gold):
    """一次顺序扫描。True优先；未命中且有未完成调用则None，不把失败当成false。"""
    gold_rounds = set() if gold.abstention else gold.rounds
    gold_sessions = set() if gold.abstention else gold.sessions
    buckets = {s: {"union": set(), "any_r": False, "any_s": False,
                   "questions": 0, "invalid_E": 0, "unknown": False} for s in STAGES}
    pack_union, pack_joint = set(), False
    counts = Counter()
    curve = []
    for i, (p, state) in enumerate(zip(packs, states), 1):
        pack_union.update(p.rids)
        pack_joint |= bool(gold_rounds) and gold_rounds <= set(p.rids)
        generated = state.get("generation", {})
        if generated.get("status") != "ok":
            counts["generation_errors" if generated.get("status") == "error" else "pending_packs"] += 1
            for b in buckets.values():
                b["unknown"] = True
        else:
            items = generated["items"]
            counts["generated_questions"] += len(items)
            counts["empty_packs"] += not items
            counts["generated_packs"] += 1
            results = state.get("results", [])
            for j, original in enumerate(items):
                result = results[j] if j < len(results) else None
                status = result.get("status") if result else None
                if status in {"accepted", "rejected"}:
                    counts[status] += 1
                else:
                    counts["gate_errors" if status == "error" else "pending_questions"] += 1
                    buckets["accepted_original"]["unknown"] = True
                    buckets["accepted"]["unknown"] = True
                for name, candidate in [("raw", original),
                                        ("accepted_original", original if status == "accepted" else None),
                                        ("accepted", result.get("item") if status == "accepted" else None)]:
                    if name != "raw" and status != "accepted":
                        continue
                    b = buckets[name]
                    b["questions"] += 1
                    ids = evidence(candidate, full)
                    if ids is None:
                        b["invalid_E"] += 1
                        continue
                    sessions = {full.rounds[r].session_id for r in ids}
                    b["union"].update(ids)
                    b["any_r"] |= bool(gold_rounds) and gold_rounds <= ids
                    b["any_s"] |= bool(gold_sessions) and gold_sessions <= sessions
        stages = {}
        for name, b in buckets.items():
            sessions = {full.rounds[r].session_id for r in b["union"]}
            def decision(hit, gold_set):
                return None if not gold_set else (True if hit else (None if b["unknown"] else False))
            stages[name] = {
                "questions": b["questions"], "invalid_evidence_questions": b["invalid_E"],
                "has_unresolved_calls": b["unknown"],
                "any_question_all_rounds": decision(b["any_r"], gold_rounds),
                "question_E_union_all_rounds": decision(gold_rounds <= b["union"], gold_rounds),
                "any_question_all_sessions": decision(b["any_s"], gold_sessions),
                "question_E_union_all_sessions": decision(gold_sessions <= sessions, gold_sessions),
                "union_round_recall_observed": len(gold_rounds & b["union"]) / len(gold_rounds) if gold_rounds else None,
            }
        curve.append({"N": i, "pack_id": p.pack_id,
                      "input_pack_joint": pack_joint if gold_rounds else None,
                      "input_pack_union": gold_rounds <= pack_union if gold_rounds else None,
                      "counts": dict(counts), "stages": stages})
    first = {s: {key: next((r["N"] for r in curve if r["stages"][s][key] is True), None)
                 for key in COVERAGE} for s in STAGES}
    return {"curve": curve, "min_observed_prefix_N": first}


def flatten(states):
    rows = []
    for s in states:
        if s.get("generation", {}).get("status") == "error":
            rows.append({"pack_index": s["pack_index"], "pack_id": s["pack_id"],
                         "status": "generation_error", "error": s["generation"]["error"]})
        else:
            for j, item in enumerate(s.get("generation", {}).get("items", [])):
                result = s.get("results", [])[j] if j < len(s.get("results", [])) else None
                rows.append(result or {"pack_index": s["pack_index"], "pack_id": s["pack_id"],
                                       "question_index": j, "generated": item, "status": "pending"})
    return rows


def export_case(path, report, packs, states, full, gold):
    result = prefix_metrics(packs, states, full, gold)
    report.update(result)
    report["counts"] = result["curve"][-1]["counts"] if result["curve"] else {}
    c = report["counts"]
    report["status"] = "error" if c.get("generation_errors") or c.get("gate_errors") else (
        "pending" if c.get("pending_packs") or c.get("pending_questions") else "ok")
    save(path, report)
    jsonl(path.parent / "questions.jsonl", flatten(states))
    ledger = {p.seed_id: {"pack_id": p.pack_id, "type": s.get("qtype"),
                         "status": "unaudited" if s.get("generation", {}).get("status") is None else (
                             "error" if s.get("generation", {}).get("status") == "error" else (
                                 "asked" if s["generation"]["items"] else "audited_empty"))}
              for p, s in zip(packs, states)}
    save(path.parent / "ledger.json", ledger)
    return report


def run_case(args, meta, case, rt, path):
    report = checkpoint(path, meta, args, rt.run_id)
    if report.get("status") == "ok":
        return report
    try:
        full = load_full(args, meta, case)
        index = None
        if args.packs_dir:
            folder = Path(args.packs_dir) / "cases" / f"{meta['case_index']:04d}"
            pool = [Pack(**r) for r in read_lines(folder / "packs.jsonl")]
        else:
            index = rt.index(full)
            sampler = Sampler(full, index, seed=args.seed, neighbors=args.neighbors,
                              max_chars=args.pack_chars, cluster_threshold=args.cluster_threshold)
            pool = sampler.sample(len(sampler.seeds))
        inspect_pool(pool, full)
        packs = pool[:args.max_packs] if args.max_packs else pool
        if not packs:
            raise ValueError("种子池为空")
        report.update(pool_size=len(pool), packs_planned=len(packs), full_pool_used=len(pool) == len(packs),
                      evidence_labels_available=any(m.get("has_answer") is True
                                                    for s in case["haystack_sessions"] for m in s))
        full.save(path.parent / "full_memory.json")
        jsonl(path.parent / "input_packs.jsonl", [p.to_dict() for p in packs])
        weights = read_json(args.weights)["weights"] if args.weights else None
        rng = random.Random(args.seed)
        types = [choose_type(p, full, rng, weights) for p in packs]
        # 所有pack/题型先固定，再构造评测侧Gold；此变量不传入run_pack。
        gold = Gold.from_case(case, full)
        states, paths = [], []
        pack_meta = {**meta, "question_date": case["question_date"]}
        for i, (p, qtype) in enumerate(zip(packs, types), 1):
            dest = path.parent / "pack_states" / f"{i:05d}.json"
            state = checkpoint(dest, pack_meta, args, rt.run_id)
            state.update(pack_index=i, pack_id=p.pack_id, qtype=qtype)
            states.append(state)
            paths.append(dest)
        if args.gate == "full" and any(s.get("status") != "ok" for s in states):
            index = index or rt.index(full)
        for i, p in enumerate(packs):
            print(f"  pack {i + 1}/{len(packs)} {p.seed_id} type={types[i]}", flush=True)
            try:
                states[i] = run_pack(args, p, full, rt, index, types[i], paths[i], pack_meta, i + 1)
            except (KeyboardInterrupt, BudgetExhausted):
                # pack内每个阶段已写盘；更新当前case导出后再向上传播中断。
                if paths[i].exists():
                    states[i] = read_json(paths[i])
                export_case(path, report, packs, states, full, gold)
                raise
            export_case(path, report, packs, states, full, gold)
        report.pop("error", None)
    except BudgetExhausted:
        raise
    except Exception as exc:
        report.update(status="error", error=error_text(exc))
    save(path, report)
    return report


def point(record, n):
    curve = record.get("curve", [])
    if not curve:
        return None
    return curve[-1] if n == "all" else curve[min(n, len(curve)) - 1]


def summarize(records, grid):
    groups = defaultdict(list)
    for r in records:
        groups[r["question_type"]].append(r)
        groups["ALL_MICRO"].append(r)
    out = []
    for name, rows in sorted(groups.items()):
        for n in [*grid, "all"]:
            points = [point(r, n) for r in rows]
            counts = Counter()
            for p in points:
                if p:
                    counts.update(p["counts"])
            known_gates = counts["accepted"] + counts["rejected"]
            for stage in STAGES:
                base = {"group": name, "N": n, "stage": stage, "selected": len(rows),
                        "case_errors": sum(r.get("status") == "error" for r in rows),
                        "case_pending": sum(r.get("status") == "pending" for r in rows),
                        "evidence_labeled_cases": sum(r.get("evidence_labels_available") is True for r in rows),
                        "no_evidence_labels": sum(r.get("evidence_labels_available") is False for r in rows),
                        "full_pool_cases_used": sum(r.get("full_pool_used") is True for r in rows),
                        "effective_N_mean": mean(p["N"] for p in points if p),
                        "input_joint_rate_known": mean(p["input_pack_joint"] for p in points if p),
                        "generated_questions": counts["generated_questions"], "accepted": counts["accepted"],
                        "rejected": counts["rejected"], "generation_errors": counts["generation_errors"],
                        "gate_errors": counts["gate_errors"], "empty_packs": counts["empty_packs"],
                        "gate_acceptance_known": counts["accepted"] / known_gates if known_gates else None,
                        "union_round_recall_observed_mean": mean(p["stages"][stage]["union_round_recall_observed"]
                                                                 for p in points if p)}
                for metric in COVERAGE:
                    vals = [p["stages"][stage][metric] if p else None for p in points]
                    stats = bool_stats(vals)
                    prefix = metric + "_"
                    base.update({prefix + k: v for k, v in stats.items() if k != "selected"})
                # 条件转化率只看输入pack已可达的case，并公开未知判定数量。
                reachable = [p for p in points if p and p["input_pack_joint"] is True]
                conv = bool_stats([p["stages"][stage]["question_E_union_all_rounds"] for p in reachable])
                base.update(input_reachable_cases=len(reachable),
                            conversion_union_known=conv["rate_completed"], conversion_union_unknown=conv["unknown"])
                out.append(base)
    return out


def flush(args, records, rt):
    out = Path(args.out)
    groups = summarize(records, args.n_grid)
    save(out / "summary.json", {"groups": groups, "current_process_api": rt.stats(),
                               "semantic_target_matching": "not_run", "defect_filter": "not_run"})
    table(out / "summary.csv", groups)
    jsonl(out / "cases.jsonl", records)


def main(argv=None):
    p = common_args(__doc__)
    p.add_argument("--questions-per-pack", type=int, default=4)
    p.add_argument("--gate", choices=["basic", "full"], default="full")
    p.add_argument("--gate-chars", type=int, default=120000)
    p.add_argument("--pack-chars", type=int, default=48000)
    p.add_argument("--neighbors", type=int, default=8, help="仅独立构造pack时生效")
    p.add_argument("--cluster-threshold", type=float, help="仅独立构造pack时生效")
    p.add_argument("--weights", help="训练集全局题型权重JSON；不传为均匀分布")
    p.add_argument("--n-grid", nargs="+", type=int, default=[5, 10, 20, 40])
    p.add_argument("--max-packs", type=int, default=0, help="0=全种子池；正数仅用于前缀小规模排错")
    args = p.parse_args(argv)
    if min(args.max_packs, args.neighbors, args.pack_chars, args.gate_chars) < 0 or args.questions_per_pack < 1:
        p.error("题数必须为正；预算不能为负")
    if any(n < 1 for n in args.n_grid):
        p.error("n-grid必须为正")
    args.n_grid = sorted(set(args.n_grid))
    rows = selection(args)
    if any(r["question_type"] == "abstention" for r in rows):
        p.error("attacker正证据覆盖实验不支持abstention，请选择六种可答类型；reader入口支持abstention")
    print_selection(rows)
    if args.list:
        return
    run_id = init_run(args, rows)
    rt = Runtime(args, run_id)
    records = [{**r, "status": "pending"} for r in rows]
    try:
        for i, (meta, case) in enumerate(selected_data(args, rows)):
            path = Path(args.out) / "cases" / f"{meta['case_index']:04d}" / "case_report.json"
            print(f"[{i + 1}/{len(rows)}] attacker {meta['question_id']} {meta['question_type']}", flush=True)
            try:
                records[i] = run_case(args, meta, case, rt, path)
            except (KeyboardInterrupt, BudgetExhausted):
                if path.exists():
                    records[i] = read_json(path)
                raise
            flush(args, records, rt)
            print(f"  status={records[i]['status']} counts={records[i].get('counts')} "
                  f"error={records[i].get('error', '')}", flush=True)
    except (KeyboardInterrupt, BudgetExhausted) as exc:
        flush(args, records, rt)
        print("已保存到pack/候选问题粒度；原命令加 --resume 继续，预算上限为 MAX_API_CALLS。", flush=True)
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 2)
    flush(args, records, rt)
    # 终端只显示最重要的全池指标，完整长表在summary.csv。
    compact = [{k: r[k] for k in ["group", "stage", "selected", "accepted", "generation_errors", "gate_errors",
                                  "question_E_union_all_rounds_rate_completed",
                                  "question_E_union_all_rounds_unknown"]}
               for r in summarize(records, args.n_grid) if r["N"] == "all"]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if any(r.get("status") == "error" for r in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
