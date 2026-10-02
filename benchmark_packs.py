"""按题型测 pack 可达性。放在 full_memory_lab/run.py 同目录；不调用生成模型。

python benchmark_packs.py --data "$DATA" --list
python benchmark_packs.py --data "$DATA" --per-type 10 --embedding local --out runs/by_type

核心指标：全种子池里是否存在一个 pack 包含全部标注证据 round。
它是当前固定 pack 构造器的标注证据共现上限，不是最终 M 正确率的上限。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common import digest, iter_cases, read_json, write_json, write_jsonl
from memory import FullMemory
from metrics import Gold, evidence_report
from packs import TYPES, Sampler
from retrieve import CrossReranker, Embedder, Retriever


def group_type(case: dict) -> str:
    # 不可答题保留原 question_type，因此必须先检查 ID，避免重复计入六类。
    return "abstention" if case["question_id"].endswith("_abs") else case["question_type"]


def hit_probability(pool_size: int, good: int, n: int) -> float:
    """均匀无放回抽取 N 个种子，至少遇到一个证据齐全 pack 的精确概率。"""
    if not 0 <= good <= pool_size or n < 0:
        raise ValueError("非法 pool_size/good/N")
    n = min(n, pool_size)  # 固定池扫完后重复，不会增加证据可达性。
    if good == 0 or n == 0:
        return 0.0
    if n > pool_size - good:
        return 1.0
    return 1.0 - math.comb(pool_size - good, n) / math.comb(pool_size, n)


def pool_report(pool, full, gold, budgets: list[int]) -> dict:
    report = evidence_report(pool, full, gold)
    rows = report["curve"]
    if gold.abstention or not gold.rounds:
        return {"status": "no_evidence_labels", "evidence_report": report}
    size = len(pool)
    good = sum(row["this_pack_all_rounds"] is True for row in rows)
    curve = []
    for n in budgets:
        used = min(n, size)
        last = rows[used - 1] if used else {}
        curve.append({
            "N": n, "effective_N": used,
            "expected_joint_coverage": hit_probability(size, good, n),
            "observed_joint_coverage": bool(last.get("any_pack_all_rounds", False)),
            "observed_union_coverage": bool(last.get("union_all_rounds", False)),
        })
    n95 = next((n for n in range(1, size + 1)
                if hit_probability(size, good, n) >= 0.95), None)
    evidence_sessions = {full.rounds[r].session_id for r in gold.rounds}
    return {
        "status": "ok", "pool_size": size, "good_packs": good,
        "evidence_rounds": len(gold.rounds), "evidence_sessions": len(evidence_sessions),
        "annotation_session_count": len(gold.sessions),
        "full_pool_joint": good > 0,
        "full_pool_union": rows[-1]["union_all_rounds"] if rows else False,
        "joint_density": good / size if size else 0.0,
        "max_single_pack_round_recall": max((r["round_recall"] for r in rows), default=0.0),
        "first_observed_N": report["min_observed_prefix_N"]["any_pack_all_rounds"],
        "N95": n95,
        "expected_first_N_if_reachable": (size + 1) / (good + 1) if good else None,
        "packs_with_dropped_neighbors": sum(bool(p.dropped_rids) for p in pool),
        "max_pack_chars": max((len(full.render(p.rids, {})) for p in pool), default=0),
        "curve": curve, "evidence_report": report,
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def aggregate(records: list[dict], budgets: list[int]) -> list[dict]:
    """每个 case 等权。失败/缺标签单独计数，不伪装成已完成的覆盖失败。"""
    groups = defaultdict(list)
    for r in records:
        groups[r["question_type"]].append(r)
        groups["ALL_MICRO"].append(r)
        if r.get("status") == "ok":
            n = r["evidence_sessions"]
            bucket = str(n) if n <= 2 else ("3-4" if n <= 4 else "5+")
            groups["evidence_sessions=" + bucket].append(r)
    out = []
    for name, rows in sorted(groups.items()):
        ok = [r for r in rows if r.get("status") == "ok"]
        count, completed = len(rows), len(ok)
        covered = sum(r["full_pool_joint"] for r in ok)
        n95 = [r["N95"] for r in ok if r["N95"] is not None]
        row = {
            "group": name, "selected": count, "completed": completed,
            "errors": sum(r.get("status") == "error" for r in rows),
            "no_evidence_labels": sum(r.get("status") == "no_evidence_labels" for r in rows),
            "full_pool_joint_rate": covered / completed if completed else None,
            # 未完成样本不被悄悄删除：另报全体所选样本的保守界。
            "joint_rate_lower_all_selected": covered / count,
            "joint_rate_upper_if_unknown_succeed": (covered + count - completed) / count,
            "full_pool_union_rate": mean(r["full_pool_union"] for r in ok),
            "mean_joint_density": mean(r["joint_density"] for r in ok),
            "mean_best_single_pack_round_recall": mean(r["max_single_pack_round_recall"] for r in ok),
            "N95_median_conditional_on_reachable": statistics.median(n95) if n95 else None,
            "mean_pool_size": mean(r["pool_size"] for r in ok),
        }
        for n in budgets:
            points = [next(c for c in r["curve"] if c["N"] == n) for r in ok]
            row[f"joint_probability_at_{n}"] = mean(c["expected_joint_coverage"] for c in points)
            row[f"observed_joint_at_{n}"] = mean(c["observed_joint_coverage"] for c in points)
            row[f"observed_union_at_{n}"] = mean(c["observed_union_coverage"] for c in points)
        out.append(row)
    # 六类宏平均只计算本次确实选中且有结果的类型，并公开分母。
    typed = [r for r in out if r["group"] in TYPES and r["completed"]]
    if typed:
        keys = ["full_pool_joint_rate", "full_pool_union_rate"] + [f"joint_probability_at_{n}" for n in budgets]
        out.append({"group": "ANSWERABLE_MACRO", "included_types": len(typed),
                    **{key: mean(r[key] for r in typed) for key in keys}})
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--list", action="store_true", help="只统计本地文件并展示每类 case 索引，不加载模型")
    p.add_argument("--ids", help="可选：只使用这些 train/dev question_id 的 JSON 字符串数组")
    p.add_argument("--types", nargs="+", choices=TYPES, default=TYPES)
    p.add_argument("--per-type", type=int, default=10, help="每类随机选几个；0=全部；不可答题不参与pack召回")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embedding", choices=["none", "local", "api"], default="local")
    p.add_argument("--embed-model")
    p.add_argument("--reranker-model")
    p.add_argument("--neighbors", type=int, default=8)
    p.add_argument("--pack-chars", type=int, default=48000)
    p.add_argument("--cluster-threshold", type=float)
    p.add_argument("--n-grid", nargs="+", type=int, default=[5, 10, 20, 40])
    p.add_argument("--cache", default=".cache")
    p.add_argument("--out", default="runs/pack_benchmark")
    p.add_argument("--resume", action="store_true", help="同配置跳过已完成 case；失败 case 重试")
    args = p.parse_args()
    if args.per_type < 0 or args.neighbors < 0 or args.pack_chars < 0 or any(n < 1 for n in args.n_grid):
        p.error("per-type/neighbors/pack-chars 必须非负，n-grid 必须为正")
    args.n_grid = sorted(set(args.n_grid))
    return args


def main():
    args = parse_args()
    allowed = None
    if args.ids:
        values = read_json(args.ids)
        if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
            raise ValueError("--ids 文件必须是 question_id 字符串数组")
        allowed = set(values)
    catalog, counts, found = [], Counter(), set()
    # 第一遍只保留小体积的 case 元数据，不把整个 S 文件装入内存。
    for index, case in enumerate(iter_cases(args.data)):
        qid = case["question_id"]
        if qid in found:
            raise ValueError(f"重复 question_id: {qid}")
        found.add(qid)
        if allowed is not None and qid not in allowed:
            continue
        kind = group_type(case)
        counts[kind] += 1
        catalog.append({"case_index": index, "question_id": qid, "question_type": kind,
                        "stored_question_type": case["question_type"],
                        "sessions": len(case["haystack_sessions"])})
    if allowed is not None and allowed - found:
        raise ValueError(f"--ids 中有数据里不存在的 ID: {sorted(allowed - found)[:10]}")
    print(json.dumps({"counts": dict(sorted(counts.items())),
                      "examples": {t: [r for r in catalog if r["question_type"] == t][:3]
                                   for t in sorted(counts)}}, ensure_ascii=False, indent=2), flush=True)
    if args.list:
        return
    if allowed is None:
        print("注意：未限定 train/dev IDs；该运行可作探索，但调参后不能再把同一批 case 称为未见测试集。", flush=True)
    selected = []
    for kind in args.types:
        # hash 随机排序：不依赖原文件按题型排列，换检索配置也保持 case 子集不变。
        rows = sorted((r for r in catalog if r["question_type"] == kind),
                      key=lambda r: digest([args.seed, "case-selection", r["question_id"]]))
        selected.extend(rows[:args.per_type] if args.per_type else rows)
    selected.sort(key=lambda r: r["case_index"])
    if not selected:
        raise ValueError("没有选中可答 case")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.data).resolve()
    config = {k: v for k, v in vars(args).items() if k not in {"resume", "list", "out"}}
    config.update(data=str(source), data_size=source.stat().st_size,
                  data_mtime_ns=source.stat().st_mtime_ns, script_version="1.0",
                  selected_ids=[r["question_id"] for r in selected])
    old = out / "config.json"
    if old.exists() and read_json(old) != config:
        raise ValueError("输出目录已有不同实验配置，请更换 --out，避免结果混用")
    write_json(old, config)
    write_json(out / "inventory.json", catalog)
    write_json(out / "selected_cases.json", selected)
    # 所有 case 共用同一模型实例；向量索引按原项目内容哈希复用。
    embedder = None if args.embedding == "none" else Embedder(args.embedding, args.embed_model, Path(args.cache) / "api")
    reranker = CrossReranker(args.reranker_model) if args.reranker_model else None
    wanted, records = {r["question_id"]: r for r in selected}, []
    for index, case in enumerate(iter_cases(args.data)):
        qid = case["question_id"]
        if qid not in wanted:
            continue
        folder = out / "cases" / f"{index:04d}"
        record_path = folder / "case_report.json"
        if args.resume and record_path.exists() and read_json(record_path).get("status") == "ok":
            record = read_json(record_path)
        else:
            record = {**wanted[qid], "case_dir": str(folder), "case_hash": digest(case)}
            try:
                full = FullMemory.build(case)
                retriever = Retriever(full.documents(), embedder, Path(args.cache) / "index", reranker)
                sampler = Sampler(full, retriever, seed=args.seed, neighbors=args.neighbors,
                                  max_chars=args.pack_chars, cluster_threshold=args.cluster_threshold)
                pool = sampler.sample(len(sampler.seeds))  # 每个固定种子恰好一次，不调用 attacker。
                full.save(folder / "full_memory.json")
                write_jsonl(folder / "packs.jsonl", [p.to_dict() for p in pool])
                write_json(folder / "ledger.json", sampler.ledger)
                # 全部 pack 固定后才构造 Gold，标签从不参与召回/排序/截取。
                gold = Gold.from_case(case, full)
                record.update(pool_report(pool, full, gold, args.n_grid))
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            write_json(record_path, record)
        records.append(record)
        write_jsonl(out / "cases.jsonl", records)  # 中途失败也保留已完成结果。
        print(f"[{len(records)}/{len(selected)}] {qid} {record['question_type']} "
              f"status={record['status']} full_pool_joint={record.get('full_pool_joint')} "
              f"K/S={record.get('good_packs')}/{record.get('pool_size')} "
              f"N95={record.get('N95')} {record.get('error', '')}", flush=True)
    summary = aggregate(records, args.n_grid)
    write_json(out / "summary.json", {"groups": summary, "abstention_excluded": counts.get("abstention", 0),
                                     "config": config})
    columns = list(dict.fromkeys(k for row in summary for k in row))
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(r.get("status") == "error" for r in records):
        raise SystemExit("部分 case 失败；查看 cases.jsonl。不要只读成功样本均值。")


if __name__ == "__main__":
    main()
