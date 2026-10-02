"""三个实验的命令行入口。python run.py --help 查看命令。"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from agents import Attacker, answer, gate, grade
from common import digest, iter_cases, load_case, normalize_answer, read_json, read_jsonl, write_json, write_jsonl
from llm import Client, ModelError
from memory import FullMemory, history_only, memory_documents, provenance_marks
from metrics import Gold, evidence_report, match_target, probe_target, question_report, repeated_sampling
from packs import Pack, Sampler, check_pack, choose_type, expand_pack
from retrieve import CrossReranker, Embedder, Retriever


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LongMemEval full memory / packs / attacker 实验")
    subs = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", required=True, help="原版或 cleaned S JSON，必须显式选择")
    common.add_argument("--case-index", type=int, default=0)
    common.add_argument("--question-id", help="提供后优先按 question_id 选 case")
    common.add_argument("--out", default="runs/case0")
    common.add_argument("--full", help="复用 full_memory.json，并校验与当前 case 完全一致")
    common.add_argument("--embedding", choices=["none", "local", "api"], default="local")
    common.add_argument("--embed-model")
    common.add_argument("--reranker-model", help="可选 cross-encoder 模型名或本地路径")
    common.add_argument("--cache", default=".cache")
    common.add_argument("--memory", help="可选 M 的 JSON数组或JSONL；不是 full memory")
    subs.add_parser("build", parents=[common], help="构建无损原文库及检索索引")
    a = subs.add_parser("answer", parents=[common], help="实验1：检索后回答官方问题")
    a.add_argument("--k", type=int, default=10)
    a.add_argument("--steps", type=int, default=0, help="LLM 补检索轮数；0=fast")
    a.add_argument("--expand", type=int, default=1, help="结构邻接候选扩展步数")
    a.add_argument("--context-chars", type=int, default=80000)
    a.add_argument("--retrieval-only", action="store_true", help="不调用 reader/judge")
    pack = argparse.ArgumentParser(add_help=False)
    pack.add_argument("--n", type=int, help="pack 数量，不是问题数；默认8或读取的文件长度")
    pack.add_argument("--seed", type=int, default=0)
    pack.add_argument("--neighbors", type=int, default=8)
    pack.add_argument("--pack-chars", type=int, default=48000, help="可见原文字数预算；0不限制")
    pack.add_argument("--cluster-threshold", type=float, help="启用额外主题簇种子，如0.75")
    b = subs.add_parser("packs", parents=[common, pack], help="实验2：不看目标题采样，事后测覆盖")
    b.add_argument("--trials", type=int, default=1, help="多次随机重排种子池；不消耗LLM")
    b.add_argument("--probe-target", action="store_true", help="额外让judge检查每个pack能否支持目标题")
    c = subs.add_parser("attack", parents=[common, pack], help="实验3：固定pack出题，再事后测覆盖")
    c.add_argument("--packs", help="复用实验2的packs.jsonl；--n可限制前缀")
    c.add_argument("--questions-per-pack", type=int, default=4)
    c.add_argument("--active-search", action="store_true", help="attacker正式出题前自主检索一次")
    c.add_argument("--weights", help="weights命令输出的训练集全局题型分布；默认均匀")
    c.add_argument("--gate", choices=["off", "basic", "full"], default="full")
    c.add_argument("--gate-chars", type=int, default=120000)
    c.add_argument("--skip-target-match", action="store_true", help="只生成，不做问题语义覆盖判分")
    w = subs.add_parser("weights", help="仅从显式指定的训练IDs统计题型权重")
    w.add_argument("--data", required=True)
    w.add_argument("--train-ids", required=True, help="JSON字符串数组，必须是你自己的训练划分")
    w.add_argument("--out", required=True)
    return p


def prepare(args):
    case = load_case(args.data, args.case_index, args.question_id)
    full = FullMemory.load(args.full) if args.full else FullMemory.build(case)
    if full.export_history() != history_only(case):
        raise ValueError("保存的 full memory 与当前 case/数据版本不同")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full.save(out / "full_memory.json")
    write_json(out / f"run.{args.command}.json", {
        "args": vars(args), "case_id": case["question_id"], "case_hash": digest(case),
        "full_hash": full.fingerprint, "code_version": "1.0"})
    embedder = None if args.embedding == "none" else Embedder(
        args.embedding, args.embed_model, Path(args.cache) / "api")
    reranker = CrossReranker(args.reranker_model) if args.reranker_model else None
    retriever = Retriever(full.documents(), embedder, Path(args.cache) / "index", reranker)
    entries = []
    if args.memory:
        entries = read_jsonl(args.memory) if args.memory.endswith(".jsonl") else read_json(args.memory)
        if not isinstance(entries, list):
            raise ValueError("M 文件顶层必须是数组，或每行一个条目的 JSONL")
        memory_documents(entries, full)  # 提前校验 provenance。
    write_json(out / "index_manifest.json", {
        "full_hash": full.fingerprint, "backend": args.embedding,
        "embedding_signature": embedder.signature if embedder else None,
        "reranker": args.reranker_model, "documents": len(retriever.docs),
        "embedding_chunks": len(retriever.owners) if retriever.owners is not None else 0})
    # Gold 只存在实验编排/评测侧，下面从不把它传给 sampler 或 Attacker。
    return case, full, retriever, entries, out, Gold.from_case(case, full)


def make_sampler(args, full, retriever, entries) -> Sampler:
    return Sampler(full, retriever, seed=args.seed, neighbors=args.neighbors,
                   max_chars=args.pack_chars, marks=provenance_marks(entries),
                   cluster_threshold=args.cluster_threshold)


def model(args, role: str) -> Client:
    return Client.from_env(role, Path(args.cache) / "api")


def experiment_answer(args, full, retriever, entries, out, gold):
    # 同一路径重跑失败时，不留下上次成功的预测，避免误送官方评测。
    for filename in ("predictions.jsonl", "retrieved_context.txt"):
        (out / filename).unlink(missing_ok=True)
    if args.k < 1 or args.steps < 0 or args.context_chars < 0:
        raise ValueError("k必须为正，steps/字符预算不能为负")
    if args.memory:
        docs = memory_documents(entries, full)
        retriever = Retriever(docs, retriever.embedder, Path(args.cache) / "index", retriever.reranker) if docs else None
    # try:
    hits, trace = retriever.retrieve(
        gold.question, gold.date, args.k,
        planner=model(args, "PLANNER") if args.steps else None, steps=args.steps,
        expand=args.expand, max_chars=args.context_chars) if retriever else ([], [])
    context, visible = retriever.context(hits, args.context_chars) if retriever else ("", [])
    rids = sorted({r for h in visible for r in retriever.docs[h.id].prov})
    pseudo = Pack("retrieval", full.fingerprint, "retrieval", "retrieval", 0, [], rids)
    report = {"status": "retrieval_only" if args.retrieval_only else "ok",
                "retrieved_hits": [asdict(h) for h in hits], "visible_ids": [h.id for h in visible],
                "context_chars": len(context), "retrieval_trace": trace,
                "coverage_kind": "provenance_only" if args.memory else "visible_raw_rounds",
                "evidence": evidence_report([pseudo], full, gold)}
    (out / "retrieved_context.txt").write_text(context, encoding="utf-8")
    if not args.retrieval_only:
        prediction = answer(model(args, "DEFENDER"), gold.question, gold.date, context)
        write_jsonl(out / "predictions.jsonl", [{"question_id": gold.qid, "hypothesis": prediction}])
        report.update(hypothesis=prediction,
                        normalized_exact_match=normalize_answer(prediction) == normalize_answer(gold.answer))
        try:
            report["llm_judge"] = grade(model(args, "JUDGE"), gold.question, gold.answer,
                                        prediction, gold.qtype, gold.abstention)
        except ModelError as exc:
            report["llm_judge"] = {"correct": None, "error": str(exc)}
    # except ModelError as exc:
    #     report = {"status": "error", "error": str(exc)}
    write_json(out / "answer_report.json", report)
    return report


def experiment_packs(args, full, retriever, entries, out, gold):
    n = args.n if args.n is not None else 8
    if args.trials < 1:
        raise ValueError("trials必须为正")
    sampler = make_sampler(args, full, retriever, entries)
    pool = sampler.sample(max(n, len(sampler.seeds)) if args.trials > 1 else n)
    packs = pool[:n]
    write_jsonl(out / "packs.jsonl", [p.to_dict() for p in packs])
    write_json(out / "ledger.json", sampler.ledger)
    report = evidence_report(packs, full, gold)
    if args.trials > 1:
        pool = pool[:len(sampler.seeds)]
        write_jsonl(out / "pack_pool.jsonl", [p.to_dict() for p in pool])
        report["repeated_sampling"] = repeated_sampling(pool, full, gold, n, args.trials, args.seed)
    if args.probe_target:
        judge, probes = model(args, "JUDGE"), []
        # pack已经全部固定；probe结果绝不影响采样顺序或检索。
        for p in packs:
            try:
                probes.append({"pack_id": p.pack_id, **probe_target(judge, p, full, gold)})
            except ModelError as exc:
                probes.append({"pack_id": p.pack_id, "supported": None, "error": str(exc)})
        report["target_support_probes"] = probes
        report["target_supported_min_observed_prefix_N"] = next(
            (i for i, p in enumerate(probes, 1) if p["supported"] is True), None)
    write_json(out / "pack_report.json", report)
    return {"min_observed_prefix_N": report["min_observed_prefix_N"], "N_tested": len(packs)}


def experiment_attack(args, full, retriever, entries, out, gold):
    if args.questions_per_pack < 1:
        raise ValueError("questions-per-pack必须为正")
    if args.packs:
        packs = [Pack(**p) for p in read_jsonl(args.packs)]
        if args.n is not None:
            if args.n < 0 or args.n > len(packs):
                raise ValueError("--n必须位于已保存pack数量范围内")
            packs = packs[:args.n]
    else:
        sampler = make_sampler(args, full, retriever, entries)
        packs = sampler.sample(args.n if args.n is not None else 8)
    for p in packs:
        check_pack(p, full)
    # 默认均匀必须标明，不能装作已经获得真实训练分布。
    weights = read_json(args.weights)["weights"] if args.weights else None
    marks, rng = provenance_marks(entries), random.Random(args.seed)
    attacker_client, defender, judge = model(args, "ATTACKER"), model(args, "DEFENDER"), model(args, "JUDGE")
    attacker = Attacker(attacker_client, full, marks)
    write_jsonl(out / "input_packs.jsonl", [p.to_dict() for p in packs])
    records, expanded = [], []
    ledger = {sid: {"status": "unaudited", "visits": []} for sid in full.sessions}
    for i, original in enumerate(packs, 1):
        p = original
        print(f"pack {i}/{len(packs)}: {p.seed_id}", file=sys.stderr, flush=True)
        visit = {"pack_id": p.pack_id, "sweep": p.sweep, "memory_version": digest(marks)}
        try:
            if args.pack_chars and len(full.render(p.rids, marks)) > args.pack_chars:
                raise ModelError("pack超出当前字符预算，请增大--pack-chars；未偷偷截断")
            if args.active_search:
                p = expand_pack(p, full, retriever, attacker_client, gold.date, marks,
                                args.pack_chars, nonce=f"{args.seed}:{i}:expand")
            qtype = choose_type(p, full, rng, weights)
            # 这里只从Gold取用户事先允许的question_date；其余目标字段完全不传入。
            items = attacker.generate(p, qtype, gold.date, args.questions_per_pack,
                                      nonce=f"{args.seed}:{i}:generate")
            visit.update(status="asked" if items else "audited_empty", generated=len(items), type=qtype)
            for item in items:
                result = gate(item, p, full, retriever, judge, defender, gold.date, qtype,
                              args.gate, args.gate_chars)
                records.append({"pack_index": i, "pack_id": p.pack_id, **result})
            visit["accepted"] = sum(r["status"] == "accepted" for r in records if r["pack_index"] == i)
        except ModelError as exc:
            visit.update(status="error", error=str(exc))
            records.append({"pack_index": i, "pack_id": p.pack_id,
                            "status": "generation_error", "error": str(exc)})
        expanded.append(p)
        entry = ledger.setdefault(p.seed_id, {"status": "unaudited", "visits": []})
        entry["status"] = visit["status"]
        entry["visits"].append(visit)
        write_jsonl(out / "questions.jsonl", records)
        write_jsonl(out / "expanded_packs.jsonl", [p.to_dict() for p in expanded])
        write_json(out / "ledger.json", ledger)
    # 到此所有pack都已出题；之后才允许拿官方目标做语义比较。
    if not args.skip_target_match:
        for row in records:
            if "generated" not in row:
                continue
            for item_key, match_key in [("generated", "raw_target_match"), ("item", "target_match")]:
                if item_key == "item" and row["status"] != "accepted":
                    continue
                if item_key == "item" and row["item"] == row["generated"]:
                    row[match_key] = row["raw_target_match"]
                    continue
                item = row[item_key]
                try:
                    if not isinstance(item, dict) or not isinstance(item.get("q"), str):
                        raise ModelError("格式无效的候选不能判问题等价")
                    row[match_key] = match_target(judge, item, gold)
                except ModelError as exc:
                    row[match_key] = {"equivalent": None, "answer_consistent": None, "error": str(exc)}
    write_jsonl(out / "questions.jsonl", records)
    report = {"N_packs": len(packs), "type_distribution": weights or "uniform_not_training_distribution",
              "counts": dict(Counter(r["status"] for r in records)),
              "input_pack_coverage": evidence_report(packs, full, gold),
              "expanded_pack_coverage": evidence_report(expanded, full, gold),
              "raw_questions": question_report(records, len(packs), full, gold, "raw"),
              "accepted_questions": question_report(records, len(packs), full, gold, "accepted"),
              "models": {role: {"model": c.model, "network_calls": c.calls, "cache_hits": c.cache_hits}
                         for role, c in [("attacker", attacker_client), ("defender", defender), ("judge", judge)]}}
    write_json(out / "attack_report.json", report)
    return {"counts": report["counts"], "raw_min_N": report["raw_questions"]["min_observed_prefix_N"],
            "accepted_min_N": report["accepted_questions"]["min_observed_prefix_N"]}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "weights":
        ids = read_json(args.train_ids)
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or not ids:
            raise ValueError("train-ids必须为非空字符串数组")
        ids, seen, counts = set(ids), set(), Counter()
        for case in iter_cases(args.data):
            if case["question_id"] in ids:
                counts[case["question_type"]] += 1
                seen.add(case["question_id"])
        if seen != ids:
            raise ValueError(f"训练IDs不在数据中：{sorted(ids - seen)}")
        write_json(args.out, {"weights": dict(counts), "training_case_ids": sorted(ids)})
        return
    case, full, retriever, entries, out, gold = prepare(args)
    if args.command == "build":
        result = {"case_id": case["question_id"], "sessions": len(full.sessions),
                  "rounds": len(full.rounds), "lossless": True, "full_hash": full.fingerprint}
    elif args.command == "answer":
        result = experiment_answer(args, full, retriever, entries, out, gold)
    elif args.command == "packs":
        result = experiment_packs(args, full, retriever, entries, out, gold)
    else:
        result = experiment_attack(args, full, retriever, entries, out, gold)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
