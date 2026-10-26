"""checkpoint验证与早停（TRAINING.md“停止准则”一节）。

不使用测试集，也不用训练奖励做停止信号。每个checkpoint在val split上评估：

Builder（python -m admem.validation builder）
  用待评checkpoint（通过--role指向的OpenAI兼容服务）把每个val case从头构建记忆（run_case mode=build），然后：
  - 主指标 bank_accuracy：固定的留出问答集上的正确率。题库由一个冻结的、更强的attacker
    一次性生成（admem.cli bank --split val --attacker-role VAL_ATTACKER），之后所有checkpoint共用；
    每题按 bank:<case>:<question_key> 记录对错，供配对bootstrap比较。
  - 辅助指标 official_accuracy：val case自带的LongMemEval目标题（每个case一题，噪声大，只作参考）。
  - 护栏：legal_rate、faith_rate（需要配置faith_gate=true才会判定）、noop_rate、fallback_rate、mean_m_tokens。
  --reference full 不构建记忆，直接用全部原文作为“记忆”，给出同一套题上的上限参考。

Attacker（python -m admem.validation attacker）
  针对固定的参考记忆（一次参考Builder运行的M_build）在val case上抽样若干pack出题：
  - 主指标 defects_per_pack：通过gate且暴露参考记忆可修复缺陷的题数/已发送pack；
  - 护栏：invalid_rate、accept_rate。

EarlyStopper：以第一个评估（起始模型）为基线与当前最佳；之后每个checkpoint
  - 护栏相对基线退化超过容差 → 该checkpoint不能当选最佳；连续guard_patience次 → 停止；
  - 主指标与当前最佳做逐题（逐pack）配对bootstrap，P(更好) ≥ confidence 且均值提升 > min_delta → 新的最佳；
  - 连续patience次没有提升 → 停止。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from .common import Config, TokenBudget, Unknown, connect, digest, read, rows, write
from .data import TypeScheduler, contexts, resolve_type_policy
from .pipeline import REPAIRABLE_DEFECTS, _bank_pack, make_pool, run_case
from .store import neutral_round, question_key, text_size


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# ================================================================ builder

def full_memory(full):
    return [{"id": rid, "text": f"Recorded on {s['date']}. {s['text']}", "prov": [rid], "kind": "raw"}
            for rid in full.ordered(full.rounds) for s in [neutral_round(full, rid)]]


def build_stats(case_dir):
    """从run_case(mode=build)的checkpoint事件统计护栏所需的计数。"""
    path = Path(case_dir) / "checkpoint.json"
    events = [e for e in read(path).get("events", []) if e.get("stage") == "build"] if path.exists() else []
    stats = Counter(windows=len(events))
    for event in events:
        checks = event.get("checks") or {}
        stats["fallback"] += bool(event.get("fallback"))
        if checks.get("legal") is True:
            stats["legal"] += 1
            if isinstance(checks.get("faith"), dict):
                stats["faith_judged"] += 1
                stats["faithful"] += checks["faith"].get("faithful") is True
            if event.get("ops") == 0:
                stats["noop"] += 1
    return stats


def bank_questions(bank_dir, key, limit=None):
    path = Path(bank_dir) / key / "bank.json"
    if not path.exists():
        return []
    questions = sorted(read(path), key=lambda q: digest([key, question_key(q)]))
    return questions[:limit] if limit else questions


def builder_metrics(env, prepared, split, out, *, role="BUILDER_EVAL", val_bank=None, reference=None,
                    max_bank_questions=None, limit=None, keys=None, question_types=None, per_type=None):
    out = Path(out)
    golds = {q["key"]: q for q in rows(Path(prepared) / "private_eval.jsonl")}
    per_question, stats, tokens, failures, unknown = {}, Counter(), [], [], 0
    for context, full_path in contexts(prepared, split, limit, keys, question_types, per_type):
        key = context["key"]
        full = env.memory_module.FullMemory.load(full_path)
        if reference == "full":
            memory = full_memory(full)
        else:
            case_dir = out / "build" / key
            try:
                if not (case_dir / "M_final.json").exists():
                    run_case(full, full_path, context, env, case_dir, mode="build", bank=[], builder_role=role)
            except Unknown as exc:
                failures.append({"key": key, "error": str(exc)})
                continue
            memory = read(case_dir / "M_final.json")
            stats.update(build_stats(case_dir))
        tokens.append(text_size(memory, env.counter))
        gold = golds[key]
        questions = [("official:" + key, gold, gold.get("abstention", False))]
        if val_bank:
            questions += [(f"bank:{key}:{question_key(q)}", q, False)
                          for q in bank_questions(val_bank, key, max_bank_questions)]
        for qid, q, abstention in questions:
            item = {k: q[k] for k in ("q", "a", "type", "question_date", "E")}
            try:
                per_question[qid] = float(env.answer(memory, item, abstention=abstention)["correct"])
            except Unknown:
                unknown += 1
    official = [v for k, v in per_question.items() if k.startswith("official:")]
    bank = [v for k, v in per_question.items() if k.startswith("bank:")]
    if stats["windows"] and not stats["faith_judged"]:
        print("[validation] faith_gate is off in this config: faith_rate is not measured", file=sys.stderr)
    metrics = {
        "role": "builder", "reference": reference, "split": split,
        "primary_name": "bank_accuracy" if bank else "official_accuracy",
        "primary": _mean(bank) if bank else _mean(official),
        "bank_accuracy": _mean(bank), "bank_questions": len(bank),
        "official_accuracy": _mean(official), "official_questions": len(official),
        "guards": {"legal_rate": _rate(stats["legal"], stats["windows"]),
                   "faith_rate": _rate(stats["faithful"], stats["faith_judged"]),
                   "noop_rate": _rate(stats["noop"], stats["legal"]),
                   "fallback_rate": _rate(stats["fallback"], stats["windows"]),
                   "mean_m_tokens": _mean(tokens)},
        "windows": stats["windows"], "failed_cases": failures, "unknown_answers": unknown,
        # bootstrap只比较主指标那一类题：与primary一致
        "per_question": {k: v for k, v in per_question.items() if k.startswith("bank:" if bank else "official:")},
        "official_per_question": {k: v for k, v in per_question.items() if k.startswith("official:")},
    }
    write(out / "metrics.json", metrics)
    return metrics


# ================================================================ attacker

def attacker_metrics(env, prepared, split, out, *, memory_run, snapshot="M_build", role="ATTACKER_EVAL",
                     packs_per_case=8, seed=0, limit=None, keys=None, question_types=None, per_type=None):
    """针对参考记忆的出题能力。每个pack独立（accepted从空开始），便于不同checkpoint逐pack配对比较。"""
    out = Path(out)
    per_pack, counts, failures = {}, Counter(), []
    for context, full_path in contexts(prepared, split, limit, keys, question_types, per_type):
        key = context["key"]
        full = env.memory_module.FullMemory.load(full_path)
        memory = read(Path(memory_run) / key / f"{snapshot}.json")
        pool = make_pool(full, env)
        rng = random.Random(digest([seed, key]))
        chosen = sorted(rng.sample(range(len(pool)), min(packs_per_case, len(pool))))
        scheduler = TypeScheduler(env.cfg) if resolve_type_policy(env.cfg) == "balanced" else None
        for pi in chosen:
            try:
                _, logs, _ = _bank_pack(full, full_path, context, pool[pi], pi, [], env, role,
                                        memory=memory, scheduler=scheduler)
            except Unknown as exc:
                failures.append({"key": key, "pack": pi, "error": str(exc)})
                continue
            status = Counter(row.get("status") for row in logs)
            if status["type_infeasible"] or (status["over_budget"] and len(logs) == 1 and "item" not in logs[0]):
                counts["not_sent"] += 1
                continue
            counts["sent"] += 1
            counts["invalid"] += status["invalid"]
            counts["empty"] += status["empty"]
            proposed = sum(1 for row in logs if row.get("status") in {"accepted", "rejected", "accepted_no_defect"})
            accepted = sum(1 for row in logs if row.get("status") in {"accepted", "accepted_no_defect"})
            defects = sum(1 for row in logs if row.get("defect") in REPAIRABLE_DEFECTS)
            counts["proposed"] += proposed
            counts["accepted"] += accepted
            counts["defects"] += defects
            per_pack[f"pack:{key}:{pi}"] = float(defects)
    metrics = {
        "role": "attacker", "split": split, "memory_run": str(memory_run), "snapshot": snapshot,
        "primary_name": "defects_per_pack", "primary": _rate(counts["defects"], counts["sent"]),
        "guards": {"invalid_rate": _rate(counts["invalid"], counts["sent"]),
                   "accept_rate": _rate(counts["accepted"], counts["proposed"]),
                   "empty_rate": _rate(counts["empty"], counts["sent"])},
        "counts": dict(counts), "failed_packs": failures, "per_question": per_pack,
    }
    write(out / "metrics.json", metrics)
    return metrics


# ================================================================ early stopping

BUILDER_GUARDS = [
    {"name": "legal_rate", "kind": "min", "tolerance": 0.05},
    {"name": "faith_rate", "kind": "min", "tolerance": 0.05},
    {"name": "noop_rate", "kind": "max", "tolerance": 0.15},
    {"name": "fallback_rate", "kind": "max", "tolerance": 0.10},
    {"name": "mean_m_tokens", "kind": "max_ratio", "tolerance": 1.5},
]
ATTACKER_GUARDS = [
    {"name": "invalid_rate", "kind": "max", "tolerance": 0.10},
    {"name": "accept_rate", "kind": "min", "tolerance": 0.10},
]


def paired_bootstrap(current, best, samples=1000, seed=0):
    """返回(P(current均值 > best均值), 均值差, 配对题数)；只用两边都有的题。平局算一半。"""
    common = sorted(set(current) & set(best))
    if not common:
        return 0.0, 0.0, 0
    diffs = [current[k] - best[k] for k in common]
    mean_diff = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    wins = 0.0
    for _ in range(samples):
        total = sum(diffs[rng.randrange(len(diffs))] for _ in range(len(diffs)))
        wins += 1.0 if total > 0 else 0.5 if total == 0 else 0.0
    return wins / samples, mean_diff, len(common)


class EarlyStopper:
    def __init__(self, guards, patience=3, confidence=0.9, min_delta=0.0, guard_patience=1,
                 max_kl=None, bootstrap=1000, seed=0, state=None):
        self.guards, self.patience, self.confidence = guards, patience, confidence
        self.min_delta, self.guard_patience, self.max_kl = min_delta, guard_patience, max_kl
        self.bootstrap, self.seed = bootstrap, seed
        self.state = state or {"history": [], "baseline": None, "best": None, "bad_rounds": 0,
                               "guard_strikes": 0, "stopped": False, "reason": None}

    def violations(self, metrics):
        baseline = (self.state["baseline"] or {}).get("guards", {})
        found = []
        for guard in self.guards:
            name, kind, tol = guard["name"], guard["kind"], guard["tolerance"]
            current, base = metrics.get("guards", {}).get(name), baseline.get(name)
            if current is None or base is None:
                continue
            if ((kind == "min" and current < base - tol) or (kind == "max" and current > base + tol)
                    or (kind == "max_ratio" and base > 0 and current > base * tol)):
                found.append(f"{name}={current:.4g} vs baseline {base:.4g} ({kind} {tol})")
        kl = metrics.get("train", {}).get("kl_loss")
        if self.max_kl is not None and kl is not None and kl > self.max_kl:
            found.append(f"kl_loss={kl:.4g} > {self.max_kl}")
        return found

    def update(self, tag, step, metrics, model_path=None):
        s = self.state
        entry = {"tag": tag, "step": step, "model_path": model_path, "primary": metrics.get("primary"),
                 "guards": metrics.get("guards", {}), "train": metrics.get("train", {})}
        if s["baseline"] is None:
            s["baseline"] = {"tag": tag, "guards": metrics.get("guards", {})}
            s["best"] = {"tag": tag, "step": step, "primary": metrics.get("primary"), "model_path": model_path,
                         "per_question": metrics.get("per_question", {})}
            entry.update(decision="baseline", improved=True, best_tag=tag)
            s["history"].append(entry)
            return entry
        violations = self.violations(metrics)
        prob, diff, paired = paired_bootstrap(metrics.get("per_question", {}), s["best"]["per_question"],
                                              self.bootstrap, self.seed)
        improved = not violations and paired > 0 and prob >= self.confidence and diff > self.min_delta
        entry.update(violations=violations, prob_better=prob, mean_diff=diff, paired_questions=paired,
                     improved=improved)
        s["guard_strikes"] = s["guard_strikes"] + 1 if violations else 0
        if improved:
            s["best"] = {"tag": tag, "step": step, "primary": metrics.get("primary"), "model_path": model_path,
                         "per_question": metrics.get("per_question", {})}
            s["bad_rounds"] = 0
        else:
            s["bad_rounds"] += 1
        if s["guard_strikes"] >= self.guard_patience:
            s["stopped"], s["reason"] = True, "guard: " + "; ".join(violations)
        elif s["bad_rounds"] >= self.patience:
            s["stopped"], s["reason"] = True, f"no significant improvement in {self.patience} evaluations"
        entry.update(decision="stop" if s["stopped"] else "continue", best_tag=s["best"]["tag"],
                     bad_rounds=s["bad_rounds"], guard_strikes=s["guard_strikes"], reason=s["reason"])
        s["history"].append(entry)
        return entry


# ================================================================ CLI

def make_env(config):
    from .environment import Environment
    cfg = Config.load(config)
    return Environment(cfg, TokenBudget(cfg.tokenizer, cfg.tokenizer_revision), connect(cfg.project))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["builder", "attacker"])
    p.add_argument("--config", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--split", choices=["val", "train"], default="val",
                   help="选择checkpoint只能用val；test留到最后报告（用admem.cli longmemeval-eval）")
    p.add_argument("--out", required=True)
    p.add_argument("--role", help="待评checkpoint的服务角色（读取<ROLE>_MODEL/_BASE_URL/_API_KEY）")
    p.add_argument("--val-bank", help="builder：冻结attacker生成的val题库目录")
    p.add_argument("--max-bank-questions", type=int, help="builder：每个case最多用多少道题库题（确定性抽取）")
    p.add_argument("--reference", choices=["full"], help="builder：不构建记忆，用全部原文作上限参考")
    p.add_argument("--memory-run", help="attacker：参考记忆所在的run目录")
    p.add_argument("--snapshot", default="M_build")
    p.add_argument("--packs-per-case", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--keys", nargs="+")
    p.add_argument("--per-type", type=int)
    a = p.parse_args(argv)
    env = make_env(a.config)
    common = dict(limit=a.limit, keys=a.keys, per_type=a.per_type)
    if a.kind == "builder":
        metrics = builder_metrics(env, a.prepared, a.split, a.out, role=a.role or "BUILDER_EVAL",
                                  val_bank=a.val_bank, reference=a.reference,
                                  max_bank_questions=a.max_bank_questions, **common)
    else:
        if not a.memory_run:
            p.error("attacker validation needs --memory-run")
        metrics = attacker_metrics(env, a.prepared, a.split, a.out, memory_run=a.memory_run,
                                   snapshot=a.snapshot, role=a.role or "ATTACKER_EVAL",
                                   packs_per_case=a.packs_per_case, **common)
    print(json.dumps({k: v for k, v in metrics.items() if "per_question" not in k}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
