"""拆解一次Builder reward probe：奖励由哪些分项驱动、测试题是否可答、忠实/保留失败长什么样。

离线部分只读 probe/samples.jsonl（不调用模型）：
  python scripts/probe_breakdown.py outputs/run/probe/samples.jsonl [--examples 5] [--json out.json]

--ceiling 另外用模型（reader+grade，走现有缓存）对每个状态的测试题作三种回答，区分“Builder没存”和
“这道题本身答不出”：
  before        当前记忆M（与probe的before相同，通常命中缓存）
  M+raw(E)      M补上E的原文（等价于defect()的augmented条件）
  raw(E) only   只给E的原文（全部放进上下文，不检索）
如果 raw(E) only 都答不对，问题在题目/参考答案或reader能力，Builder的gain项对这道题无法得分。
  python scripts/probe_breakdown.py outputs/run/probe/samples.jsonl --ceiling \\
      --config configs/type_aware_4b_nogate.json --states outputs/run/states.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def r3(value):
    return None if value is None else round(value, 3)


def ops_of(sample):
    try:
        return json.loads(sample["raw_text"]).get("ops") or []
    except (TypeError, ValueError, AttributeError):
        return []


def written_entries(sample):
    """与store.apply的changed顺序一致：先UPDATE（按出现顺序），再ADD/MERGE（按出现顺序）。"""
    ops = ops_of(sample)
    return ([op for op in ops if op.get("op") == "UPDATE"] +
            [op for op in ops if op.get("op") in {"ADD", "MERGE"}])


def breakdown(samples, examples=5):
    ok = [s for s in samples if s.get("status") == "ok" and s.get("legal")]
    groups = defaultdict(list)
    for s in ok:
        groups[s["state_id"]].append(s)
    report = {"samples": len(samples), "scored_legal": len(ok), "states": len(groups)}

    # 1. 未评分样本的原因
    report["unknown_errors"] = Counter((s.get("error") or "")[:90] for s in samples
                                       if s.get("status") == "unknown").most_common()

    # 2. 奖励分项：均值，以及组内（同一状态）标准差——GRPO只看组内差别
    names = sorted({k for s in ok for k in (s.get("reward_components") or {})})
    comp = {}
    for name in names:
        values = [s["reward_components"].get(name, 0.0) for s in ok]
        within = [statistics.pstdev([s["reward_components"].get(name, 0.0) for s in g]) for g in groups.values()
                  if len(g) > 1]
        comp[name] = {"mean": r3(mean(values)), "within_state_std": r3(mean(within)),
                      "nonzero_rate": r3(mean([float(v != 0) for v in values]))}
    reward_within = [statistics.pstdev([s["reward"] for s in g]) for g in groups.values() if len(g) > 1]
    report["reward"] = {"mean": r3(mean([s["reward"] for s in ok])),
                        "within_state_std": r3(mean(reward_within)), "components": comp}

    # 3. 与NOOP（奖励恒为0）比较：组均值<0时，RL会被推向“什么都不写”
    state_means = [statistics.mean(s["reward"] for s in g) for g in groups.values()]
    report["noop_comparison"] = {
        "states_mean_below_noop": sum(m < 0 for m in state_means),
        "states": len(state_means),
        "samples_above_noop_rate": r3(mean([float(s["reward"] > 0) for s in ok])),
        "noop_samples": sum(bool(s.get("noop")) for s in ok)}

    # 4. 测试题：编辑前后可答性
    n_tests = [len(g[0].get("before_scores") or []) for g in groups.values()]
    before = [mean([float(x) for x in g[0].get("before_scores") or []]) for g in groups.values()]
    dead = [sid for sid, g in groups.items()
            if not any(any(s.get("before_scores") or []) or any(s.get("test_scores") or []) for s in g)]
    fixed = sum(1 for s in ok for b, a in zip(s.get("before_scores") or [], s.get("test_scores") or []) if a and not b)
    broken = sum(1 for s in ok for b, a in zip(s.get("before_scores") or [], s.get("test_scores") or []) if b and not a)
    report["tests"] = {"tests_per_state_mean": r3(mean(n_tests)), "before_accuracy_mean": r3(mean(before)),
                       "states_where_no_test_is_ever_correct": len(dead),
                       "wrong_to_right_flips": fixed, "right_to_wrong_flips": broken}

    # 5. 操作类型与失败的关系
    op_counts, by_kind = Counter(), defaultdict(list)
    entry_faith = defaultdict(list)
    for s in ok:
        ops = ops_of(s)
        op_counts.update(op.get("op") for op in ops)
        kind = "touches_old_entries" if any(op.get("op") in {"UPDATE", "MERGE", "DELETE"} for op in ops) else "add_only"
        by_kind[kind].append(s)
        for op, verdict in zip(written_entries(s), s.get("faith_scores") or []):
            entry_faith[op.get("op")].append(float(verdict))
    report["operations"] = {
        "op_counts": dict(op_counts),
        "faithful_entry_rate_by_op": {k: r3(mean(v)) for k, v in entry_faith.items()},
        "by_sample_kind": {k: {"samples": len(v), "faith_ratio": r3(mean([s.get("faith_ratio") for s in v])),
                               "information_loss": r3(mean([s.get("information_loss") for s in v])),
                               "reward": r3(mean([s["reward"] for s in v]))} for k, v in by_kind.items()}}

    # 6. 例子：被判不忠实的写入、被丢掉的旧事实
    unfaithful, lost = [], []
    for s in ok:
        for op, verdict in zip(written_entries(s), s.get("faith_scores") or []):
            if not verdict and len(unfaithful) < examples:
                unfaithful.append({"state": s["state_id"][:12], "op": op.get("op"), "text": op.get("text"),
                                   "prov": op.get("prov")})
        for check in s.get("retention_checks") or []:
            missing = [check["facts"][i]["text"] for i in check.get("missing_fact_indices") or []
                       if i < len(check.get("facts") or [])]
            if missing and len(lost) < examples:
                lost.append({"state": s["state_id"][:12], "entry": check["entry_id"], "missing_facts": missing[:4]})
    report["examples"] = {"unfaithful_writes": unfaithful, "lost_facts": lost}
    report["timing"] = {"generation_seconds_per_sample": r3(mean([s.get("generation_seconds") for s in samples])),
                        "scoring_seconds_per_sample": r3(mean([s.get("scoring_seconds") for s in samples
                                                               if not s.get("score_cache_hit")]))}
    return report


def make_env(config):
    from admem.common import Config, TokenBudget, connect
    from admem.environment import Environment
    cfg = Config.load(config)
    return Environment(cfg, TokenBudget(cfg.tokenizer, cfg.tokenizer_revision), connect(cfg.project))


def ceiling(env, states_path, state_ids, limit=None):
    """对每个状态的每道测试题，比较 M / M+raw(E) / raw(E) only 三种条件下是否答对（调用reader与grade）。"""
    from admem.common import Unknown
    from admem.store import augment, raw_chunks
    cfg = env.cfg
    results, errors = [], Counter()
    picked = [r for r in rows(states_path) if r["id"] in state_ids][:limit]
    for row in picked:
        state = row["state"]
        full = env.memory_module.FullMemory.load(state["full_path"])
        for q in state["tests"]:
            q = {k: q[k] for k in ("q", "a", "type", "question_date", "E")}
            raw = raw_chunks(full, q["E"], env.counter, cfg.timeline_tokens)
            outcome = {"state": row["id"][:12], "q": q["q"], "a": q["a"], "E_sessions": len({r.split(":")[0] for r in q["E"]})}
            for name, memory, all_memory in (("before", state["M"], False), ("M+raw(E)", augment(state["M"], raw), False),
                                              ("raw(E) only", raw, True)):
                try:
                    answer = env.answer(memory, q, all_memory=all_memory)
                    outcome[name], outcome[name + ":answer"] = answer["correct"], answer["answer"]
                except Unknown as exc:
                    outcome[name] = None
                    errors[str(exc)[:80]] += 1
            results.append(outcome)
    summary = {name: r3(mean([float(o[name]) for o in results if o.get(name) is not None]))
               for name in ("before", "M+raw(E)", "raw(E) only")}
    return {"tests": len(results), "accuracy": summary, "errors": errors.most_common(),
            "unanswerable_even_with_raw_E": [o for o in results if o.get("raw(E) only") is False][:10],
            "details": results}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples")
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--json")
    p.add_argument("--ceiling", action="store_true")
    p.add_argument("--config")
    p.add_argument("--states")
    p.add_argument("--ceiling-states", type=int, help="ceiling最多检查几个状态")
    a = p.parse_args(argv)
    samples = rows(a.samples)
    report = breakdown(samples, a.examples)
    if a.ceiling:
        if not a.config or not a.states:
            p.error("--ceiling needs --config and --states")
        report["ceiling"] = ceiling(make_env(a.config), a.states, {s["state_id"] for s in samples}, a.ceiling_states)
    printable = {k: v for k, v in report.items() if k != "ceiling"}
    if "ceiling" in report:
        printable["ceiling"] = {k: v for k, v in report["ceiling"].items() if k != "details"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
