"""离线比较pack构建方式：官方跨session证据能否同时出现在attacker可见范围内。不调用任何生成模型。

attacker只能用一个pack里看得到的round出题。官方证据分布在多个session时，如果没有任何pack同时
看得到这些session，multi-session/knowledge-update题在结构上就出不来。本脚本对每种
pack_builder:neighbor_mode组合构建pack池（会用到embedding与attacker tokenizer），按attacker的
跨会话视图（含token预算裁剪）统计：

- round_pair_coverage：官方证据中跨session的round对，被同一个pack同时看到的比例
- session_pair_coverage：官方证据session对，被同一个pack同时看到的比例
- all_sessions_in_one_pack：是否有一个pack同时看到全部证据session
- all_rounds_in_one_pack：是否有一个pack同时看到全部证据round

只统计官方证据跨≥2个session的case。官方证据来自private_eval.jsonl（评测标签），只用于比较设计，
结果绝不能交给attacker。请在train split上做选择（默认拒绝test，避免在测试集上调参）。

用法：
  python scripts/pack_coverage.py --config configs/hidden_4b_v2.json \\
      --prepared data/prepared_longmemeval --split train \\
      --modes legacy:topical token:topical token:facts token:hybrid --out runs/pack_coverage_train.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admem.audit_prompts import attacker_rids  # noqa: E402
from admem.common import BudgetExceeded, Config, TokenBudget, connect, rows, write  # noqa: E402
from admem.data import contexts  # noqa: E402
from admem.environment import Environment  # noqa: E402
from admem.packing import PLACEHOLDER_DATE  # noqa: E402
from admem.pipeline import make_pool  # noqa: E402
from admem.prompts import attacker_messages  # noqa: E402

DEFAULT_MODES = ["legacy:topical", "token:topical", "token:facts", "token:hybrid"]


def visible_sets(full, pool, env, fit=True):
    """每个pack在跨会话视图下attacker实际看得到的round集合（fit=True时按真实prompt预算裁剪）。"""
    sets = []
    for pack in pool:
        rids = attacker_rids(full, pack, "multi-session", env.cfg.attacker_view)
        if fit:
            try:
                _, rids = attacker_messages(full, pack, "multi-session", PLACEHOLDER_DATE, [], [],
                                            env.cfg, env.counter)
            except BudgetExceeded:
                rids = []
        sets.append(set(rids))
    return sets


def case_metrics(full, gold_rids, sets):
    gold = [rid for rid in dict.fromkeys(gold_rids) if rid in full.rounds]
    by_session = defaultdict(set)
    for rid in gold:
        by_session[full.rounds[rid].session_id].add(rid)
    if len(by_session) < 2:
        return None
    round_pairs = [(a, b) for a, b in itertools.combinations(gold, 2)
                   if full.rounds[a].session_id != full.rounds[b].session_id]
    session_pairs = list(itertools.combinations(by_session, 2))
    return {
        "gold_sessions": len(by_session),
        "round_pair_coverage": statistics.mean(any(a in v and b in v for v in sets) for a, b in round_pairs),
        "session_pair_coverage": statistics.mean(any(v & by_session[s] and v & by_session[t] for v in sets)
                                                 for s, t in session_pairs),
        "all_sessions_in_one_pack": any(all(v & rids for rids in by_session.values()) for v in sets),
        "all_rounds_in_one_pack": any(set(gold) <= v for v in sets),
    }


def aggregate(values):
    keys = ["round_pair_coverage", "session_pair_coverage", "all_sessions_in_one_pack", "all_rounds_in_one_pack"]
    if not values:
        return {"cases": 0}
    return {"cases": len(values), **{k: round(statistics.mean(float(v[k]) for v in values), 3) for k in keys}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--allow-test", action="store_true", help="允许在test上运行（只用于最终报告，不用于选择设计）")
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="pack_builder:neighbor_mode 组合")
    p.add_argument("--neighbors", type=int, help="覆盖config里的neighbors")
    p.add_argument("--limit", type=int)
    p.add_argument("--keys", nargs="+")
    p.add_argument("--no-fit", action="store_true", help="不按attacker token预算裁剪（只看结构上的可见性）")
    p.add_argument("--out", help="把完整结果另存为JSON")
    a = p.parse_args(argv)
    if a.split == "test" and not a.allow_test:
        p.error("refusing to tune pack construction on the test split; pass --allow-test only for a final report")
    modes = []
    for mode in a.modes:
        builder, _, neighbor = mode.partition(":")
        if builder not in {"legacy", "token"} or neighbor not in {"topical", "facts", "hybrid"}:
            p.error(f"invalid mode {mode!r}; use legacy:topical or token:topical|facts|hybrid")
        if builder == "legacy" and neighbor != "topical":
            p.error("legacy pack builder only supports topical neighbors")
        modes.append((builder, neighbor))
    cfg = Config.load(a.config)
    if a.neighbors is not None:
        cfg.neighbors = a.neighbors
    env = Environment(cfg, TokenBudget(cfg.tokenizer, cfg.tokenizer_revision), connect(cfg.project))
    golds = {row["key"]: row for row in rows(Path(a.prepared) / "private_eval.jsonl")}
    per_mode = {f"{b}:{n}": {"by_type": defaultdict(list), "pool": defaultdict(list)} for b, n in modes}
    for context, full_path in contexts(a.prepared, a.split, a.limit, a.keys):
        gold = golds[context["key"]]
        if gold.get("abstention"):
            continue
        full = env.memory_module.FullMemory.load(full_path)
        for builder, neighbor in modes:
            cfg.pack_builder, cfg.neighbor_mode = builder, neighbor
            report = {}
            pool = make_pool(full, env, report)
            sets = visible_sets(full, pool, env, fit=not a.no_fit)
            name = f"{builder}:{neighbor}"
            stats = per_mode[name]["pool"]
            stats["packs"].append(len(pool))
            stats["visible_rounds"].append(statistics.mean(len(v) for v in sets) if sets else 0)
            stats["visible_sessions"].append(statistics.mean(
                len({full.rounds[r].session_id for r in v}) for v in sets) if sets else 0)
            stats["skipped_rounds"].append(len(report.get("skipped_rounds", [])))
            stats["skipped_sessions"].append(len(report.get("skipped_sessions", [])))
            metrics = case_metrics(full, gold["E"], sets)
            if metrics is not None:
                per_mode[name]["by_type"][gold["type"]].append(metrics)
                per_mode[name]["by_type"]["ALL"].append(metrics)
        print(f"{context['key']} done", file=sys.stderr, flush=True)
    result = {"split": a.split, "attacker_view": cfg.attacker_view, "neighbors": cfg.neighbors,
              "fit_to_attacker_budget": not a.no_fit, "modes": {}}
    for name, data in per_mode.items():
        pool = {k: round(statistics.mean(v), 3) if v else None for k, v in data["pool"].items()}
        result["modes"][name] = {"pool": pool,
                                 "coverage": {t: aggregate(v) for t, v in sorted(data["by_type"].items())}}
    if a.out:
        write(a.out, result)
    cols = ["round_pair_coverage", "session_pair_coverage", "all_sessions_in_one_pack"]
    print("| mode | type | cases | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 3) + "|")
    for name, data in result["modes"].items():
        for qtype, agg in data["coverage"].items():
            print(f"| {name} | {qtype} | {agg['cases']} | " + " | ".join(str(agg.get(c)) for c in cols) + " |")
    print(json.dumps({name: data["pool"] for name, data in result["modes"].items()}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
