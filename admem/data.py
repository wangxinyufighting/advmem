"""数据准备与严格隔离：策略只能读取context，官方q/a留在private_eval.jsonl。"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from .common import digest, read, rows, write, write_rows


def iter_cases(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        yield from rows(path)
        return
    try:
        import ijson
    except ImportError:
        if path.stat().st_size > 50_000_000:
            raise RuntimeError("大JSON需要ijson，请安装requirements.txt")
        yield from read(path)
    else:
        with path.open("rb") as f:
            yield from ijson.items(f, "item", use_float=True)


def history_group(full):
    """完全相同的历史放在同一split；不使用answer_* ID或证据标签做分组。"""
    values = []
    for s in full.sessions.values():
        msgs = [{"role": m["role"], "content": m["content"]}
                for rid in s.rids for m in full.rounds[rid].messages]
        values.append([s.date, msgs])
    return digest(sorted(values, key=lambda v: digest(v))), {digest(v) for v in values}


def assign_splits(catalog, sizes, seed):
    total = len(catalog)
    if not total or len(sizes) != 3 or any(type(n) is not int or n < 0 for n in sizes):
        raise ValueError("Need nonempty data and three nonnegative split sizes")
    if sum(sizes) != total:
        raise ValueError(f"split sizes sum={sum(sizes)} but cases={total}; explicitly choose matching sizes")
    names = ["train", "val", "test"]
    proportions = dict(zip(names, [n / total for n in sizes]))
    grouped = defaultdict(list)
    for row in catalog:
        grouped[row["history_group"]].append(row)
    global_types = Counter(r["stratum"] for r in catalog)
    counts = {s: Counter() for s in names}
    assigned = {s: [] for s in names}
    # 大组先分配，完全相同历史绝不拆开。精确300/50/150可能因组大小偏移，报告实际值。
    order = sorted(grouped.items(), key=lambda x: (-len(x[1]), digest([seed, x[0]])))
    for _, members in order:
        bytype = Counter(m["stratum"] for m in members)
        def cost(split):
            if proportions[split] == 0:
                return float("inf")
            n = len(assigned[split])
            target = proportions[split] * total
            delta = ((n + len(members) - target) ** 2 - (n - target) ** 2) / max(1, target)
            for typ, amount in bytype.items():
                t = global_types[typ] * proportions[split]
                old = counts[split][typ]
                delta += ((old + amount - t) ** 2 - (old - t) ** 2) / max(1, t)
            return delta
        split = min(names, key=cost)
        for row in members:
            row["split"] = split
            assigned[split].append(row["key"])
        counts[split].update(bytype)
    return {s: {"requested": sizes[i], "actual": len(assigned[s]), "strata": dict(counts[s])}
            for i, s in enumerate(names)}


def history_for_builder(case):
    """Whitelist history and disambiguate duplicate source session IDs.

    Some LongMemEval exports reuse a session ID inside one case. FullMemory
    needs unique internal keys, so only the duplicate key is suffixed; dates,
    messages, ordering, and all message text remain unchanged.
    """
    seen = {}
    used = set()
    session_ids = []
    for value in case["haystack_session_ids"]:
        base = str(value)
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        candidate = base if occurrence == 0 else f"{base}__duplicate_{occurrence}"
        while candidate in used:
            occurrence += 1
            seen[base] = occurrence + 1
            candidate = f"{base}__duplicate_{occurrence}"
        used.add(candidate)
        session_ids.append(candidate)
    return {"haystack_session_ids": session_ids,
            "haystack_dates": case["haystack_dates"],
            "haystack_sessions": case["haystack_sessions"]}


def prepare(data, out, core_memory, sizes=(300, 50, 150), seed=0):
    out = Path(out)
    if (out / "manifest.json").exists():
        raise ValueError("Prepared dataset already exists; use a new directory")
    catalog, private, seen, session_sets = [], [], set(), {}
    for i, case in enumerate(iter_cases(data)):
        qid = case["question_id"]
        if qid in seen:
            raise ValueError("Duplicate question_id")
        seen.add(qid)
        # Only the session IDs are disambiguated for FullMemory's internal keys;
        # the original case (incl. haystack_sessions) is still used for labels.
        full = core_memory.FullMemory.build(history_for_builder(case))
        key = f"c{i:04d}"
        folder = out / "cases" / key
        full.save(folder / "full.json")
        group, sessions = history_group(full)
        session_sets[key] = sessions
        row = {"key": key, "case_index": i, "question_id": qid,
               "question_type": case["question_type"], "question_date": case["question_date"],
               "full_hash": full.fingerprint, "history_group": group,
               "stratum": "abstention" if qid.endswith("_abs") else case["question_type"]}
        catalog.append(row)
        # 官方标签不写入context/model prompt。它们只供独立evaluate子命令读取。
        evidence = []
        for r in full.rounds.values():
            s = full.sessions[r.session_id]
            if any(case["haystack_sessions"][s.original_index][mi].get("has_answer") is True for mi in r.message_indices):
                evidence.append(r.rid)
        private.append({"key": key, "question_id": qid, "q": case["question"], "a": str(case["answer"]),
                        "type": case["question_type"], "question_date": case["question_date"],
                        "E": evidence, "answer_session_ids": case.get("answer_session_ids", []),
                        "abstention": qid.endswith("_abs")})
    split_report = assign_splits(catalog, sizes, seed)
    overlaps = []
    for i, a in enumerate(catalog):
        for b in catalog[i + 1:]:
            if a["split"] == b["split"]:
                continue
            x, y = session_sets[a["key"]], session_sets[b["key"]]
            score = len(x & y) / max(1, len(x | y))
            if score >= 0.8:
                overlaps.append({"a": a["key"], "b": b["key"], "session_jaccard": score})
    for row in catalog:
        # context不包含qid/_abs，避免将不可答标记或answer_*当线索。
        context = {k: row[k] for k in ["key", "question_type", "question_date", "full_hash", "split"]}
        write(out / "cases" / row["key"] / "context.json", context)
    write(out / "manifest.json", {"schema": 1, "seed": seed, "cases": catalog,
          "splits": split_report, "near_duplicate_cross_split": overlaps,
          "note": "Exact histories grouped; high-overlap pairs require inspection. All500 prompts were previously inspected."})
    write_rows(out / "private_eval.jsonl", private)
    return split_report


def contexts(prepared, split, limit=None, keys=None, question_types=None):
    p = Path(prepared)
    manifest = read(p / "manifest.json")
    wanted_types = set(question_types or [])
    entries = [r for r in manifest["cases"]
               if r["split"] == split
               and (keys is None or r["key"] in keys)
               and (not wanted_types or r.get("question_type") in wanted_types)]
    if limit:
        entries = entries[:limit]
    for row in entries:
        folder = p / "cases" / row["key"]
        yield read(folder / "context.json"), folder / "full.json"


def hint(context, cfg):
    return context["question_type"] if cfg.hint_mode == "target_type" else None


def policy_type(context, cfg, identity, allowed=None):
    """target_type固定为case题型；不在allowed内时返回None（该pack跳过，不强迫编题）。"""
    from .audit_prompts import TYPES
    if cfg.hint_mode == "target_type":
        qtype = context["question_type"]
        return qtype if allowed is None or qtype in allowed else None
    choices = [t for t in TYPES if allowed is None or t in allowed]
    return choices[int(digest([cfg.seed, identity]), 16) % len(choices)] if choices else None
