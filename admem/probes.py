"""Builder基础能力诊断：独立合成情境，不使用LongMemEval官方问题作为训练奖励。"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
import hashlib
import json
import os
import random
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

from .common import BudgetExceeded, Progress, Unknown, digest, read, write, write_rows, parse
from .data import TypeScheduler, resolve_type_policy
from .pipeline import attacker_state, builder_state, make_pool
from .store import neutral_round


def _json_text(value):
    """把模型返回值稳定地转换成可记录的文本。"""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _answer_correct(result):
    """从现有 builder_score 结果中拆出答案是否全部正确。"""
    if result.get("answer_correct") is not None:
        return result["answer_correct"]
    accuracy = result.get("accuracy")
    if accuracy is None:
        return None
    return bool(accuracy == 1)


def _edit_quality_constraint(result):
    """训练与 probe 共用编辑质量；family 仅用于汇总分组。"""
    ok = result.get("edit_quality_pass")
    if ok is None or ok:
        return ok, None
    reasons = []
    if result.get("faith") is False:
        reasons.append("not all written entries are faithful")
    if result.get("loss", 0) > 0:
        reasons.append("previously correct answers were lost")
    if result.get("retention_pass") is False:
        reasons.append("key information was not retained")
    if result.get("redundancy_ratio", 0) > 0:
        reasons.append("redundant or ineffective edits")
    return False, result.get("edit_quality_reason") or "; ".join(reasons) or "edit quality check failed"


def make_suite(env, out, variants=5):
    out = Path(out)
    suite = []
    families = ["append_event", "update_preserve_old", "merge_timeline", "dedup_refine", "assistant_list", "preference_context", "temporal_anchor", "noop_repeat"]
    for variant in range(variants):
        for family in families:
            name = f"Juniper_{variant}"
            if family == "append_event":
                sources = [f"I adopted a cat named {name} in January.", f"In March I also adopted a dog named Cedar_{variant}."]
                typ, mode = "multi-session", "patch"
                mem = [{"id": "m1", "text": f"In January the user adopted a cat named {name}.", "prov": ["s1:r1"], "kind": "card"}]
                tests = [("What are the names of my adopted pets?", f"{name} and Cedar_{variant}", ["s1:r1", "s2:r1"])]
                raw_ids, old_ids = ["s2:r1"], ["m1"]
            elif family == "update_preserve_old":
                sources = [f"For the {name} program, my initial weekly training schedule was Monday and Friday.",
                           f"Starting March 1, I train Tuesday, Thursday, and Saturday instead for the {name} program."]
                typ, mode = "knowledge-update", "patch"
                mem = [{"id": "m1", "text": f"The user's initial weekly training schedule for the {name} program was Monday and Friday.", "prov": ["s1:r1"], "kind": "card"}]
                tests = [(f"What were my original training days in the {name} program?", "Monday and Friday", ["s1:r1"]),
                         (f"What are my training days in the {name} program after March 1?", "Tuesday, Thursday, and Saturday", ["s2:r1"])]
                raw_ids, old_ids = ["s2:r1"], ["m1"]
            elif family == "merge_timeline":
                sources = [f"I attended Fern Academy_{variant} for two years before taking a break.",
                           f"After the break I attended Grove College_{variant} for three years."]
                typ, mode = "multi-session", "refine"
                mem = [{"id": f"m{i+1}", "text": "The user stated the following education history: " + text,
                        "prov": [f"s{i+1}:r1"], "kind": "card"} for i, text in enumerate(sources)]
                tests = [("How many years did I spend attending these two schools, excluding the break?", "5 years", ["s1:r1", "s2:r1"]),
                         ("Which school did I attend before the break?", f"Fern Academy_{variant}", ["s1:r1"])]
                raw_ids, old_ids = [], ["m1", "m2"]
            elif family == "dedup_refine":
                sources = [f"My cat is named {name}.", f"As I mentioned earlier, my cat's name is {name}."]
                typ, mode = "single-session-user", "refine"
                mem = [{"id": f"m{i+1}", "text": f"The user's cat is named {name}.", "prov": [f"s{i+1}:r1"], "kind": "card"} for i in range(2)]
                tests = [("What is my cat's name?", name, ["s1:r1"])]
                raw_ids, old_ids = [], ["m1", "m2"]
            elif family == "assistant_list":
                sources = ["Please recommend three workspace improvements."]
                typ, mode, mem = "single-session-assistant", "stream", []
                tests = [("What was the second workspace improvement you recommended?", f"A monitor stand named Rise_{variant}", ["s1:r1"])]
                raw_ids, old_ids = ["s1:r1"], []
            elif family == "temporal_anchor":
                sources = [f"Yesterday I bought a desk lamp named Glow_{variant}."]
                typ, mode, mem = "temporal-reasoning", "stream", []
                tests = [(f"On what date did I buy the Glow_{variant} lamp?", "2023-01-04", ["s1:r1"])]
                raw_ids, old_ids = ["s1:r1"], []
            elif family == "noop_repeat":
                sources = [f"My cat is named {name}.", f"As I said before, my cat is named {name}."]
                typ, mode = "single-session-user", "patch"
                mem = [{"id": "m1", "text": f"The user's cat is named {name}.", "prov": ["s1:r1"], "kind": "card"}]
                tests = [("What is my cat's name?", name, ["s1:r1"])]
                raw_ids, old_ids = ["s2:r1"], ["m1"]
            else:
                hour = 7 + variant % 4
                sources = [f"I finish work at {hour} pm, avoid screens before bed, and enjoy quiet stretching."]
                typ, mode, mem = "single-session-preference", "stream", []
                tests = [("What could I do to unwind after work?", f"Suggest quiet screen-free activities after {hour} pm, such as gentle stretching; avoid screen-based entertainment.", ["s1:r1"])]
                raw_ids, old_ids = ["s1:r1"], []
            sessions = [[{"role": "user", "content": text}, {"role": "assistant", "content": "Noted."}] for text in sources]
            if family == "assistant_list":
                sessions[0][1]["content"] = f"1. Improve lighting.\n2. Add a monitor stand named Rise_{variant}.\n3. Add a footrest."
                if variant % 2:
                    sessions[0][0]["content"] = "Please list 100 example workspace item labels in order."
                    sessions[0][1]["content"] = "\n".join(f"{j}. DeskItem_{variant}_{j:03d}" for j in range(1, 101))
                    tests = [(f"What was item number {j} in your list of workspace item labels?",
                              f"DeskItem_{variant}_{j:03d}", ["s1:r1"]) for j in [2, 27, 100]]
            case = {"haystack_session_ids": [f"source_{i}" for i in range(len(sources))],
                    "haystack_dates": ["2023-01-05", "2023-03-05"][:len(sources)], "haystack_sessions": sessions}
            full = env.memory_module.FullMemory.build(case)
            key = f"probe_{family}_{variant}"
            full_path = out / "full" / f"{key}.json"
            full.save(full_path)
            context = {"key": key, "split": "probe", "question_type": typ, "question_date": "2023-04-01"}
            questions = [{"q": q, "a": a, "type": typ, "question_date": context["question_date"], "E": E} for q, a, E in tests]
            x = [neutral_round(full, rid) for rid in raw_ids]
            state = builder_state(full, full_path, context, mem, x, mode, questions, env, old_ids)
            suite.append({"id": digest(state), "family": family, "state": state, "prompt": state["prompt"]})
    write_rows(out / "probes.jsonl", suite)
    return suite


def _item_summary(scored):
    """Compact per-question verdicts so identical rewards with different outputs are debuggable."""
    summary = []
    for d in scored.get("items", []) or []:
        defect = d.get("defect")
        summary.append({"status": d.get("status"),
                        "defect": defect.get("kind") if isinstance(defect, dict) else None,
                        "weight": d.get("weight")})
    return summary


def run_attacker_probe(contexts_iter, env, out, samples=4, role="ATTACKER", max_packs=None,
                       temperature=1.0, pack_sample=None, pack_seed=0, memory_run=None,
                       memory_snapshot="M_build"):
    """诊断 attacker reward 是否有组内学习信号（GRPO 要求每个 prompt 内 reward 有方差）。

    对每个 pack 构造一个 attacker state（honor ``cfg.attacker_type_filter`` 等开关），
    采样 ``samples`` 条 completion，用 ``env.attacker_score`` 打分；报告组内 std、
    非零方差组占比、reward 分布。reward 全部相同 = 无梯度信号。

    ``pack_sample=N`` 随机抽 N 个 pack（可复现，seed 由 ``pack_seed+case`` 决定）
    而非跑满该 case 的所有 session；``max_packs=N`` 仍是取前 N 个。

    ``memory_run`` 指向一次 ``run --mode build`` 的输出目录时，attacker 针对每个 case 的
    ``memory_snapshot``（默认 M_build）出题，奖励里的 defect 项才有意义；不传时沿用空 M，
    此时通过 gate 的题几乎都被判为 missing，defect 项退化为常数（summary 中会注明）。
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples_path, raw_path = out / "samples.jsonl", out / "raw_samples.jsonl"
    for path in (samples_path, raw_path):
        if path.exists():
            path.unlink()
    records, over_budget, memory_cases = [], 0, 0
    for context, full_path in contexts_iter:
        full = env.memory_module.FullMemory.load(full_path)
        memory = read(Path(memory_run) / context["key"] / f"{memory_snapshot}.json") if memory_run else []
        memory_cases += bool(memory)
        scheduler = TypeScheduler(env.cfg) if resolve_type_policy(env.cfg) == "balanced" else None
        pool = make_pool(full, env)
        if pack_sample is not None:
            rng = random.Random(digest([pack_seed, context["key"]]))
            if pack_sample < len(pool):
                pool = sorted(rng.sample(pool, pack_sample), key=lambda p: p.pack_id)
        elif max_packs:
            pool = pool[:max_packs]
        bar = Progress(len(pool), label=f"{context['key']} attacker-probe")
        for pi, pack in enumerate(pool):
            try:
                state = attacker_state(full, full_path, context, memory, pack, [], env, ["probe", pi], scheduler)
            except BudgetExceeded:
                over_budget += 1
                bar.update(suffix=f"p{pi} over_budget")
                continue
            if state is None:
                bar.update(suffix=f"p{pi} type_infeasible")
                continue
            group = []
            for i in range(samples):
                nonce = "probe:" + digest([context["key"], pi, i, env.policy(role).tag])
                raw = env.policy(role).complete(state["prompt"], nonce, temperature=temperature)
                raw_text = _json_text(raw)
                try:
                    scored = dict(env.attacker_score(full, state, raw))
                except BudgetExceeded as exc:
                    # 确定性的judge预算溢出：该样本不可评分（reward=None，不计入统计），继续下一个。
                    scored = {"reward": None, "effective_reward": None, "legal": None,
                              "reason": f"over_budget: {exc}", "items": []}
                except Unknown as exc:
                    write(out / "interrupted.json", {"case": context["key"], "pack": pi,
                                                     "sample": i, "error": str(exc)})
                    raise
                record = {
                    "case_key": context["key"], "pack": pi, "pack_id": pack.pack_id,
                    "seed": pack.seed_rids[0].split(":")[0] if pack.seed_rids else None,
                    "sample_idx": i,
                    "qtype": state["qtype"], "temperature": temperature, "nonce": nonce,
                    "raw_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16],
                    "reward": scored.get("reward"),
                    "effective_reward": scored.get("effective_reward"),
                    "legal": scored.get("legal"), "reason": scored.get("reason"),
                    "items": _item_summary(scored),
                }
                records.append(record)
                group.append(record)
                _append_jsonl(samples_path, record)
                _append_jsonl(raw_path, {"case_key": context["key"], "pack": pi,
                                         "sample_idx": i, "raw_text": raw_text})
            rewards = [float(r["reward"]) for r in group if r.get("reward") is not None]
            std = pstdev(rewards) if len(rewards) > 1 else 0.0
            bar.update(suffix=f"p{pi} qtype={state['qtype']} mean={mean(rewards) if rewards else float('nan'):.3f} std={std:.3f}")
        bar.close()

    groups = defaultdict(list)
    for record in records:
        groups[(record["case_key"], record["pack"])].append(record)
    per_group = []
    for (case_key, pack), group in sorted(groups.items()):
        rewards = [float(r["reward"]) for r in group if r.get("reward") is not None]
        if len(rewards) < 2:
            continue
        unique_raw = len({r["raw_hash"] for r in group})
        if pstdev(rewards) > 1e-8:
            diagnosis = "signal"              # 组内有方差，能提供GRPO梯度
        elif all(r < 0 for r in rewards):
            diagnosis = "all_fail"            # 全部被拒/非法：模型能力问题
        elif unique_raw > 1:
            diagnosis = "same_reward_collapse"  # 输出不同但同分：奖励区分度问题
        else:
            diagnosis = "deterministic"       # 输出几乎不变：采样多样性问题
        per_group.append({
            "case_key": case_key, "pack": pack, "pack_id": group[0].get("pack_id"),
            "seed": group[0].get("seed"), "qtype": group[0].get("qtype"),
            "samples": len(rewards), "mean": mean(rewards), "std": pstdev(rewards),
            "min": min(rewards), "max": max(rewards),
            "unique_rewards": len(set(rewards)), "unique_raw": unique_raw,
            "diagnosis": diagnosis,
        })
    all_rewards = [float(r["reward"]) for r in records if r.get("reward") is not None]
    signal = [g for g in per_group if g["std"] > 1e-8]
    diagnoses = Counter(g["diagnosis"] for g in per_group)
    total_groups = len(per_group)
    summary = {
        "role": role, "model": env.policy(role).client.model, "temperature": temperature,
        "type_policy": resolve_type_policy(env.cfg),
        "memory": {"run": str(memory_run) if memory_run else None, "snapshot": memory_snapshot if memory_run else None,
                   "cases_with_memory": memory_cases,
                   "note": None if memory_run else "empty M: accepted questions are almost always 'missing', "
                                                     "so the defect term is constant; pass --memory-run"},
        "over_budget_packs": over_budget,
        "groups": total_groups, "samples_per_group": samples,
        "groups_with_signal": len(signal),
        "signal_group_rate": len(signal) / total_groups if total_groups else None,
        "diagnosis_counts": dict(diagnoses),
        "diagnosis_rates": ({k: v / total_groups for k, v in diagnoses.items()} if total_groups else {}),
        "note": "diagnosis per zero-variance group: all_fail=模型能力, "
                "same_reward_collapse=奖励区分度(可能只是等价质量), deterministic=采样多样性",
        "reward_std_mean": mean([g["std"] for g in per_group]) if total_groups else None,
        "reward_overall_mean": mean(all_rewards) if all_rewards else None,
        "reward_overall_std": pstdev(all_rewards) if len(all_rewards) > 1 else None,
        "reward_min": min(all_rewards) if all_rewards else None,
        "reward_max": max(all_rewards) if all_rewards else None,
        "negative_rate": mean(float(r < 0) for r in all_rewards) if all_rewards else None,
        "zero_rate": mean(float(r == 0) for r in all_rewards) if all_rewards else None,
        "per_group": per_group,
    }
    write(out / "summary.json", summary)
    return summary


def run_probe(records, env, out, samples=8, role="BUILDER", limit=None, temperature=1.0):
    """运行 Builder probe，并保存逐样本、可审计的评估记录。

    采样温度默认 1.0，对齐 VERL GRPO 的 rollout.temperature，否则诊断比训练保守。
    """
    from .common import rows

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = list(rows(records))
    if not records:
        raise ValueError("No probe states")
    empty_test = sum(1 for r in records if not r["state"].get("tests"))
    records = [r for r in records if r["state"].get("tests")]
    if not records:
        raise ValueError(
            "No probe states with non-empty tests; the answer-retention reward "
            "requires test questions (check that `bank` produced accepted questions)"
        )
    # Collected real-history windows often have no eligible bank question.
    # Apply the probe budget to scoreable states, not hash-ordered empty ones.
    if limit is not None:
        records = records[:limit]

    # 每次 probe 使用新的输出目录；若目录已存在，避免旧样本混入本次汇总。
    samples_path = out / "samples.jsonl"
    raw_path = out / "raw_samples.jsonl"
    for path in (samples_path, raw_path):
        if path.exists():
            path.unlink()

    aggregates = defaultdict(list)
    # 瞬时云端失败不应中止整轮诊断：单样本记unknown继续，连续失败到上限才中止。
    fail_limit = max(1, int(os.getenv("PROBE_FAIL_LIMIT", "10")))
    consecutive_errors = 0

    bar = Progress(len(records) * samples, label="builder-probe")
    try:
        for row in records:
            full = env.memory_module.FullMemory.load(row["state"]["full_path"])
            values = []
            family = row.get("family", row["state"]["mode"])
            score_cache = OrderedDict()

            with env.cached_answers():
                for i in range(samples):
                    # 禁用逐字回退；所有结果完全来自该模型本次提议。
                    nonce = "probe:" + digest([row["id"], i, env.policy(role).tag])
                    raw_text, raw_hash, record = None, None, None
                    generation_seconds = scoring_seconds = 0.0
                    score_cache_hit = False
                    phase, started = "generation", perf_counter()
                    try:
                        raw = env.policy(role).complete(
                            row["prompt"], nonce, temperature=temperature
                        )
                        generation_seconds = perf_counter() - started
                        phase, started = "scoring", perf_counter()
                        raw_text = _json_text(raw)
                        raw_hash = hashlib.sha256(
                            raw_text.encode("utf-8")
                        ).hexdigest()[:16]
                        score_cache_hit = raw_text in score_cache
                        if score_cache_hit:
                            score_cache.move_to_end(raw_text)
                            result = deepcopy(score_cache[raw_text])
                        else:
                            result = dict(env.builder_score(full, row["state"], raw))
                            result.pop("proposed", None)

                        legal = result.get("legal")
                        faith = result.get("faith")
                        answer_correct = _answer_correct(result)
                        quality_ok, quality_reason = _edit_quality_constraint(result)
                        success = bool(
                            legal
                            and faith
                            and answer_correct is True
                            and quality_ok is True
                        )

                        result.update({
                            "answer_correct": answer_correct,
                            # Legacy column names remain readable, but carry no family-specific rule.
                            "family_constraint_pass": quality_ok,
                            "family_constraint_reason": quality_reason,
                            "edit_quality_pass": quality_ok,
                            "edit_quality_reason": quality_reason,
                            "noop": parse(raw).get("ops") == [] if legal else None,
                            "compression_success": (
                                bool(result.get("safe_compression") and result.get("shrink_ratio", 0) > 0)
                                if row["state"]["mode"] == "refine" and legal else None
                            ),
                            "success": success,
                            "status": "ok",
                        })
                        if not score_cache_hit:
                            # 只缓存完成的评分；失败/Unknown 留待后续相同候选重新检查。
                            score_cache[raw_text] = deepcopy(result)
                            if len(score_cache) > 64:
                                score_cache.popitem(last=False)
                        scoring_seconds = perf_counter() - started
                        consecutive_errors = 0
                    except Unknown as exc:
                        # 瞬时/服务端失败记unknown（不计入reward统计）；连续失败到上限才中止。
                        consecutive_errors += 1
                        if phase == "generation":
                            generation_seconds = perf_counter() - started
                        else:
                            scoring_seconds = perf_counter() - started
                        record = {
                            "state_id": row["id"],
                            "family": family,
                            "mode": row["state"]["mode"],
                            "sample_idx": i,
                            "temperature": temperature,
                            "nonce": nonce,
                            "raw_hash": raw_hash,
                            "raw_text": raw_text,
                            "status": "unknown",
                            "error": str(exc),
                            "reward": None,
                            "effective_reward": None,
                            "generation_seconds": generation_seconds,
                            "scoring_seconds": scoring_seconds,
                            "score_cache_hit": score_cache_hit,
                        }
                        _append_jsonl(samples_path, record)
                        _append_jsonl(raw_path, {
                            "state_id": row["id"],
                            "family": family,
                            "sample_idx": i,
                            "temperature": temperature,
                            "raw_hash": raw_hash,
                            "raw_text": raw_text,
                        })
                        values.append(record)
                        bar.update(suffix=f"{row['state']['case_key']} sample={i + 1}/{samples} unknown")
                        if consecutive_errors >= fail_limit:
                            write(
                                out / "interrupted.json",
                                {"case": row["id"], "sample": i, "error": str(exc),
                                 "consecutive_errors": consecutive_errors},
                            )
                            raise
                        continue

                    record = {
                        "state_id": row["id"],
                        "family": family,
                        "mode": row["state"]["mode"],
                        "sample_idx": i,
                        "temperature": temperature,
                        "nonce": nonce,
                        "raw_hash": raw_hash,
                        "raw_text": raw_text,
                        "legal": result.get("legal"),
                        "legal_reason": result.get("legal_reason"),
                        "faith": result.get("faith"),
                        "faith_reason": result.get("faith_reason"),
                        "answer_correct": result.get("answer_correct"),
                        "family_constraint_pass": result.get("family_constraint_pass"),
                        "family_constraint_reason": result.get("family_constraint_reason"),
                        "reward": result.get("reward"),
                        "effective_reward": result.get("effective_reward"),
                        "success": result.get("success"),
                        "status": result.get("status", "ok"),
                        "generation_seconds": generation_seconds,
                        "scoring_seconds": scoring_seconds,
                        "score_cache_hit": score_cache_hit,
                    }
                    # 保存每个奖励分项及对应证据，便于区分内容错误、压缩、冗余和长度差异。
                    for key in (
                        "accuracy", "before_accuracy", "delta", "gain", "loss", "before_scores", "test_scores",
                        "faith_ratio", "faith_scores", "before_tokens", "after_tokens", "growth_ratio",
                        "shrink_ratio", "compression_eligible", "safe_compression", "reward_components", "retention_pass",
                        "information_loss", "retention_checks", "redundancy_ratio", "redundant_ops",
                        "edit_quality_pass", "edit_quality_reason", "noop", "compression_success", "answer_checks",
                    ):
                        if key in result:
                            record[key] = result[key]

                    _append_jsonl(samples_path, record)
                    _append_jsonl(raw_path, {
                        "state_id": row["id"],
                        "family": family,
                        "sample_idx": i,
                        "temperature": temperature,
                        "raw_hash": raw_hash,
                        "raw_text": raw_text,
                    })

                    values.append(record)
                    bar.update(suffix=f"{row['state']['case_key']} sample={i + 1}/{samples} "
                                      f"score_cache_hit={score_cache_hit}")

            aggregates[family].append(values)

    finally:
        bar.close()

    def rate(values, key):
        checked = [x[key] for x in values if x.get(key) is not None]
        return mean(float(bool(x)) for x in checked) if checked else None

    def all_sample_pass_rate(values, key):
        # Unknown stays null in samples; this is the observed pass fraction over all attempts.
        return sum(x.get(key) is True for x in values) / len(values) if values else None

    def numeric_mean(values, key):
        checked = [x[key] for x in values if x.get(key) is not None]
        return mean(checked) if checked else None

    def reward_values(group, key):
        return [float(x[key]) for x in group if x.get(key) is not None]

    report = []
    for family, groups in aggregates.items():
        values = [x for group in groups for x in group]
        reward_stds = [pstdev(v) for v in (reward_values(g, "reward") for g in groups) if len(v) > 1]
        effective_stds = [pstdev(v) for v in (reward_values(g, "effective_reward") for g in groups) if len(v) > 1]
        report.append({
            "family": family,
            "states": len(groups),
            "samples_per_state": samples,
            "samples": len(values),
            "unknown_samples": sum(x.get("status") == "unknown" for x in values),
            "score_cache_hits": sum(x.get("score_cache_hit", False) for x in values),
            "generation_seconds": sum(x.get("generation_seconds", 0.0) for x in values),
            "scoring_seconds": sum(x.get("scoring_seconds", 0.0) for x in values),
            "legal_evaluated_samples": sum(x.get("legal") is not None for x in values),
            "faith_evaluated_samples": sum(x.get("faith") is not None for x in values),
            "answer_evaluated_samples": sum(x.get("answer_correct") is not None for x in values),
            "retention_evaluated_samples": sum(x.get("retention_pass") is not None for x in values),
            "legal_rate": all_sample_pass_rate(values, "legal"),
            "legal_rate_evaluated": rate(values, "legal"),
            "faith_rate_all_samples": all_sample_pass_rate(values, "faith"),
            "faith_rate_evaluated": rate(values, "faith"),
            "faith_ratio_mean": numeric_mean(values, "faith_ratio"),
            "answer_correct_rate": rate(values, "answer_correct"),
            "family_constraint_rate": rate(values, "family_constraint_pass"),
            "edit_quality_rate": rate(values, "edit_quality_pass"),
            "retention_pass_rate": rate(values, "retention_pass"),
            "compression_success_rate": rate(values, "compression_success"),
            "noop_rate": rate(values, "noop"),
            "gain_mean": numeric_mean(values, "gain"),
            "loss_mean": numeric_mean(values, "loss"),
            "information_loss_mean": numeric_mean(values, "information_loss"),
            "redundancy_ratio_mean": numeric_mean(values, "redundancy_ratio"),
            "success_rate_all_samples": all_sample_pass_rate(values, "success"),
            "first_sample_success_rate": (
                mean(float(group[0].get("success") is True) for group in groups if group)
                if any(groups) else None
            ),
            "empirical_at_least_one_success": mean(
                float(any(x.get("success") for x in group)) for group in groups
            ),
            "unique_raw_mean": mean(len({x["raw_hash"] for x in group if x.get("raw_hash")}) for group in groups),
            "nonzero_reward_variance_groups": sum(std > 1e-8 for std in reward_stds),
            "reward_std_mean": mean(reward_stds) if reward_stds else 0.0,
            "nonzero_effective_reward_variance_groups": sum(std > 1e-8 for std in effective_stds),
            "effective_reward_std_mean": mean(effective_stds) if effective_stds else 0.0,
        })

    all_samples = [sample for groups in aggregates.values() for group in groups for sample in group]
    write(out / "summary.json", {
        "fallback": False,
        "groups": report,
        "role": role,
        "model": env.policy(role).client.model,
        "temperature": temperature,
        "skipped_empty_test_states": empty_test,
        "performance": {
            "generation_attempts": len(all_samples),
            "generated_samples": sum(s.get("raw_text") is not None for s in all_samples),
            "scoring_attempts": sum(s.get("raw_text") is not None and not s["score_cache_hit"]
                                    for s in all_samples),
            "scored_samples": sum(s.get("status") == "ok" and not s["score_cache_hit"]
                                  for s in all_samples),
            "score_cache_hits": sum(s["score_cache_hit"] for s in all_samples),
            "generation_seconds": sum(s["generation_seconds"] for s in all_samples),
            "scoring_seconds": sum(s["scoring_seconds"] for s in all_samples),
        },
        "metric_notes": {
            "all_sample_rates": "Observed passes divided by all attempts; unknowns remain null in samples and are counted separately.",
            "conditional_rates": "Answer, edit quality, retention, compression and noop rates use only evaluated samples.",
            "family_constraint": "Compatibility alias for generic edit_quality_pass; family is used only for grouping.",
            "compression_success": "Refine only: safe positive shortening. A valid NOOP can succeed with zero reward and no compression.",
            "score_cache": "Exact raw repeats within one state reuse a completed score; every sample still generates independently. Unknown scores are not cached.",
            "timing": "Wall seconds spent generating and scoring, including cache access and failed attempts; not pure model inference latency.",
        },
    })
    return report
