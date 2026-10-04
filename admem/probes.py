"""Builder基础能力诊断：独立合成情境，不使用LongMemEval官方问题作为训练奖励。"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev

from .common import Progress, Unknown, digest, read, write, write_rows, parse
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


def _family_constraint(result, raw, state, family):
    """只检查 probe 的额外约束；未执行的阶段返回 None。"""
    if "family_constraint_pass" in result:
        return result.get("family_constraint_pass"), result.get("family_constraint_reason")
    if not result.get("legal") or not result.get("faith"):
        return None, None

    mode = state.get("mode")
    if mode == "refine":
        before = result.get("before_tokens")
        after = result.get("after_tokens")
        if before is None or after is None:
            return None, "missing token counts"
        ok = after < before
        return ok, None if ok else "memory was not shortened"

    if family == "noop_repeat":
        try:
            ok = parse(raw).get("ops") == []
        except Exception:
            return False, "invalid proposal JSON"
        return ok, None if ok else "NOOP must contain an empty ops list"

    return True, None


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
                mem = [{"id": "m1", "text": f"In January the user trained Monday and Friday for the {name} program.", "prov": ["s1:r1"], "kind": "card"}]
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


def run_attacker_probe(contexts_iter, env, out, samples=4, role="ATTACKER", max_packs=None,
                       temperature=0.7):
    """诊断 attacker reward 是否有组内学习信号（GRPO 要求每个 prompt 内 reward 有方差）。

    对每个 pack 构造一个 attacker state（honor ``cfg.attacker_type_filter`` 等开关），
    采样 ``samples`` 条 completion，用 ``env.attacker_score`` 打分；报告组内 std、
    非零方差组占比、reward 分布。reward 全部相同 = 无梯度信号。
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples_path, raw_path = out / "samples.jsonl", out / "raw_samples.jsonl"
    for path in (samples_path, raw_path):
        if path.exists():
            path.unlink()
    records = []
    for context, full_path in contexts_iter:
        full = env.memory_module.FullMemory.load(full_path)
        pool = make_pool(full, env)
        if max_packs:
            pool = pool[:max_packs]
        bar = Progress(len(pool), label=f"{context['key']} attacker-probe")
        for pi, pack in enumerate(pool):
            state = attacker_state(full, full_path, context, [], pack, [], env, ["probe", pi])
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
                except Unknown as exc:
                    write(out / "interrupted.json", {"case": context["key"], "pack": pi,
                                                     "sample": i, "error": str(exc)})
                    raise
                record = {
                    "case_key": context["key"], "pack": pi, "sample_idx": i,
                    "qtype": state["qtype"], "temperature": temperature, "nonce": nonce,
                    "raw_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16],
                    "reward": scored.get("reward"),
                    "effective_reward": scored.get("effective_reward"),
                    "legal": scored.get("legal"), "reason": scored.get("reason"),
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
        per_group.append({
            "case_key": case_key, "pack": pack, "qtype": group[0].get("qtype"),
            "samples": len(rewards), "mean": mean(rewards), "std": pstdev(rewards),
            "min": min(rewards), "max": max(rewards),
            "unique_rewards": len(set(rewards)), "unique_raw": len({r["raw_hash"] for r in group}),
        })
    all_rewards = [float(r["reward"]) for r in records if r.get("reward") is not None]
    signal = [g for g in per_group if g["std"] > 1e-8]
    summary = {
        "role": role, "model": env.policy(role).client.model, "temperature": temperature,
        "groups": len(per_group), "samples_per_group": samples,
        "groups_with_signal": len(signal),
        "signal_group_rate": len(signal) / len(per_group) if per_group else None,
        "reward_std_mean": mean([g["std"] for g in per_group]) if per_group else None,
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


def run_probe(records, env, out, samples=8, role="BUILDER", limit=None):
    """运行 Builder probe，并保存逐样本、可审计的评估记录。"""
    from .common import rows

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = list(rows(records))
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("No probe states")

    # 每次 probe 使用新的输出目录；若目录已存在，避免旧样本混入本次汇总。
    samples_path = out / "samples.jsonl"
    raw_path = out / "raw_samples.jsonl"
    for path in (samples_path, raw_path):
        if path.exists():
            path.unlink()

    aggregates = defaultdict(list)
    temperature = 0.7

    for row in records:
        full = env.memory_module.FullMemory.load(row["state"]["full_path"])
        values = []
        family = row.get("family", row["state"]["mode"])

        for i in range(samples):
            # 禁用逐字回退；所有结果完全来自该模型本次提议。
            nonce = "probe:" + digest([row["id"], i, env.policy(role).tag])
            raw = env.policy(role).complete(
                row["prompt"], nonce, temperature=temperature
            )
            raw_text = _json_text(raw)
            raw_hash = hashlib.sha256(
                raw_text.encode("utf-8")
            ).hexdigest()[:16]

            try:
                result = dict(env.builder_score(full, row["state"], raw, family=family))
                result.pop("proposed", None)

                legal = result.get("legal")
                faith = result.get("faith")
                answer_correct = _answer_correct(result)
                family_ok, family_reason = _family_constraint(
                    result, raw, row["state"], family
                )
                success = bool(
                    legal
                    and faith
                    and answer_correct is True
                    and family_ok is True
                )

                result.update({
                    "answer_correct": answer_correct,
                    "family_constraint_pass": family_ok,
                    "family_constraint_reason": family_reason,
                    "success": success,
                    "status": "ok",
                })
            except Unknown as exc:
                # Unknown 是环境/Judge 未知，不应伪装成模型负样本。
                write(
                    out / "interrupted.json",
                    {"case": row["id"], "sample": i, "error": str(exc)},
                )
                raise

            record = {
                "state_id": row["id"],
                "family": family,
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
            }
            # 保留 accuracy、token 数、答案检查等已有诊断字段。
            for key in ("accuracy", "before_tokens", "after_tokens", "answer_checks"):
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

        aggregates[family].append(values)

    def rate(values, key):
        checked = [x[key] for x in values if x.get(key) is not None]
        return mean(float(bool(x)) for x in checked) if checked else None

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
            "legal_rate": rate(values, "legal"),
            "faith_rate_all_samples": rate(values, "faith"),
            "answer_correct_rate": rate(values, "answer_correct"),
            "family_constraint_rate": rate(values, "family_constraint_pass"),
            "success_rate_all_samples": rate(values, "success"),
            "first_sample_success_rate": mean(float(group[0]["success"]) for group in groups),
            "empirical_at_least_one_success": mean(
                float(any(x["success"] for x in group)) for group in groups
            ),
            "unique_raw_mean": mean(len({x["raw_hash"] for x in group}) for group in groups),
            "nonzero_reward_variance_groups": sum(std > 1e-8 for std in reward_stds),
            "reward_std_mean": mean(reward_stds) if reward_stds else 0.0,
            "nonzero_effective_reward_variance_groups": sum(std > 1e-8 for std in effective_stds),
            "effective_reward_std_mean": mean(effective_stds) if effective_stds else 0.0,
        })

    write(out / "summary.json", {
        "fallback": False,
        "groups": report,
        "role": role,
        "model": env.policy(role).client.model,
        "temperature": temperature,
    })
    return report
