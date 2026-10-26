"""Build → Audit/Patch → Refine。纯模型提议与外部回退分开记录。"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .common import BudgetExceeded, InvalidAction, Progress, Unknown, digest, parse, read, write, write_rows
from .data import TypeScheduler, hint, policy_type, resolve_type_policy
from .audit_prompts import TYPES, attacker_rids, feasible_types, plan_item_type, structural_types
from .prompts import attacker_messages, attacker_view_type, builder_prompt, evidence_usage, source_view
from .store import apply, augment, neutral_round, question_key, raw_chunks, round_dates, text_size

# defect类别里，能靠把E原文补进M修复的那几类；none/unresolved不算当前M的缺陷。
REPAIRABLE_DEFECTS = {"missing", "lossy_or_reasoning", "unretrievable", "retrieval_composition_candidate"}


def make_pool(full, env, report=None):
    """attacker的pack池。report（可选dict）回填被切分/跳过的session与round，供pack_pool.json记账。

    pack_builder=legacy：旧Sampler（字符预算，超长种子session整段跳过——现在会记录下来）。
    pack_builder=token：admem.packing.token_pool（token预算、切分长session、事实句邻居）。"""
    if getattr(env.cfg, "pack_builder", "legacy") == "token":
        from .packing import token_pool
        pool, details = token_pool(full, env)
    else:
        pool, details = _legacy_pool(full, env)
    if report is not None:
        report.update(details)
    return pool


def _legacy_pool(full, env):
    idx = env.index(full.documents())
    sampler = env.pack_module.Sampler(full, idx, seed=env.cfg.seed, neighbors=env.cfg.neighbors,
                                       max_chars=env.cfg.pack_chars)
    # 跳过 seed 原文超出 pack 预算的 session（不截断原文）；测试替身无 fit_seeds 时保持原行为。
    if hasattr(sampler, "fit_seeds"):
        sampler.fit_seeds()
    skipped = []
    for sid, row in (getattr(sampler, "ledger", None) or {}).items():
        if isinstance(row, dict) and row.get("status") == "seed_over_budget" and sid in full.sessions:
            rids = list(full.sessions[sid].rids)
            # 只记中性sN，不记原始session ID（可能是answer_*标签）。
            skipped.append({"session": rids[0].split(":")[0] if rids else None, "rounds": len(rids),
                            "reason": "seed_over_char_budget", "pack_chars": env.cfg.pack_chars})
    return sampler.sample(len(sampler.seeds)), {"skipped_sessions": skipped}


def pool_summary(pool, report, cfg):
    """pack池的可审计摘要：只含中性rid/sN与事实句查询，不含原始session ID。"""
    packs = []
    for i, pack in enumerate(pool):
        seed = list(pack.seed_rids)
        packs.append({"pack": i, "seed": seed[0].split(":")[0] if seed else None, "kind": pack.kind,
                      "seed_rids": seed, "other_rids": [r for r in pack.rids if r not in set(seed)],
                      "fact_queries": list(getattr(pack, "searches", None) or [])})
    summary = {"schema": 1, "pack_builder": getattr(cfg, "pack_builder", "legacy"),
               "neighbor_mode": getattr(cfg, "neighbor_mode", "topical"), "neighbors": cfg.neighbors,
               "packs": packs, "skipped_sessions": report.get("skipped_sessions", []),
               "skipped_rounds": report.get("skipped_rounds", []),
               "split_sessions": report.get("split_sessions", [])}
    summary.update({k: report[k] for k in ("seed_token_limit", "attacker_input_tokens") if k in report})
    return summary


def windows(full, counter, limit):
    """按 session 对齐切窗口：每个窗口只含一个 session 的 round，不跨 session。

    session 内按 round 打包到 token 预算；只有单个 round 超过预算才按 token
    切 span。所有原文字符都保留，不删不丢。"""
    for sid in full.chronological_ids:
        session = full.sessions[sid]
        current, size = [], 0
        for rid in session.rids:
            source = neutral_round(full, rid)
            for start, end, text in counter.split(source["text"], limit):
                chunk = {**source, "text": text, "span": [start, end]}
                n = counter.count(text)
                if current and size + n > limit:
                    yield current
                    current, size = [], 0
                current.append(chunk)
                size += n
        if current:
            yield current


def related(memory, x, env, limit=None):
    limit = env.cfg.old_entries if limit is None else limit
    if not memory:
        return []
    source_ids = {s["rid"] for s in x}
    overlap = [e["id"] for e in memory if set(e["prov"]) & source_ids]
    idx = env.memory_index(memory)
    found = idx.search_many([s["text"] for s in x], limit) if x else []
    return list(dict.fromkeys(overlap + [h.id for h in found]))[:limit]


def state_tests(tests):
    """测试题按question_key去重；保留v2的context_rids（奖励时额外补进这些round的原文）。"""
    kept = {}
    for q in tests:
        item = clean_question(q)
        if q.get("context_rids"):
            item["context_rids"] = list(q["context_rids"])
        kept[question_key(q)] = item
    return list(kept.values())


def stream_tests(bank, x, visited, old, env):
    """stream窗口的奖励测试题。

    v1：E已读完，且全部落在当前窗口或待编辑旧条目的prov里（跨session题只在最后一块证据出现、
    且之前的证据已被存进M时才会成为测试题）。
    v2（builder_reward=v2）另加“前瞻”题：E与本窗口已读完的round有交集、但不满足上面条件的题。
    评分时把E中不在本窗口的round原文补进编辑前后的记忆（context_rids），只衡量本窗口那部分
    证据有没有被存下——之前漏存、或证据还在未来session里的跨session题也能给本窗口记分。"""
    visible = {s["rid"] for s in x}
    complete = visible & visited  # 调用前visited已加入本窗口读完的round
    plain = [q for q in bank or [] if set(q["E"]) <= visited and set(q["E"]) <= visible | old]
    tests = sorted(plain, key=lambda q: bool(set(q["E"]) & visible), reverse=True)
    if env.cfg.builder_reward == "v2":
        keys = {question_key(q) for q in plain}
        ahead = [{**q, "context_rids": sorted(set(q["E"]) - visible)} for q in bank or []
                 if question_key(q) not in keys and set(q["E"]) & complete]
        overlapping = [q for q in tests if set(q["E"]) & visible]
        tests = overlapping + ahead + [q for q in tests if not set(q["E"]) & visible]
    return tests[:env.cfg.replay_questions]


def replay(qbook, n, identity):
    # 一半高错误频次，一半确定性随机；已解决题仍保留，防止遗忘。
    ranked = sorted(qbook, key=lambda q: (-q.get("fails", 0), question_key(q)))
    priority = ranked[:n // 2]
    rest = sorted(ranked[n // 2:], key=lambda q: digest([identity, question_key(q)]))
    return priority + rest[:max(0, n - len(priority))]


def clean_question(q):
    return {k: deepcopy(q[k]) for k in ["q", "a", "type", "question_date", "E"]}


def builder_state(full, full_path, context, memory, x, mode, tests, env, old_ids=None):
    ids = related(memory, x, env) if old_ids is None else old_ids
    v2_prompt = env.cfg.builder_prompt == "v2"
    visible_type = context.get("question_type") if env.cfg.hint_mode == "target_type" else None
    extra_system = ""
    if visible_type == "temporal-reasoning" and not v2_prompt:  # v2提示词的TIME一节对所有题型生效
        # Put temporal arithmetic in the single system message and count the
        # addition before selecting the old-entry subset. Small instruction
        # models often copy the report date when this rule is appended later.
        extra_system = (
            "TEMPORAL MEMORY CHECK (must follow before emitting JSON):\n"
            "1. A source object's date is the report/reference date, not automatically the event date.\n"
            "2. If a source says \"Yesterday I bought X\", the event date is one calendar day "
            "before that source's report date.\n"
            "3. Preserve the relative wording alongside the resolved event date when useful.\n"
            "4. Never use the report date as X's event date in that case.\n"
            "5. Cite only a provenance rid copied verbatim from the current payload's allowed_source_rids.\n"
            "6. Do not add first/only/no-other-event claims.\n"
            "7. Output exactly one JSON object; do not output a worked example, Markdown, or a thinking block."
        )
    # v2：目录先列与x相关的条目（不止old_entries个），其余最新优先。
    order = related(memory, x, env, limit=len(memory)) if v2_prompt and x else []
    prompt, visible, cap = builder_prompt(
        full, memory, ids, x, mode, hint(context, env.cfg), env.cfg, env.counter,
        extra_system=extra_system, catalog_order=order,
    )
    tests = state_tests(tests)
    state = {"role": "builder", "mode": mode, "split": context["split"], "case_key": context["key"],
             "full_path": str(Path(full_path).resolve()), "full_hash": full.fingerprint,
             "hint_mode": env.cfg.hint_mode, "environment_fingerprint": env.cfg.fingerprint(), "M": deepcopy(memory), "old_ids": visible,
             "x": deepcopy(x), "entry_limit": cap, "tests": tests, "prompt": prompt}
    if env.cfg.builder_reward == "v2":
        # 冗余只对照Builder看得到的条目（M_old全文+目录摘要）；有x时没有测试题也能按覆盖率评分。
        payload = json.loads(prompt[-1]["content"])
        state["catalog_ids"] = [item["id"] for item in payload.get("stored_catalog", [])]
        state["reward_version"] = "v2"
        state["coverage_scorable"] = bool(x) and mode in {"stream", "patch"} and env.cfg.builder_coverage_weight > 0
    return state


def _sessions(full, rids):
    return {full.rounds[r].session_id for r in rids}


def attacker_feasible(full, pack, cfg):
    """当前题型策略下该pack可出的题型（type_infeasible日志用）。"""
    if resolve_type_policy(cfg) == "balanced":
        feasible = structural_types(full, pack, cfg.attacker_view)
        if cfg.attacker_type_filter:
            allowed = set(feasible_types(full, pack, cfg.seed_min_personal))
            feasible = [t for t in feasible if t in allowed]
        return feasible
    return feasible_types(full, pack, cfg.seed_min_personal)


def attacker_state(full, full_path, context, memory, pack, accepted, env, identity, scheduler=None):
    """题型对该pack不可出题时返回None：调用方记为type_infeasible，不调用attacker、不产生奖励。

    official/uniform为单题型（旧行为）；balanced由TypeScheduler按结构可行性分配多题型计划，
    不读取context里的官方题型。prompt只剩种子仍超出attacker预算时抛BudgetExceeded。"""
    cfg = env.cfg
    policy = resolve_type_policy(cfg)
    plan = None
    if policy == "balanced":
        plan = (scheduler or TypeScheduler(cfg)).plan(attacker_feasible(full, pack, cfg), identity)
        if not plan:
            return None
        qtype = next(iter(plan))
        plan = plan if len(plan) > 1 else None  # 单题型计划沿用旧版prompt与校验
    else:
        allowed = feasible_types(full, pack, cfg.seed_min_personal) if cfg.attacker_type_filter else None
        qtype = policy_type(context, cfg, identity, allowed)
        if qtype is None:
            return None
        # compact 视图可能去掉只有 assistant 的邻居；跨会话输入须有两个实际可见会话。
        if qtype == "multi-session" and len(_sessions(full, attacker_rids(full, pack, qtype, cfg.attacker_view))) < 2:
            return None
    shown = attacker_rids(full, pack, attacker_view_type(qtype, plan), cfg.attacker_view)
    prompt, visible = attacker_messages(full, pack, qtype, context["question_date"], memory, accepted,
                                        cfg, env.counter, type_plan=plan)
    needs_two = "multi-session" in (plan or {qtype: 1})
    if needs_two and len(_sessions(full, visible)) < 2:
        if not plan:
            raise BudgetExceeded("Attacker pack exceeds token budget: the multi-session pack keeps fewer "
                                 "than two sessions", stage="attacker_prompt", limit=cfg.attacker_input_tokens)
        # 预算裁掉了全部邻居：去掉multi-session槽位，按剩下的题型重新渲染。
        plan = {t: n for t, n in plan.items() if t != "multi-session"}
        qtype = next(iter(plan))
        plan = plan if len(plan) > 1 else None
        shown = attacker_rids(full, pack, attacker_view_type(qtype, plan), cfg.attacker_view)
        prompt, visible = attacker_messages(full, pack, qtype, context["question_date"], memory, accepted,
                                            cfg, env.counter, type_plan=plan)
    # gate只接受attacker实际看到的round作E；compact视图下隐去的邻居、被预算裁掉的邻居都不能被引用。
    budget_dropped = [r for r in shown if r not in set(visible)]
    pack = type(pack)(**{**pack.to_dict(), "rids": visible})
    state = {"role": "attacker", "split": context["split"], "case_key": context["key"],
             "full_path": str(Path(full_path).resolve()), "full_hash": full.fingerprint,
             "hint_mode": cfg.hint_mode, "environment_fingerprint": cfg.fingerprint(), "M": deepcopy(memory), "pack": pack.to_dict(),
             "date": context["question_date"], "qtype": qtype, "prompt": prompt,
             "seen_keys": [question_key(q) for q in accepted], "evidence_usage": evidence_usage(accepted),
             "type_policy": policy}
    if plan:
        state["type_plan"] = dict(plan)
    if cfg.attacker_reward == "v2":
        # v2奖励的语义去重需要已接受题的题面（只取与可见round有交集的最近若干条）。
        overlap = [{"q": q["q"], "E": list(q["E"])} for q in accepted if set(q["E"]) & set(visible)]
        state["prior_questions"] = overlap[-max(1, cfg.attacker_context_items):]
    if budget_dropped:
        state["budget_dropped_rids"] = budget_dropped
    return state


class Collector:
    def __init__(self, out):
        self.out = Path(out)

    def save(self, state, demo=None, quality=None):
        # reward metadata仅由环境读取；导出的actor prompt里没有tests。
        key = digest({k: v for k, v in state.items() if k != "full_path"})
        record = {"id": key, "state": state, "prompt": state["prompt"]}
        if demo is not None:
            record.update(completion=demo, demo_quality=quality)
        folder = self.out / state["split"] / state["role"]
        write(folder / (key + ".json"), record)
        return key


def propose(state, env, role, nonce):
    temperature = 0.7 if state["role"] == "attacker" else 0.0
    return env.policy(role).complete(state["prompt"], nonce, temperature=temperature)


def apply_options(full, cfg):
    """builder_prompt=v2：卡片记来源日期，第一个窗口允许NOOP。v1：与旧实现相同。"""
    if cfg.builder_prompt == "v2":
        return {"round_dates": round_dates(full), "forbid_initial_noop": False}
    return {}


def edit(full, state, raw, env):
    try:
        proposal, changed = apply(state["M"], parse(raw), mode=state["mode"], visible_ids=state["old_ids"],
            source_ids=[s["rid"] for s in state["x"]], full_ids=full.rounds,
            counter=env.counter, limit=state["entry_limit"], max_ops=env.cfg.max_ops,
            **apply_options(full, env.cfg))
    except InvalidAction as exc:
        return None, {"legal": False, "reason": str(exc)}
    if env.cfg.faith_gate:
        faith = env.faith(full, changed)
        return (proposal if faith["faithful"] else None), {"legal": True, "faith": faith, "faith_gated": True}
    # 诊断模式：不调用faith judge，直接应用合法edit（非法schema仍拒绝→fallback）。
    return proposal, {"legal": True, "faith": None, "faith_gated": False}


def raw_from_window(x, env):
    items = []
    for s in x:
        header = f"Recorded on {s['date']}. "
        limit = env.cfg.timeline_tokens - env.counter.count(header) - 8
        for a, b, text in env.counter.split(s["text"], limit):
            span = [s.get("span", [0])[0] + a, s.get("span", [0])[0] + b]
            text = header + text
            items.append({"id": "raw_" + digest([s["rid"], span, text])[:20], "text": text,
                          "prov": [s["rid"]], "kind": "raw", "span": span})
    return items


def with_context(memory, q, full, env):
    """v2前瞻测试题：把E中不在当前窗口的round原文补进记忆再答。"""
    rids = q.get("context_rids") or []
    if not rids:
        return memory
    return augment(memory, raw_chunks(full, rids, env.counter, env.cfg.timeline_tokens))


def all_pass(memory, questions, env, full=None):
    if full is None and any(q.get("context_rids") for q in questions):
        raise ValueError("context_rids tests need the full memory to supply context")
    return all(env.answer(with_context(memory, q, full, env), q)["correct"] for q in questions)


def run_case(full, full_path, context, env, out, *, mode="closed_loop", bank=None,
             builder_role="BUILDER", attacker_role="ATTACKER", collect=None):
    out = Path(out)
    # 保存策略服务身份和所有输入：更换checkpoint必须换模型名/out，避免缓存混用。
    signature = {"full": full.fingerprint, "context": context, "config": env.cfg.fingerprint(), "mode": mode,
                 "builder": env.policy(builder_role).tag,
                 "attacker": env.policy(attacker_role).tag if mode != "build" else None,
                 "bank_hash": digest(bank or []), "defender": env.policy("DEFENDER").tag, "judge": env.policy("JUDGE").tag}
    cfg_path = out / "run_config.json"
    if cfg_path.exists() and read(cfg_path) != signature:
        raise ValueError("Run configuration differs; use a new output directory")
    write(cfg_path, signature)
    memory, qbook, accepted, events = [], [], [], []
    namespace = digest(signature)
    visited = set()
    def checkpoint(event):
        events.append(event)
        write(out / "checkpoint.json", {"M": memory, "Q": qbook, "accepted": accepted, "events": events,
              "note": "Resume replays deterministic cached requests; never manually edit this snapshot."})
    try:
        for wi, x in enumerate(windows(full, env.counter, env.cfg.window_tokens)):
            # long round尚未读完时，不能把整条rid的未来事实当成本窗口可答测试。
            visited.update(s["rid"] for s in x if s["span"][1] == len(neutral_round(full, s["rid"])["text"]))
            ids = related(memory, x, env)
            old = {r for e in memory if e["id"] in ids for r in e["prov"]}
            # Δ可答性测试：证据已读全、落在当前窗口或待编辑条目prov的题；新题优先，历史题防遗忘。
            # builder_reward=v2时另加跨session的前瞻题（见stream_tests）。
            tests = stream_tests(bank, x, visited, old, env)
            state = builder_state(full, full_path, context, memory, x, "stream", tests, env, ids)
            raw = propose(state, env, builder_role, f"{namespace}:build:{wi}")
            proposal, checks = edit(full, state, raw, env)
            fallback = False
            if proposal is not None:
                memory = proposal
            elif env.cfg.fallback:
                memory = augment(memory, raw_from_window(x, env))
                fallback = True
            if collect:
                # 不把逐字fallback当作模型的SFT目标。无tests只用于faith+格式示范。
                useful_demo = proposal is not None and (all_pass(proposal, state["tests"], env, full) if tests
                                                        else bool(parse(raw)["ops"]))
                collect.save(state, raw if useful_demo else None,
                             {"faith": proposal is not None, "tests_verified": bool(tests) and useful_demo, "semantic_imitation": True})
            try:
                n_ops = len(parse(raw)["ops"])
            except (InvalidAction, KeyError, TypeError):
                n_ops = None  # 非法输出：checks.legal已记为False
            checkpoint({"stage": "build", "window": wi, "checks": checks, "fallback": fallback, "ops": n_ops,
                        "M_tokens": text_size(memory, env.counter)})
        write(out / "M_build.json", memory)
        if mode == "build":
            write(out / "M_final.json", memory)
            write(out / "Q.json", qbook)
            return {"status": "ok", "M_tokens": text_size(memory, env.counter), "Q_size": 0,
                    "fallback_count": sum(e.get("fallback", False) for e in events)}
        pool = make_pool(full, env)
        scheduler = TypeScheduler(env.cfg) if resolve_type_policy(env.cfg) == "balanced" else None
        ledger = []
        for sweep in range(env.cfg.sweeps):
            before_cycle = digest(memory)
            for pi, pack in enumerate(pool):
                identity = [namespace, sweep, pi]
                try:
                    state = attacker_state(full, full_path, context, memory, pack, accepted, env, identity, scheduler)
                except BudgetExceeded:
                    # 确定性预算溢出：跳过该pack，不让一个装不下的pack中止整个case。
                    state, evaluation = None, {"legal": True, "items": [], "skipped": "over_budget"}
                else:
                    evaluation = {"legal": True, "items": [], "skipped": "type_infeasible"}
                if state is not None:
                    raw = propose(state, env, attacker_role, f"{namespace}:audit:{sweep}:{pi}")
                    try:
                        evaluation = env.attacker_score(full, state, raw)
                    except BudgetExceeded:
                        evaluation = {"legal": True, "items": [], "skipped": "over_budget"}
                if collect and state is not None:
                    # teacher出题SFT只收本来就有效的完整输出，不能把gate修订答案写成policy输出。
                    all_valid = evaluation.get("items") and all(
                        r["status"] == "accepted" and not r.get("repaired") for r in evaluation["items"])
                    collect.save(state, raw if all_valid else None, {"gate": bool(all_valid)})
                ledger.append({"seed": pack.seed_rids[0].split(":")[0] if pack.seed_rids else None,
                               "sweep": sweep, "M_hash": digest(memory),
                               "status": evaluation.get("skipped") or ("asked" if evaluation.get("items") else
                                         "audited_empty" if evaluation.get("legal") else "invalid_output")})
                infos = list(evaluation.get("items", []))
                # 旧active问题不因为duplicate去重就永久失去修复机会；每轮最多重试一次。
                infos.extend({"status": "accepted", "item": clean_question(q)} for q in qbook
                             if not q.get("resolved") and q.get("last_retry_sweep", -1) < sweep
                             and set(q["E"]) <= set(pack.rids))
                for info in infos:
                    if info["status"] != "accepted":
                        continue
                    q = info["item"]
                    if question_key(q) not in {question_key(v) for v in accepted}:
                        accepted.append(q)
                    # 一包中的前一题可能已修改M；必须相对当前M重新检查defect。
                    defect = env.defect(full, memory, q)
                    if defect["kind"] in {"none", "unresolved"}:
                        checkpoint({"stage": "audit", "sweep": sweep, "pack": pi, "question": q,
                                    "defect": defect["kind"]})
                        continue
                    key = question_key(q)
                    existing = next((v for v in qbook if question_key(v) == key), None)
                    if existing:
                        existing["fails"] += 1
                        existing["last_retry_sweep"] = sweep
                    else:
                        qbook.append({**q, "fails": 1, "resolved": False, "last_retry_sweep": sweep})
                    protected = [v for v in qbook if v.get("resolved")]
                    tests = [q] + replay(qbook, env.cfg.replay_questions, identity)
                    x = source_view(full, q["E"])
                    bstate = builder_state(full, full_path, context, memory, x, "patch", tests, env)
                    proposal_raw = propose(bstate, env, builder_role, f"{namespace}:patch:{sweep}:{pi}:{key}")
                    proposal, checks = edit(full, bstate, proposal_raw, env)
                    patch_ok = proposal is not None and all_pass(proposal, protected + [q], env)
                    fallback = False
                    if patch_ok:
                        memory = proposal
                    elif env.cfg.fallback:
                        candidate = augment(memory, raw_chunks(full, q["E"], env.counter, env.cfg.timeline_tokens))
                        if all_pass(candidate, protected + [q], env):
                            memory, fallback = candidate, True
                    fixed = env.answer(memory, q)["correct"]
                    for entry in qbook:
                        if question_key(entry) == key:
                            entry["resolved"] = fixed
                    if collect:
                        collect.save(bstate, proposal_raw if patch_ok else None,
                                     {"faith": proposal is not None, "tests_verified": patch_ok})
                    checkpoint({"stage": "patch", "sweep": sweep, "pack": pi, "question": q,
                                "defect": defect["kind"], "checks": checks, "model_fixed": patch_ok,
                                "fallback": fallback, "fixed": fixed})
            write(out / f"M_audit_{sweep}.json", memory)
            # Q为空不允许vacuous全通过；有未解决题时先不压缩。
            if qbook and all(q.get("resolved") for q in qbook):
                seeds = [e["id"] for e in memory]
                for rr in range(env.cfg.refine_rounds):
                    for si, seed in enumerate(seeds):
                        lookup = {e["id"]: e for e in memory}
                        if seed not in lookup:
                            continue
                        idx = env.memory_index(memory)
                        ids = list(dict.fromkeys([seed] + [h.id for h in idx.search(lookup[seed]["text"], env.cfg.old_entries)]))[:env.cfg.old_entries]
                        bstate = builder_state(full, full_path, context, memory, [], "refine",
                                               replay(qbook, env.cfg.replay_questions, [sweep, rr, si]), env, ids)
                        proposal_raw = propose(bstate, env, builder_role, f"{namespace}:refine:{sweep}:{rr}:{si}")
                        proposal, checks = edit(full, bstate, proposal_raw, env)
                        shrink = proposal is not None and text_size(proposal, env.counter) < text_size(memory, env.counter)
                        keep = shrink and all_pass(proposal, qbook, env)
                        if keep:
                            memory = proposal
                        if collect:
                            collect.save(bstate, proposal_raw if keep else None,
                                         {"faith": proposal is not None, "all_Q_verified": bool(keep)})
                        checkpoint({"stage": "refine", "accepted": bool(keep), "checks": checks})
            write(out / f"M_refine_{sweep}.json", memory)
            write(out / "ledger.json", ledger)
            if digest(memory) == before_cycle:
                break
        write(out / "M_final.json", memory)
        write(out / "Q.json", qbook)
        return {"status": "ok", "M_tokens": text_size(memory, env.counter), "entries": len(memory),
                "Q_size": len(qbook), "Q_solved": sum(q.get("resolved", False) for q in qbook),
                "fallback_count": sum(e.get("fallback", False) for e in events),
                "model_patch_successes": sum(e.get("model_fixed", False) for e in events)}
    except Unknown:
        # 整个case保留为未知，不把API失败当作模型缺陷/负奖励。
        write(out / "interrupted.json", {"status": "unknown", "events_completed": len(events)})
        raise


def _bank_pack(full, full_path, context, pack, pi, accepted, env, role, trace_path=None, attempt=0,
               memory=None, scheduler=None, defects_only=False, collect=None):
    """Run one pack (attacker propose + gate). Raises Unknown on API/service failure.

    Pure with respect to the caller's ``accepted``/``logs`` so a failed pack can
    be retried without duplicating rows.  Returns ``(info, logs, new_items)``.

    预算溢出是确定性的，不抛给重试逻辑：pack级（只剩种子仍装不下）记一行over_budget并返回；
    题级（gate的judge装不下）只把该题记为over_budget，同包其余题照常判定。
    memory非空时attacker针对这份记忆出题（看得到provenance标记与memory_entries），通过gate的题
    记录defect类别；defects_only=True时不暴露缺陷的题记为accepted_no_defect、不进bank。
    """
    memory = list(memory or [])
    balanced = resolve_type_policy(env.cfg) == "balanced"
    # 日志只记中性sN，不记原始seed_id（可能是answer_*标签）。
    info = {"pack": pi, "seed": pack.seed_rids[0].split(":")[0] if pack.seed_rids else None,
            "case_question_type": context["question_type"],
            "qtype": None if balanced else hint(context, env.cfg),
            "pack_sessions": len({full.rounds[r].session_id for r in pack.rids})}
    if memory:
        info["memory_entries"] = len(memory)
    trace = None
    if trace_path is not None:
        trace_path = Path(trace_path)
        info["trace_file"] = str(Path("pack_traces") / trace_path.name)
        trace = {"schema": 1, "case_key": context["key"], "pack_index": pi, "attempt": attempt,
                 "full_hash": full.fingerprint, "environment_fingerprint": env.cfg.fingerprint(),
                 "seed_rids": list(pack.seed_rids), "pack_rids": list(pack.rids),
                 "dropped_rids": list(pack.dropped_rids or []), "visible_rids": None,
                 "prompt": None, "prompt_kind": "policy_prompt", "prompt_hash": None,
                 "request_body": None, "raw": None, "logs": [], "added_items": [],
                 "stage": "preparing", "error": None}
        if memory:
            trace["memory_hash"] = digest(memory)

    def snapshot(stage, **values):
        if trace is not None:
            trace.update(stage=stage, **values)
            write(trace_path, trace)

    snapshot("preparing")
    try:
        try:
            state = attacker_state(full, full_path, context, memory, pack, accepted, env, ["bank", pi], scheduler)
        except BudgetExceeded as exc:
            logs = [{**info, "status": "over_budget", **exc.details()}]
            snapshot("over_budget", logs=logs)
            return info, logs, []
        if state is None:
            logs = [{**info, "status": "type_infeasible", "reason": "requested_type_not_feasible",
                     "feasible_types": attacker_feasible(full, pack, env.cfg)}]
            snapshot("skipped", logs=logs)
            return info, logs, []
        pack = env.pack_module.Pack(**state["pack"])
        info.update(qtype=state["qtype"], prompt_tokens=env.counter.prompt_count(state["prompt"]),
                    visible_rounds=len(pack.rids),
                    visible_sessions=len({full.rounds[r].session_id for r in pack.rids}))
        if state.get("type_plan"):
            info["type_plan"] = dict(state["type_plan"])
        if state.get("budget_dropped_rids"):
            info["budget_dropped_rounds"] = len(state["budget_dropped_rids"])
        remote = env.policy(role)
        nonce = "bank:" + digest([state["prompt"], remote.tag])
        snapshot("prepared", prompt=state["prompt"], prompt_hash=digest(state["prompt"]),
                 visible_rids=list(pack.rids), qtype=state["qtype"], nonce=nonce,
                 policy_role=role, policy_tag=remote.tag)
        try:
            raw = propose(state, env, role, nonce)
        except Unknown:
            snapshot("request_failed", request_body=deepcopy(getattr(remote, "last_request", None)))
            raise
        snapshot("response", raw=raw, request_body=deepcopy(getattr(remote, "last_request", None)))
        try:
            data = parse(raw)
            if set(data) != {"items"} or not isinstance(data["items"], list) or len(data["items"]) > env.cfg.questions_per_pack:
                raise InvalidAction("Invalid items")
        except InvalidAction as exc:
            logs = [{**info, "status": "invalid", "reason": str(exc)}]
            snapshot("invalid", logs=logs)
            return info, logs, []
        logs = []
        if not data["items"]:
            logs.append({**info, "status": "empty"})
        new_items, seen = [], {question_key(x) for x in accepted}
        plan, used = state.get("type_plan"), Counter()
        for q in data["items"]:
            qtype = plan_item_type(state, q, used)
            try:
                if plan:
                    result = env.gate(full, pack, q, state["date"], qtype, seed_only=True)
                else:
                    result = env.gate(full, pack, q, state["date"], qtype)
            except BudgetExceeded as exc:
                logs.append({**info, "status": "over_budget", **exc.details(), "item": q})
                snapshot("gating", logs=logs, added_items=new_items)
                continue
            row = {**info, **result}
            if plan and isinstance(q, dict) and q.get("type") in TYPES:
                row["qtype"] = q["type"]
            if result["status"] == "accepted":
                checked = result["item"]
                if memory and env.cfg.bank_defects:
                    try:
                        row["defect"] = env.defect(full, memory, checked)["kind"]
                    except BudgetExceeded as exc:
                        row.update(defect="over_budget", defect_budget=exc.details())
                    if defects_only and row["defect"] not in REPAIRABLE_DEFECTS:
                        row["status"] = "accepted_no_defect"
                if row["status"] == "accepted" and question_key(checked) not in seen:
                    seen.add(question_key(checked))
                    new_items.append(checked)
            logs.append(row)
            snapshot("gating", logs=logs, added_items=new_items)
        snapshot("complete", logs=logs, added_items=new_items)
        if collect is not None:
            # attacker GRPO状态（与closed_loop相同的规则）：只有每题都原样通过gate时才保存为SFT示范。
            all_valid = bool(data["items"]) and all(
                row.get("status") == "accepted" and not row.get("repaired") for row in logs)
            collect.save(state, raw if all_valid else None, {"gate": all_valid})
        return info, logs, new_items
    except Unknown as exc:
        # Preserve response and completed judgments even if a later gate request
        # failed. Failed attempts never commit their items to the bank.
        snapshot("error", failed_stage=trace["stage"] if trace else None, error=str(exc))
        raise


def generate_bank(full, full_path, context, env, out, role="ATTACKER", *, memory=None, base_bank=None,
                  memory_source=None, defects_only=False, collect=None):
    """为一个case生成题库。

    memory：attacker针对的当前M（默认空M，即旧行为的冷启动题库）。base_bank：已有题库，作为
    accepted的初值参与去重、novelty计数与audit_context，输出的bank.json是合并后的结果。
    pack_pool.json记录pack构成与被切分/跳过的session、round；bank_manifest.json记录题型策略、
    记忆快照与基础题库的hash。"""
    out = Path(out)
    memory = list(memory or [])
    accepted, keys = [], set()
    for q in base_bank or []:
        if question_key(q) not in keys:
            keys.add(question_key(q))
            accepted.append(clean_question(q))
    base_size = len(accepted)
    logs = []
    pool_report = {}
    pool = make_pool(full, env, pool_report)
    policy = resolve_type_policy(env.cfg)
    scheduler = TypeScheduler(env.cfg) if policy == "balanced" else None
    write(out / "pack_pool.json", pool_summary(pool, pool_report, env.cfg))
    write(out / "bank_manifest.json", {
        "schema": 1, "type_policy": policy, "hint_mode": env.cfg.hint_mode,
        "label_informed_types": policy == "official",
        "pack_builder": getattr(env.cfg, "pack_builder", "legacy"),
        "neighbor_mode": getattr(env.cfg, "neighbor_mode", "topical"),
        "memory": {"entries": len(memory), "hash": digest(memory) if memory else None, "source": memory_source},
        "base_bank": {"questions": base_size, "hash": digest(accepted) if accepted else None},
        "defects_only": bool(defects_only), "environment_fingerprint": env.cfg.fingerprint()})
    if policy == "official":
        print(f"[bank] {context['key']}: type_policy=official uses the case's gold question type "
              "(diagnostic only, not a label-free result)", file=sys.stderr, flush=True)
    # <variant>/<case>：PARALLEL=1 多进程混排时能区分是哪条流水线。
    parent = Path(out).parent.name
    bar = Progress(len(pool), label=f"{parent}/{context['key']}")
    # 瞬时API失败只重试这条pack（attacker响应走缓存，重试便宜）；连续失败到上限才中止整个case。
    # 预算溢出是确定性的：不重试，不计入连续失败。
    retries = max(1, int(os.getenv("PACK_RETRIES", "2")))
    backoff = float(os.getenv("PACK_BACKOFF", "1.0"))
    fail_limit = max(1, int(os.getenv("PACK_FAIL_LIMIT", "5")))
    consecutive_errors = 0
    for pi, pack in enumerate(pool):
        info, pack_logs, new_items = {"pack": pi}, None, None
        for attempt in range(retries):
            trace_path = out / "pack_traces" / f"pack_{pi:05d}_attempt_{attempt:02d}.json"
            try:
                info, pack_logs, new_items = _bank_pack(
                    full, full_path, context, pack, pi, accepted, env, role,
                    trace_path=trace_path, attempt=attempt, memory=memory, scheduler=scheduler,
                    defects_only=defects_only, collect=collect)
                break
            except BudgetExceeded as exc:  # 兜底：_bank_pack之外的确定性溢出
                pack_logs, new_items = [{"pack": pi, "status": "over_budget", **exc.details(),
                                         "trace_file": str(trace_path.relative_to(out))}], []
                break
            except Unknown as exc:
                if attempt + 1 >= retries:
                    logs.append({"pack": pi, "status": "transport_error", "reason": str(exc),
                                 "trace_file": str(trace_path.relative_to(out))})
                    bar.update(suffix=f"p{pi} transport_error")
                    pack_logs = None
                    break
                time.sleep(backoff * 2 ** attempt)
        if pack_logs is None:
            consecutive_errors += 1
            write(out / "bank.json", accepted)
            write(out / "bank_log.json", logs)
            if consecutive_errors >= fail_limit:
                raise Unknown(f"连续 {consecutive_errors} 个 pack API失败，放弃case {context['key']}")
            continue
        consecutive_errors = 0
        logs.extend(pack_logs)
        accepted.extend(new_items)
        write(out / "bank.json", accepted)
        write(out / "bank_log.json", logs)
        status = "over_budget" if pack_logs and all(r.get("status") == "over_budget" for r in pack_logs) else None
        bar.update(suffix=f"p{pi} qtype={info.get('qtype')} bank={len(accepted)}" + (f" {status}" if status else ""))
    bar.close()
    write(out / "bank.json", accepted)
    write(out / "bank_log.json", logs)
    return len(accepted)
