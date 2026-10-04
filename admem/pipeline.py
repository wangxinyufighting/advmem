"""Build → Audit/Patch → Refine。纯模型提议与外部回退分开记录。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from .common import InvalidAction, Unknown, digest, parse, read, write, write_rows
from .data import hint, policy_type
from .audit_prompts import feasible_types
from .prompts import attacker_messages, builder_prompt, evidence_usage, source_view
from .store import apply, augment, neutral_round, question_key, raw_chunks, text_size


def make_pool(full, env):
    idx = env.index(full.documents())
    sampler = env.pack_module.Sampler(full, idx, seed=env.cfg.seed, neighbors=env.cfg.neighbors,
                                       max_chars=env.cfg.pack_chars)
    return sampler.sample(len(sampler.seeds))


def windows(full, counter, limit):
    """完整遍历原文；长round按span切窗口但不删字符，rid仍指向同一原文。"""
    current, size = [], 0
    for rid in full.ordered(full.rounds):
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


def related(memory, x, env):
    if not memory:
        return []
    source_ids = {s["rid"] for s in x}
    overlap = [e["id"] for e in memory if set(e["prov"]) & source_ids]
    idx = env.memory_index(memory)
    found = idx.search_many([s["text"] for s in x], env.cfg.old_entries) if x else []
    return list(dict.fromkeys(overlap + [h.id for h in found]))[:env.cfg.old_entries]


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
    visible_type = context.get("question_type") if env.cfg.hint_mode == "target_type" else None
    extra_system = ""
    if visible_type == "temporal-reasoning":
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
    prompt, visible, cap = builder_prompt(
        full, memory, ids, x, mode, hint(context, env.cfg), env.cfg, env.counter,
        extra_system=extra_system,
    )
    tests = list({question_key(q): clean_question(q) for q in tests}.values())
    return {"role": "builder", "mode": mode, "split": context["split"], "case_key": context["key"],
            "full_path": str(Path(full_path).resolve()), "full_hash": full.fingerprint,
            "hint_mode": env.cfg.hint_mode, "environment_fingerprint": env.cfg.fingerprint(), "M": deepcopy(memory), "old_ids": visible,
            "x": deepcopy(x), "entry_limit": cap, "tests": tests, "prompt": prompt}


def attacker_state(full, full_path, context, memory, pack, accepted, env, identity):
    """题型对该pack不可出题时返回None：调用方记为type_infeasible，不调用attacker、不产生奖励。"""
    allowed = feasible_types(full, pack, env.cfg.seed_min_personal) if env.cfg.attacker_type_filter else None
    qtype = policy_type(context, env.cfg, identity, allowed)
    if qtype is None:
        return None
    prompt, visible = attacker_messages(full, pack, qtype, context["question_date"], memory, accepted,
                                        env.cfg, env.counter)
    # gate只接受attacker实际看到的round作E；compact视图下隐去的邻居不能被引用。
    pack = type(pack)(**{**pack.to_dict(), "rids": visible})
    return {"role": "attacker", "split": context["split"], "case_key": context["key"],
            "full_path": str(Path(full_path).resolve()), "full_hash": full.fingerprint,
            "hint_mode": env.cfg.hint_mode, "environment_fingerprint": env.cfg.fingerprint(), "M": deepcopy(memory), "pack": pack.to_dict(),
            "date": context["question_date"], "qtype": qtype, "prompt": prompt,
            "seen_keys": [question_key(q) for q in accepted], "evidence_usage": evidence_usage(accepted)}


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


def edit(full, state, raw, env):
    try:
        proposal, changed = apply(state["M"], parse(raw), mode=state["mode"], visible_ids=state["old_ids"],
            source_ids=[s["rid"] for s in state["x"]], full_ids=full.rounds,
            counter=env.counter, limit=state["entry_limit"], max_ops=env.cfg.max_ops)
    except InvalidAction as exc:
        return None, {"legal": False, "reason": str(exc)}
    faith = env.faith(full, changed)
    return (proposal if faith["faithful"] else None), {"legal": True, "faith": faith}


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


def all_pass(memory, questions, env):
    return all(env.answer(memory, q)["correct"] for q in questions)


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
            visible = set(s["rid"] for s in x)
            ids = related(memory, x, env)
            old = {r for e in memory if e["id"] in ids for r in e["prov"]}
            tests = [q for q in bank or [] if set(q["E"]) <= visited and set(q["E"]) <= visible | old
                     and set(q["E"]) & visible]
            tests = tests[:env.cfg.replay_questions]
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
                useful_demo = proposal is not None and (all_pass(proposal, tests, env) if tests else bool(parse(raw)["ops"]))
                collect.save(state, raw if useful_demo else None,
                             {"faith": proposal is not None, "tests_verified": bool(tests) and useful_demo, "semantic_imitation": True})
            checkpoint({"stage": "build", "window": wi, "checks": checks, "fallback": fallback,
                        "M_tokens": text_size(memory, env.counter)})
        write(out / "M_build.json", memory)
        if mode == "build":
            write(out / "M_final.json", memory)
            write(out / "Q.json", qbook)
            return {"status": "ok", "M_tokens": text_size(memory, env.counter), "Q_size": 0,
                    "fallback_count": sum(e.get("fallback", False) for e in events)}
        pool = make_pool(full, env)
        ledger = []
        for sweep in range(env.cfg.sweeps):
            before_cycle = digest(memory)
            for pi, pack in enumerate(pool):
                identity = [namespace, sweep, pi]
                state = attacker_state(full, full_path, context, memory, pack, accepted, env, identity)
                if state is None:
                    evaluation = {"legal": True, "items": [], "skipped": "type_infeasible"}
                else:
                    raw = propose(state, env, attacker_role, f"{namespace}:audit:{sweep}:{pi}")
                    evaluation = env.attacker_score(full, state, raw)
                if collect and state is not None:
                    # teacher出题SFT只收本来就有效的完整输出，不能把gate修订答案写成policy输出。
                    all_valid = evaluation.get("items") and all(r["status"] == "accepted" for r in evaluation["items"])
                    collect.save(state, raw if all_valid else None, {"gate": bool(all_valid)})
                ledger.append({"seed": pack.seed_id, "sweep": sweep, "M_hash": digest(state["M"]),
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


def generate_bank(full, full_path, context, env, out, role="ATTACKER"):
    out = Path(out)
    accepted, logs = [], []
    pool = make_pool(full, env)
    for pi, pack in enumerate(pool):
        state = attacker_state(full, full_path, context, [], pack, accepted, env, ["bank", pi])
        if state is None:
            logs.append({"pack": pi, "status": "type_infeasible"})
            continue
        pack = env.pack_module.Pack(**state["pack"])
        raw = propose(state, env, role, "bank:" + digest([state["prompt"], env.policy(role).tag]))
        try:
            data = parse(raw)
            if set(data) != {"items"} or not isinstance(data["items"], list) or len(data["items"]) > env.cfg.questions_per_pack:
                raise InvalidAction("Invalid items")
        except InvalidAction as exc:
            logs.append({"pack": pi, "status": "invalid", "reason": str(exc)})
            continue
        for q in data["items"]:
            result = env.gate(full, pack, q, state["date"], state["qtype"])
            logs.append({"pack": pi, **result})
            if result["status"] == "accepted" and question_key(q) not in {question_key(x) for x in accepted}:
                accepted.append(q)
        write(out / "bank.json", accepted)
        write(out / "bank_log.json", logs)
    write(out / "bank.json", accepted)
    write(out / "bank_log.json", logs)
    return len(accepted)
