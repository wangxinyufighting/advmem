"""从已构建M制造受控的缺失/碎片/重复状态，避免Builder早期奖励全零或只学NOOP。"""
from __future__ import annotations

from copy import deepcopy

from .common import digest
from .pipeline import builder_state, clean_question, edit, propose, replay
from .prompts import source_view
from .store import augment, raw_chunks


def bootstrap(full, full_path, context, memory, bank, env, collector, builder_role="BUILDER", limit=12):
    # 仅使用自生成并经gate验证的bank；绝不从private_eval读取问题。
    eligible = []
    for q in sorted(bank, key=lambda q: digest([env.cfg.seed, q])):
        if env.answer(memory, q)["correct"]:
            eligible.append(clean_question(q))
        if len(eligible) >= limit:
            break
    stats = {"eligible_questions": len(eligible), "states": 0, "verified_demonstrations": 0}
    for i, q in enumerate(eligible):
        linked = [e for e in memory if set(e["prov"]) & set(q["E"])]
        if not linked:
            continue
        remainder = [e for e in memory if e not in linked]
        # 缺失与碎片状态都有真实原文，不把伪造事实混入F。
        fragmented = augment(remainder, raw_chunks(full, q["E"], env.counter, env.cfg.timeline_tokens))
        doubled = deepcopy(memory)
        duplicate = deepcopy(linked[0])
        duplicate["id"] = "duplicate_" + digest([i, duplicate])[0:16]
        doubled.append(duplicate)
        variants = [("missing", remainder, "patch", source_view(full, q["E"]), None),
                    ("fragmented", fragmented, "patch", source_view(full, q["E"]), None),
                    ("duplicate", doubled, "refine", [], [linked[0]["id"], duplicate["id"]])]
        for name, m, mode, x, ids in variants:
            tests = [q] + replay(eligible, env.cfg.replay_questions, [i, name])
            state = builder_state(full, full_path, context, m, x, mode, tests, env, ids)
            raw = propose(state, env, builder_role, "bootstrap:" + digest([state, env.policy(builder_role).tag]))
            score = env.builder_score(full, state, raw)
            verified = bool(score.get("legal") and score.get("faith") and score.get("accuracy") == 1)
            if mode == "refine":
                verified = verified and score.get("after_tokens", 0) < score.get("before_tokens", 0)
            collector.save(state, raw if verified else None,
                           {"faith": score.get("faith", False), "tests_verified": verified, "mutation": name})
            stats["states"] += 1
            stats["verified_demonstrations"] += verified
    return stats
