"""冻结的检索/reader/oracle环境；训练奖励与部署使用同一编辑和读取路径。"""
from __future__ import annotations

import re
from collections import Counter, OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from .common import BudgetExceeded, Remote, Unknown, InvalidAction, digest, messages, parse
from . import prompts
from .audit_prompts import (SINGLE_SESSION, TYPES, GATE_ORACLE_SYSTEM, GATE_SUPPORT_SYSTEM,
                            personal_score, plan_item_type)
from .store import apply, augment, question_key, raw_chunks, text_size
from .edit_checks import assess_edits


def boolean(value, field):
    if type(value.get(field)) is not bool:
        raise Unknown(f"Judge field {field} must be boolean")
    return value[field]


def rejection_reward(gate):
    """把 gate 的拒绝原因映射成「离 accepted 还有多远」的梯度分。

    阶梯严格单调（越晚失败 = 越接近合格）且全部低于 accepted 下限 0.5：
      schema 非法 -0.1 → 形状错 0.0 → 内容空 0.15 → 答案错 0.25
      → 太简单 0.35 → scope 偏 0.45 → accepted 0.5~1.0
    """
    reason = gate.get("reason")
    if reason == "scope_or_completeness":
        return 0.45
    if reason == "closed_book":
        return 0.35
    if reason == "unsupported_answer":
        return 0.25
    if reason == "oracle_or_type":
        oracle = gate.get("oracle") or {}
        if not oracle.get("type_valid", True):
            return 0.0   # 单 session 冒充 / 类型错
        if not oracle.get("user_relevant", True) or not oracle.get("no_answer_leak", True):
            return 0.0   # 无关 / 答案泄漏
        if not oracle.get("answerable", True):
            return 0.15  # 形状对，但答案不在引用证据里
        return 0.0       # multi-session 的 E<2 sessions 硬检查（等同单 session）
    return -0.1          # schema/evidence 非法（validate_question 抛出的各类错）


class Environment:
    def __init__(self, cfg, counter, legacy):
        self.cfg, self.counter = cfg, counter
        self.memory_module, self.pack_module, self.retrieval_module, self.llm_module = legacy
        self.clients = {}
        self.embedder = self.reranker = None
        self.loaded = False
        self.index_cache = {}  # 每个进程仅保留最近少量M索引，不随训练无限增长。
        self._answer_cache = None

    @contextmanager
    def cached_answers(self, max_entries=256):
        """同一冻结环境中的一组采样复用完整答案；退出时释放，不跨状态混用。"""
        if max_entries < 1:
            raise ValueError("Answer cache size must be positive")
        previous = self._answer_cache
        self._answer_cache = (OrderedDict(), max_entries)
        try:
            yield
        finally:
            self._answer_cache = previous

    def policy(self, role):
        if role not in self.clients:
            self.clients[role] = Remote(role, self.llm_module.Client, str(Path(self.cfg.cache) / "api"))
        return self.clients[role]

    def judge(self, system, payload, key):
        prompt = messages(system, payload)
        count = self.counter.prompt_count(prompt)
        if count > self.cfg.judge_input_tokens:
            raise BudgetExceeded(
                f"Judge context exceeds configured token budget ({count}>{self.cfg.judge_input_tokens})",
                stage="judge:" + (key.split(":", 1)[0] if ":" in key else
                                  "faith" if system == prompts.FAITH else "other"),
                tokens=count, limit=self.cfg.judge_input_tokens)
        return self.policy("JUDGE").json(prompt, "judge:" + key, prompt_tokens=count)

    def index(self, documents):
        if not documents:
            return None
        key = digest([(d.id, d.text, d.date, d.session_ids, d.prov, d.neighbors) for d in documents])
        if key in self.index_cache:
            return self.index_cache[key]
        if not self.loaded:
            if self.cfg.embedding != "none":
                self.embedder = self.retrieval_module.Embedder(self.cfg.embedding, self.cfg.embed_model,
                                                               str(Path(self.cfg.cache) / "api"))
            if self.cfg.reranker:
                self.reranker = self.retrieval_module.CrossReranker(self.cfg.reranker)
            self.loaded = True
        idx = self.retrieval_module.Retriever(documents, self.embedder,
                                              str(Path(self.cfg.cache) / "index"), self.reranker)
        if len(self.index_cache) >= 4:
            self.index_cache.pop(next(iter(self.index_cache)))
        self.index_cache[key] = idx
        return idx

    def memory_index(self, memory):
        # Reader只能看到M.text，不能通过prov回取F，不能在render中泄露原session ID。
        D = self.memory_module.Document
        return self.index([D(e["id"], e["text"], "", [], [], []) for e in memory])

    def reader(self, q, history, key, *, task_type=None):
        payload = {"question": q["q"], "question_date": q["question_date"], "history": history}
        if task_type is not None:
            payload["task_type"] = task_type
        prompt = messages(prompts.READ, payload)
        count = self.counter.prompt_count(prompt)
        if count > self.cfg.reader_tokens + 1500:
            raise BudgetExceeded(f"Reader prompt exceeds budget ({count}>{self.cfg.reader_tokens + 1500})",
                                 stage="reader", tokens=count, limit=self.cfg.reader_tokens + 1500)
        value = self.policy("DEFENDER").json(prompt, "reader:" + key, prompt_tokens=count)
        if not isinstance(value.get("answer"), str):
            raise Unknown("Reader response has no answer string")
        return value["answer"]

    def grade(self, q, answer, key, abstention=False):
        value = self.judge(prompts.GRADE, {"question": q["q"], "reference": q["a"],
            "prediction": answer, "type": q["type"], "abstention": abstention}, "grade:" + key)
        return boolean(value, "correct")

    def context(self, memory, q, all_memory=False):
        if all_memory:
            chosen = memory
        else:
            index = self.memory_index(memory)
            hits = index.search(q["q"], self.cfg.reader_k) if index else []
            mapping = {e["id"]: e for e in memory}
            chosen = [mapping[h.id] for h in hits]
        kept, chunks = [], []
        for entry in chosen:
            new = "\n\n".join(chunks + [entry["text"]])
            if self.counter.count(new) > self.cfg.reader_tokens:
                if all_memory:
                    raise BudgetExceeded("All-M condition cannot fit; do not report truncated M as all-M",
                                         stage="reader_all_memory", limit=self.cfg.reader_tokens)
                continue
            chunks.append(entry["text"])
            kept.append(entry["id"])
        return "\n\n".join(chunks), kept

    def answer(self, memory, q, *, all_memory=False, abstention=False, type_hint=None):
        if type_hint is None:
            type_hint = self.cfg.hint_mode == "target_type"
        cache = self._answer_cache
        cache_key = digest([memory, q, all_memory, abstention, type_hint]) if cache is not None else None
        if cache is not None and cache_key in cache[0]:
            cache[0].move_to_end(cache_key)
            return deepcopy(cache[0][cache_key])
        context, ids = self.context(memory, q, all_memory)
        # cache身份包含实际文本、日期、模型配置和完整问答；服务客户端还会hash完整request。
        key = digest([context, q, all_memory, abstention, type_hint])
        answer = self.reader(q, context, key, task_type=q["type"] if type_hint else None)
        correct = self.grade(q, answer, key, abstention)
        result = {"correct": correct, "answer": answer, "visible_ids": ids,
                  "context_tokens": self.counter.count(context)}
        if cache is not None:
            cache[0][cache_key] = deepcopy(result)
            if len(cache[0]) > cache[1]:
                cache[0].popitem(last=False)
        return result

    def faith(self, full, changed):
        if not changed:
            return {"faithful": True, "reason": "No written entry"}
        for entry in changed:
            value = self.judge(prompts.FAITH, {"entries": [entry],
                "history": prompts.source_view(full, entry["prov"])}, digest([full.fingerprint, entry]))
            if not boolean(value, "faithful"):
                return value
        return {"faithful": True, "reason": "All written entries supported"}

    def faith_scores(self, full, changed):
        """逐条faithful判定（不提前返回），供builder_score做连续折扣；无改动返回[]。"""
        return [boolean(self.judge(prompts.FAITH, {"entries": [entry],
                        "history": prompts.source_view(full, entry["prov"])},
                        digest([full.fingerprint, entry])), "faithful")
                for entry in changed]

    def validate_question(self, q, pack, date, qtype, *, seed_only=False):
        if not isinstance(q, dict) or set(q) != {"q", "a", "type", "question_date", "E"}:
            raise InvalidAction("Question schema mismatch")
        if any(not isinstance(q.get(k), str) or not q[k].strip() for k in ["q", "a", "type", "question_date"]):
            raise InvalidAction("Question has empty/non-string fields")
        if q["type"] != qtype or qtype not in TYPES or q["question_date"] != date:
            raise InvalidAction("Question type/date mismatch")
        evidence = q["E"]
        cap = 8 if qtype in {"multi-session", "knowledge-update", "temporal-reasoning"} else 3
        if (not isinstance(evidence, list) or not evidence or len(evidence) > cap or
            any(not isinstance(r, str) for r in evidence) or len(set(evidence)) != len(evidence) or
            not set(evidence) <= set(pack.rids)):
            raise InvalidAction("Unknown/empty/repeated/overbudget evidence")
        if seed_only and qtype in SINGLE_SESSION and not set(evidence) <= set(pack.seed_rids):
            # 多题型计划里单session题也能看到邻居；它们只能引用种子round（同一session）。
            raise InvalidAction("Single-session question cites non-seed rounds")

    def gate(self, full, pack, q, date, qtype, *, seed_only=False):
        options = {"seed_only": True} if seed_only else {}
        result = self._gate_once(full, pack, q, date, qtype, **options)
        if result["status"] != "rejected" or result.get("reason") != "scope_or_completeness":
            return result
        # 一次修复尝试：screen 指出 wrong_rids / additional_rids，修 E 后重跑整条校验链。
        screen = result.get("screen") or {}
        wrong, extra = screen.get("wrong_rids", []), screen.get("additional_rids", [])
        if not (isinstance(wrong, list) and isinstance(extra, list) and (wrong or extra)):
            return result
        if not set(wrong) <= set(q["E"]) or not set(extra) <= set(pack.rids):
            return result
        repaired = {**q, "E": sorted(set(q["E"]) - set(wrong) | set(extra))}
        if not repaired["E"]:
            return result
        fixed = self._gate_once(full, pack, repaired, date, qtype, **options)
        return dict(fixed, repaired=True) if fixed["status"] == "accepted" else result

    def _screen_fits(self, full, q, rids):
        prompt = messages(prompts.SCREEN, {"candidate": q, "history": prompts.source_view(full, rids)})
        return self.counter.prompt_count(prompt) <= self.cfg.judge_input_tokens

    def _screen_rids(self, full, q):
        """宽检索候选：E∪top-screen_k。全部装得进judge预算时与旧实现逐字相同；装不下时E必须
        完整保留（E本身装不下就BudgetExceeded），其余按检索排名逐个放入、放不下的跳过并计数。"""
        index = self.index(full.documents())
        ranked = [h.id for h in index.search(q["q"], self.cfg.screen_k, expand=0)]
        rids = set(q["E"]) | set(ranked)
        if self._screen_fits(full, q, rids):
            return rids, None
        chosen = set(q["E"])
        if not self._screen_fits(full, q, chosen):
            raise BudgetExceeded("Screen evidence alone exceeds judge budget",
                                 stage="judge:screen", limit=self.cfg.judge_input_tokens)
        omitted = 0
        for rid in ranked:
            if rid in chosen:
                continue
            if self._screen_fits(full, q, chosen | {rid}):
                chosen.add(rid)
            else:
                omitted += 1
        return chosen, {"candidates": len(ranked), "kept": len(chosen), "omitted": omitted,
                        "limit": self.cfg.judge_input_tokens}

    def _gate_once(self, full, pack, q, date, qtype, *, seed_only=False):
        try:
            self.validate_question(q, pack, date, qtype, **({"seed_only": True} if seed_only else {}))
        except InvalidAction as exc:
            return {"status": "rejected", "reason": str(exc), "item": q}
        evidence = prompts.source_view(full, q["E"])
        key = digest([full.fingerprint, q])
        oracle = self.judge(GATE_ORACLE_SYSTEM, {"q": q["q"], "question_date": date, "type": qtype,
                              "E_history": evidence}, "oracle:" + key)
        valid = all(boolean(oracle, k) for k in ["answerable", "user_relevant", "type_valid", "no_answer_leak"])
        if qtype == "multi-session":
            valid = valid and len({full.rounds[r].session_id for r in q["E"]}) >= 2
        result = {"item": q, "oracle": oracle}
        if not valid:
            return dict(result, status="rejected", reason="oracle_or_type")
        support = self.judge(GATE_SUPPORT_SYSTEM, {"q": q["q"], "a": q["a"], "oracle_answer": oracle.get("answer"),
            "type": qtype, "question_date": date, "E_history": evidence}, "support:" + key)
        result["support"] = support
        if not boolean(support, "correct"):
            return dict(result, status="rejected", reason="unsupported_answer")
        wide = qtype in {"multi-session", "knowledge-update", "temporal-reasoning"} or bool(re.search(
            r"\b(all|total|latest|current|currently|ever|so far)\b", q["q"], re.I))
        if self.cfg.gate_mode == "full" and wide:
            rids, fitted = self._screen_rids(full, q)
            if fitted:
                result["screen_budget"] = fitted
            screen = self.judge(prompts.SCREEN, {"candidate": q, "history": prompts.source_view(full, rids)},
                                "screen:" + key)
            for field in ("additional_rids", "wrong_rids"):
                value = screen.get(field, [])
                if not isinstance(value, list) or any(not isinstance(r, str) for r in value) or not set(value) <= rids:
                    raise Unknown(f"Screen returned invalid {field}")
            result["screen"] = screen
            if not boolean(screen, "stable"):
                return dict(result, status="rejected", reason="scope_or_completeness")
        closed = self.reader(q, "", "closed:" + key)
        result["closed_book"] = closed
        if self.grade(q, closed, "closed:" + key):
            return dict(result, status="rejected", reason="closed_book")
        return dict(result, status="accepted", reason=None)

    def defect(self, full, memory, q):
        before = self.answer(memory, q)
        if before["correct"]:
            return {"kind": "none", "before": before}
        extra = raw_chunks(full, q["E"], self.counter, self.cfg.timeline_tokens)
        after = self.answer(augment(memory, extra), q)
        if after["correct"]:
            prov = {r for e in memory for r in e["prov"]}
            visible = {r for e in memory if e["id"] in before["visible_ids"] for r in e["prov"]}
            kind = "missing" if not set(q["E"]) <= prov else "lossy_or_reasoning" if set(q["E"]) <= visible else "unretrievable"
            return {"kind": kind, "before": before, "augmented": after}
        result = {"kind": "unresolved", "before": before, "augmented": after}
        if self.cfg.defect_mode == "direct":
            history = "\n\n".join(e["text"] for e in extra)
            if self.counter.count(history) <= self.cfg.reader_tokens:
                ans = self.reader(q, history, "direct:" + digest(q))
                if self.grade(q, ans, "direct:" + digest(q)):
                    result["kind"] = "retrieval_composition_candidate"
        return result

    def builder_score(self, full, state, completion, family=None):
        """统一训练/probe 奖励；family 仅为兼容旧调用保留，不参与评分。

        答题改善、遗忘、不忠实、关键信息丢失和冗余编辑各自计分。
        正向答题奖励要求所有写入忠实；压缩还要求没有已答对题目退化，
        并通过被编辑旧条目的事实保留检查。合法 NOOP 的奖励恒为 0。
        """
        cfg = self.cfg
        # 每项负奖励有界；非法输出严格低于合法输出的理论下界，避免
        # 学到用坏 JSON 逃避信息损失/忠实度惩罚。
        invalid_reward = -(cfg.builder_loss_weight + cfg.builder_faith_weight
                           + cfg.builder_retention_weight + cfg.size_weight
                           + cfg.builder_redundancy_weight * cfg.max_ops + 1.0)
        try:
            action = parse(completion)
            proposed, changed = apply(state["M"], action, mode=state["mode"], visible_ids=state["old_ids"],
                source_ids=[x["rid"] for x in state["x"]], full_ids=full.rounds, counter=self.counter,
                limit=state["entry_limit"], max_ops=cfg.max_ops)
        except InvalidAction as exc:
            return {"reward": invalid_reward, "effective_reward": invalid_reward,
                    "legal": False, "edit_quality_pass": None,
                    "legal_reason": str(exc), "reason": str(exc)}
        tests = state["tests"]
        if not tests:
            raise Unknown("Empty test set: no answer-retention reward is defined")

        faith_scores = self.faith_scores(full, changed)
        faithful = all(faith_scores)  # 空写入（NOOP/纯 DELETE）不产生新断言。
        faith_ratio = sum(faith_scores) / len(faith_scores) if faith_scores else 1.0
        faith_reason = ("No written entry" if not faith_scores else "All written entries supported"
                        if faithful else f"{len(faith_scores) - sum(faith_scores)} of {len(faith_scores)} written entries unsupported")

        before_scores = [self.answer(state["M"], q)["correct"] for q in tests]
        # NOOP 不重复调用有随机性的 Reader/Judge，避免凭环境噪声得到奖励。
        after_scores = ([self.answer(proposed, q)["correct"] for q in tests]
                        if action["ops"] else list(before_scores))
        before_acc = sum(before_scores) / len(tests)
        after_acc = sum(after_scores) / len(tests)
        gain = sum(not b and a for b, a in zip(before_scores, after_scores)) / len(tests)
        loss = sum(b and not a for b, a in zip(before_scores, after_scores)) / len(tests)
        checks = assess_edits(self, full, state, action, proposed, changed)

        before_size = text_size(state["M"], self.counter)
        after_size = text_size(proposed, self.counter)
        source_size = sum(self.counter.count(x["text"]) for x in state["x"])
        editable_size = text_size([e for e in state["M"] if e["id"] in state["old_ids"]], self.counter)
        # 对同一 state 使用固定分母。refine 的 x 为空时也不因一个 token
        # 的增长产生过大罚分；上限保证非法 JSON 的奖励下界有效。
        growth_ratio = min(1.0, max(0.0, (after_size - before_size) / max(1, source_size, editable_size)))
        shrink_ratio = max(0.0, (before_size - after_size) / max(1, before_size))
        compression_eligible = faithful and loss == 0 and checks["retention_pass"]
        safe_compression = compression_eligible and shrink_ratio > 0
        components = {
            "gain": cfg.builder_gain_weight * gain if faithful else 0.0,
            "loss": -cfg.builder_loss_weight * loss,
            "faith": -cfg.builder_faith_weight * (1.0 - faith_ratio),
            "retention": -cfg.builder_retention_weight * checks["information_loss"],
            "compression": cfg.size_weight * shrink_ratio if safe_compression else 0.0,
            "growth": -cfg.size_weight * growth_ratio,
            # 按冗余操作数处罚，加入其他无害操作不能稀释这项罚分。
            "redundancy": -cfg.builder_redundancy_weight * len(checks["redundant_ops"]),
        }
        reward = sum(components.values())
        quality_ok = faithful and loss == 0 and checks["retention_pass"] and not checks["redundant_ops"]
        return {"reward": reward, "effective_reward": reward, "reward_components": components,
                "edit_quality_pass": quality_ok, "legal": True, "faith": faithful,
                "faith_reason": faith_reason, "faith_ratio": faith_ratio, "faith_scores": faith_scores,
                "accuracy": after_acc, "before_accuracy": before_acc, "delta": after_acc - before_acc,
                "gain": gain, "loss": loss, "test_scores": after_scores, "before_scores": before_scores,
                "before_tokens": before_size, "after_tokens": after_size,
                "growth_ratio": growth_ratio, "shrink_ratio": shrink_ratio,
                "compression_eligible": compression_eligible, "safe_compression": safe_compression,
                "noop": not action["ops"], **checks, "proposed": proposed}

    def attacker_score(self, full, state, completion):
        try:
            value = parse(completion)
            if set(value) != {"items"} or not isinstance(value["items"], list) or len(value["items"]) > self.cfg.questions_per_pack:
                raise InvalidAction("items schema/budget violation")
        except InvalidAction as exc:
            return {"reward": -0.1, "effective_reward": -0.1,
                    "reason": str(exc), "legal": False}
        pack = self.pack_module.Pack(**state["pack"])
        seen = set(state.get("seen_keys", []))
        usage = dict(state.get("evidence_usage", {}))
        rewards, details = [], []
        plan, used = state.get("type_plan"), Counter()
        for q in value["items"]:
            qtype = plan_item_type(state, q, used)
            if plan:
                gate = self.gate(full, pack, q, state["date"], qtype, seed_only=True)
            else:
                gate = self.gate(full, pack, q, state["date"], qtype)
            if gate["status"] != "accepted":
                r = rejection_reward(gate)
                rewards.append(r)
                details.append({**gate, "reward": r})
                continue
            # gate may repair E after scope screening. All downstream checks must
            # use the item it actually accepted, not the original incomplete E.
            q = gate["item"]
            key = question_key(q)
            if key in seen:
                rewards.append(0.0)
                details.append({"status": "duplicate", "item": q})
                continue
            seen.add(key)
            defect = self.defect(full, state["M"], q)
            repairable = defect["kind"] not in {"none", "unresolved"}
            weight = self.evidence_weight(full, q, usage)
            for rid in q["E"]:
                usage[rid] = usage.get(rid, 0) + 1
            rewards.append((0.5 + 0.5 * repairable) * weight)
            details.append({"status": "accepted", "item": q, "defect": defect, "weight": weight,
                            "repaired": gate.get("repaired", False)})
        # 固定题数分母：不让单道简单题比多道有价值题占便宜；空列表reward=0。
        reward = sum(rewards) / self.cfg.questions_per_pack
        # Attacker rewards currently have no family-specific gate, so the
        # effective value equals the base value.  Keep the same schema as the
        # builder path so every training entry point optimizes one field.
        return {"reward": reward, "effective_reward": reward,
                "legal": True, "items": details}

    def evidence_weight(self, full, q, usage):
        """同一round被反复出题时边际收益递减；非SSA题证据里没有一人称个人陈述时降权。

        不做按E硬去重：同一round的不同属性仍可得分，只是逐题递减，使"围着一个session刷题"
        的收益有上界（调和级数）而不是线性增长。"""
        weight = 1.0
        if self.cfg.evidence_novelty:
            weight /= 1 + sum(usage.get(r, 0) for r in q["E"]) / len(q["E"])
        if q["type"] != "single-session-assistant" and personal_score(full, q["E"]) == 0:
            weight *= self.cfg.impersonal_weight
        return weight

    def score(self, full, state, completion):
        if state["role"] == "builder":
            return self.builder_score(full, state, completion)
        if state["role"] == "attacker":
            return self.attacker_score(full, state, completion)
        raise ValueError("Unknown policy role")
