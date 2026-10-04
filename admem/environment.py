"""冻结的检索/reader/oracle环境；训练奖励与部署使用同一编辑和读取路径。"""
from __future__ import annotations

import re
from pathlib import Path

from .common import Remote, Unknown, InvalidAction, digest, messages, parse
from . import prompts
from .audit_prompts import TYPES, GATE_ORACLE_SYSTEM, GATE_SUPPORT_SYSTEM, personal_score
from .store import apply, augment, question_key, raw_chunks, text_size


def boolean(value, field):
    if type(value.get(field)) is not bool:
        raise Unknown(f"Judge field {field} must be boolean")
    return value[field]


class Environment:
    def __init__(self, cfg, counter, legacy):
        self.cfg, self.counter = cfg, counter
        self.memory_module, self.pack_module, self.retrieval_module, self.llm_module = legacy
        self.clients = {}
        self.embedder = self.reranker = None
        self.loaded = False
        self.index_cache = {}  # 每个进程仅保留最近少量M索引，不随训练无限增长。

    def policy(self, role):
        if role not in self.clients:
            self.clients[role] = Remote(role, self.llm_module.Client, str(Path(self.cfg.cache) / "api"))
        return self.clients[role]

    def judge(self, system, payload, key):
        prompt = messages(system, payload)
        count = self.counter.prompt_count(prompt)
        if count > self.cfg.judge_input_tokens:
            raise Unknown(f"Judge context exceeds configured token budget ({count}>{self.cfg.judge_input_tokens})")
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
            raise Unknown(f"Reader prompt exceeds budget ({count}>{self.cfg.reader_tokens + 1500})")
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
                    raise Unknown("All-M condition cannot fit; do not report truncated M as all-M")
                continue
            chunks.append(entry["text"])
            kept.append(entry["id"])
        return "\n\n".join(chunks), kept

    def answer(self, memory, q, *, all_memory=False, abstention=False, type_hint=None):
        if type_hint is None:
            type_hint = self.cfg.hint_mode == "target_type"
        context, ids = self.context(memory, q, all_memory)
        # cache身份包含实际文本、日期、模型配置和完整问答；服务客户端还会hash完整request。
        key = digest([context, q, all_memory, abstention, type_hint])
        answer = self.reader(q, context, key, task_type=q["type"] if type_hint else None)
        correct = self.grade(q, answer, key, abstention)
        return {"correct": correct, "answer": answer, "visible_ids": ids,
                "context_tokens": self.counter.count(context)}

    def faith(self, full, changed):
        if not changed:
            return {"faithful": True, "reason": "No written entry"}
        for entry in changed:
            value = self.judge(prompts.FAITH, {"entries": [entry],
                "history": prompts.source_view(full, entry["prov"])}, digest([full.fingerprint, entry]))
            if not boolean(value, "faithful"):
                return value
        return {"faithful": True, "reason": "All written entries supported"}

    def validate_question(self, q, pack, date, qtype):
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

    def gate(self, full, pack, q, date, qtype):
        try:
            self.validate_question(q, pack, date, qtype)
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
            index = self.index(full.documents())
            rids = set(q["E"]) | {h.id for h in index.search(q["q"], 30, expand=0)}
            screen = self.judge(prompts.SCREEN, {"candidate": q, "history": prompts.source_view(full, rids)},
                                "screen:" + key)
            extra = screen.get("additional_rids", [])
            if not isinstance(extra, list) or any(not isinstance(r, str) for r in extra) or not set(extra) <= rids:
                raise Unknown("Screen returned invalid additional_rids")
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
        """评估一次 Builder 提议，同时返回原始和约束后的奖励。

        ``reward`` 保留答案/压缩的基础奖励，``effective_reward`` 才是
        probe 或训练器可选择使用的任务奖励。这样不会把“答案答对但
        没有压缩”误记为 refine 成功。
        """
        try:
            action = parse(completion)
            proposed, changed = apply(state["M"], action, mode=state["mode"], visible_ids=state["old_ids"],
                source_ids=[x["rid"] for x in state["x"]], full_ids=full.rounds, counter=self.counter,
                limit=state["entry_limit"], max_ops=self.cfg.max_ops)
        except InvalidAction as exc:
            return {"reward": -1.0, "effective_reward": -1.0,
                    "legal": False, "family_constraint_pass": None,
                    "reason": str(exc)}
        faith = self.faith(full, changed)
        if not faith["faithful"]:
            return {"reward": 0.0, "effective_reward": 0.0,
                    "legal": True, "faith": False,
                    "family_constraint_pass": None,
                    "faith_reason": faith.get("reason"),
                    "reason": faith.get("reason")}
        tests = state["tests"]
        if not tests:
            raise Unknown("Empty test set: no answer-retention reward is defined")
        scores = [self.answer(proposed, q)["correct"] for q in tests]
        acc = sum(scores) / len(scores)
        before = text_size(state["M"], self.counter)
        source_size = sum(self.counter.count(x["text"]) for x in state["x"])
        after = text_size(proposed, self.counter)
        saving = max(0.0, 1 - after / max(1, before + source_size))
        bonus = self.cfg.size_weight * saving if all(scores) else 0.0
        constraint_ok = True
        constraint_reason = None
        if state["mode"] == "refine" and after >= before:
            bonus = 0.0
            constraint_ok = False
            constraint_reason = "memory was not shortened"
        if family == "noop_repeat":
            if action.get("ops") != []:
                constraint_ok = False
                constraint_reason = "NOOP must contain an empty ops list"

        base_reward = acc + bonus
        effective_reward = base_reward if constraint_ok else 0.0
        return {"reward": base_reward, "effective_reward": effective_reward,
                "family_constraint_pass": constraint_ok,
                "family_constraint_reason": constraint_reason,
                "legal": True, "faith": True, "accuracy": acc,
                "test_scores": scores, "before_tokens": before, "after_tokens": after,
                "compression_bonus": bonus, "proposed": proposed}

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
        for q in value["items"]:
            gate = self.gate(full, pack, q, state["date"], state["qtype"])
            if gate["status"] != "accepted":
                rewards.append(-0.1)
                details.append(gate)
                continue
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
            details.append({"status": "accepted", "item": q, "defect": defect, "weight": weight})
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
