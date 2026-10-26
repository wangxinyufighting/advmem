"""冻结的检索/reader/oracle环境；训练奖励与部署使用同一编辑和读取路径。"""
from __future__ import annotations

import re
from collections import Counter, OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from .common import BudgetExceeded, Remote, Unknown, InvalidAction, digest, messages, parse
from . import prompts
from .audit_prompts import (EVIDENCE_AUDIT_SYSTEM, SINGLE_SESSION, TYPES, GATE_ORACLE_SYSTEM,
                            GATE_SUPPORT_SYSTEM, personal_score, plan_item_type)
from . import rewards as rv2
from .store import apply, augment, question_key, raw_chunks, text_size
from .edit_checks import assess_edits, source_coverage
from .store import round_dates


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

    def render_entry(self, entry):
        """reader看到的条目文本。builder_prompt=v2时，模型卡片前加host记录的来源日期；
        raw块本身已有“Recorded on”。v1与旧实现逐字相同。"""
        if self.cfg.builder_prompt != "v2" or entry.get("kind") == "raw" or not entry.get("dates"):
            return entry["text"]
        dates = entry["dates"]
        label = f"Source date: {dates[0]}" if len(dates) == 1 else f"Source dates: {dates[0]} to {dates[-1]}"
        return f"[{label}] {entry['text']}"

    def context(self, memory, q, all_memory=False, k=None):
        if all_memory:
            chosen = memory
        else:
            index = self.memory_index(memory)
            hits = index.search(q["q"], k or self.cfg.reader_k) if index else []
            mapping = {e["id"]: e for e in memory}
            chosen = [mapping[h.id] for h in hits]
        kept, chunks = [], []
        for entry in chosen:
            text = self.render_entry(entry)
            new = "\n\n".join(chunks + [text])
            if self.counter.count(new) > self.cfg.reader_tokens:
                if all_memory:
                    raise BudgetExceeded("All-M condition cannot fit; do not report truncated M as all-M",
                                         stage="reader_all_memory", limit=self.cfg.reader_tokens)
                continue
            chunks.append(text)
            kept.append(entry["id"])
        return "\n\n".join(chunks), kept

    def answer(self, memory, q, *, all_memory=False, abstention=False, type_hint=None, k=None):
        if type_hint is None:
            type_hint = self.cfg.hint_mode == "target_type"
        cache = self._answer_cache
        identity = [memory, q, all_memory, abstention, type_hint] + ([k] if k else [])
        cache_key = digest(identity) if cache is not None else None
        if cache is not None and cache_key in cache[0]:
            cache[0].move_to_end(cache_key)
            return deepcopy(cache[0][cache_key])
        context, ids = self.context(memory, q, all_memory, k) if k else self.context(memory, q, all_memory)
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
        if self.cfg.builder_reward == "v2":
            return self._builder_score_v2(full, state, completion)
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

    def _test_memory(self, full, memory, q):
        rids = q.get("context_rids") or []
        return augment(memory, raw_chunks(full, rids, self.counter, self.cfg.timeline_tokens)) if rids else memory

    def reader_answerable(self, full, q):
        """DEFENDER只看E的原文能否答对（全部放进上下文；放不下时退回检索）。"""
        raw = raw_chunks(full, q["E"], self.counter, self.cfg.timeline_tokens)
        try:
            return self.answer(raw, q, all_memory=True)["correct"]
        except BudgetExceeded:
            return self.answer(raw, q)["correct"]

    def validity_defect(self, full, memory, q):
        """require_reader_answerable下的defect：返回(defect, valid)。
        M非空：先算defect；unresolved（M+原文都答不对）时再看只给原文能否答对——能答对说明是M里的噪声
        挤掉了证据（Builder可修，记为retrieval_composition_candidate），答不对则题目无效。
        M为空：只检查原文可答性，defect记为skipped_empty_memory。"""
        if memory:
            defect = self.defect(full, memory, q)
            if defect["kind"] != "unresolved":
                return defect, True
            if self.reader_answerable(full, q):
                return dict(defect, kind="retrieval_composition_candidate"), True
            return defect, False
        return {"kind": "skipped_empty_memory"}, self.reader_answerable(full, q)

    def _builder_score_v2(self, full, state, completion):
        """v2 Builder奖励（README第14.2节）。与v1的区别：
        - gain/loss按“目标题数”归一化（与当前x有证据交集的测试题，至少1），不再被回放题数稀释；
          两者都截断到[0,1]，回放题被改错照样扣分；
        - 前瞻题（context_rids）在编辑前后都补进其余证据的原文，只衡量本窗口的那部分；
        - k下答案翻转时用2k复核，仅检索排名被挤动的翻转只算一半（NOOP仍不重新作答）；
        - 新增覆盖率项：x里Builder可见范围内尚未存过的事实被存下的比例（不依赖题库，需忠实）；
        - 冗余只对照Builder看得到的条目，且最多计builder_redundancy_cap个；
        - 没有测试题但有x的stream/patch状态也能评分（只有覆盖率/忠实/保留/长度/冗余项）。
        合法NOOP仍为0；非法输出低于合法下界（rewards.builder_invalid_reward_v2）。"""
        cfg = self.cfg
        invalid_reward = rv2.builder_invalid_reward_v2(cfg)
        try:
            action = parse(completion)
            proposed, changed = apply(state["M"], action, mode=state["mode"], visible_ids=state["old_ids"],
                source_ids=[x["rid"] for x in state["x"]], full_ids=full.rounds, counter=self.counter,
                limit=state["entry_limit"], max_ops=cfg.max_ops,
                **({"round_dates": round_dates(full), "forbid_initial_noop": False}
                   if cfg.builder_prompt == "v2" else {}))
        except InvalidAction as exc:
            return {"reward": invalid_reward, "effective_reward": invalid_reward,
                    "legal": False, "edit_quality_pass": None, "reward_version": "v2",
                    "legal_reason": str(exc), "reason": str(exc)}
        tests = state["tests"]
        coverage_on = cfg.builder_coverage_weight > 0 and bool(state["x"]) and state["mode"] in {"stream", "patch"}
        if not tests and not coverage_on:
            raise Unknown("Empty test set and no source window: no reward is defined")
        noop = not action["ops"]
        faith_scores = self.faith_scores(full, changed)
        faithful = all(faith_scores)
        faith_ratio = sum(faith_scores) / len(faith_scores) if faith_scores else 1.0

        x_rids = {s["rid"] for s in state["x"]}
        targets = [bool(set(q["E"]) & x_rids) for q in tests]
        denom = max(1, sum(targets))
        before_scores, after_scores, deltas, confirmed = [], [], [], 0
        for q in tests:
            before_memory = self._test_memory(full, state["M"], q)
            before = self.answer(before_memory, q)["correct"]
            if noop:
                after, delta = before, 0.0
            else:
                after_memory = self._test_memory(full, proposed, q)
                after = self.answer(after_memory, q)["correct"]
                if after != before:
                    confirmed += 1
                    k2 = 2 * cfg.reader_k
                    delta = rv2.flip_delta(before, after, self.answer(before_memory, q, k=k2)["correct"],
                                           self.answer(after_memory, q, k=k2)["correct"])
                else:
                    delta = 0.0
            before_scores.append(before)
            after_scores.append(after)
            deltas.append(delta)
        gain = min(1.0, sum(d for d in deltas if d > 0) / denom)
        loss = min(1.0, -sum(d for d in deltas if d < 0) / denom)
        scope = set(state["old_ids"]) | set(state.get("catalog_ids", []))
        checks = assess_edits(self, full, state, action, proposed, changed, scope_ids=scope)
        coverage = (source_coverage(self, state, proposed, changed) if coverage_on and not noop
                    else {"facts": None, "new": None, "stored": 0, "ratio": None})

        before_size = text_size(state["M"], self.counter)
        after_size = text_size(proposed, self.counter)
        source_size = sum(self.counter.count(x["text"]) for x in state["x"])
        editable_size = text_size([e for e in state["M"] if e["id"] in state["old_ids"]], self.counter)
        growth_ratio = min(1.0, max(0.0, (after_size - before_size) / max(1, source_size, editable_size)))
        shrink_ratio = max(0.0, (before_size - after_size) / max(1, before_size))
        compression_eligible = faithful and loss == 0 and checks["retention_pass"]
        safe_compression = compression_eligible and shrink_ratio > 0
        redundant = min(len(checks["redundant_ops"]), cfg.builder_redundancy_cap)
        components = {
            "gain": cfg.builder_gain_weight * gain if faithful else 0.0,
            "loss": -cfg.builder_loss_weight * loss,
            "faith": -cfg.builder_faith_weight * (1.0 - faith_ratio),
            "retention": -cfg.builder_retention_weight * checks["information_loss"],
            "coverage": (cfg.builder_coverage_weight * coverage["ratio"]
                         if faithful and coverage["ratio"] is not None else 0.0),
            "compression": cfg.size_weight * shrink_ratio if safe_compression else 0.0,
            "growth": -cfg.size_weight * growth_ratio,
            "redundancy": -cfg.builder_redundancy_weight * redundant,
        }
        reward = sum(components.values())
        quality_ok = faithful and loss == 0 and checks["retention_pass"] and not checks["redundant_ops"]
        return {"reward": reward, "effective_reward": reward, "reward_components": components,
                "reward_version": "v2", "edit_quality_pass": quality_ok, "legal": True, "faith": faithful,
                "faith_reason": ("No written entry" if not faith_scores else "All written entries supported"
                                 if faithful else f"{len(faith_scores) - sum(faith_scores)} of "
                                 f"{len(faith_scores)} written entries unsupported"),
                "faith_ratio": faith_ratio, "faith_scores": faith_scores,
                "accuracy": sum(after_scores) / len(tests) if tests else None,
                "before_accuracy": sum(before_scores) / len(tests) if tests else None,
                "gain": gain, "loss": loss, "test_scores": after_scores, "before_scores": before_scores,
                "test_deltas": deltas, "target_tests": sum(targets), "confirmed_flips": confirmed,
                "coverage": coverage, "before_tokens": before_size, "after_tokens": after_size,
                "growth_ratio": growth_ratio, "shrink_ratio": shrink_ratio,
                "compression_eligible": compression_eligible, "safe_compression": safe_compression,
                "noop": noop, **checks, "proposed": proposed}

    def attacker_score(self, full, state, completion):
        if self.cfg.attacker_reward == "v2":
            return self._attacker_score_v2(full, state, completion)
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

    def evidence_audit(self, full, q, prior):
        """v2：一次judge调用判断E里哪些round是必要的、证据是否关于这个用户、是否与已有题语义重复。"""
        payload = {"q": q["q"], "a": q["a"], "type": q["type"],
                   "E_history": prompts.source_view(full, q["E"]),
                   "prior_questions": [{"index": i, "q": p["q"]} for i, p in enumerate(prior)]}
        value = self.judge(EVIDENCE_AUDIT_SYSTEM, payload, "audit:" + digest([full.fingerprint, q, prior]))
        needed = value.get("necessary_rids")
        if (not isinstance(needed, list) or not needed or any(not isinstance(r, str) for r in needed)
                or not set(needed) <= set(q["E"])):
            raise Unknown("Evidence audit returned invalid necessary_rids")
        duplicate = value.get("duplicate_of")
        if duplicate is not None and (type(duplicate) is not int or not 0 <= duplicate < len(prior)):
            raise Unknown("Evidence audit returned invalid duplicate_of")
        return {"necessary_rids": sorted(set(needed)), "personal": boolean(value, "personal"),
                "duplicate_of": duplicate, "reason": value.get("reason")}

    def _attacker_score_v2(self, full, state, completion):
        """v2 attacker奖励（README第14.1节）。每题：
        accepted → value × novelty(max) × 必要证据比例 × (impersonal_weight若证据与用户无关) × (repair_credit若E被修补)，
        value = 1（M为空，不计算defect）或 0.5+0.5×可修复（M非空）；
        rejected → ∈[-penalty,0)，按gate阶段递减；重复（字面或语义）→ 小的负值；
        judge预算溢出 → 0且记over_budget（环境原因，不惩罚策略）。
        总奖励 = Σ / questions_per_pack；空列表 = 0；非法输出 = -(penalty+0.1)，低于任何合法输出。"""
        cfg = self.cfg
        penalty = cfg.attacker_reject_penalty
        try:
            value = parse(completion)
            if set(value) != {"items"} or not isinstance(value["items"], list) or len(value["items"]) > cfg.questions_per_pack:
                raise InvalidAction("items schema/budget violation")
        except InvalidAction as exc:
            reward = rv2.attacker_invalid_reward_v2(penalty)
            return {"reward": reward, "effective_reward": reward, "reason": str(exc), "legal": False,
                    "reward_version": "v2"}
        pack = self.pack_module.Pack(**state["pack"])
        seen = set(state.get("seen_keys", []))
        usage = dict(state.get("evidence_usage", {}))
        prior = [dict(p) for p in state.get("prior_questions", [])]
        rewards, details = [], []
        plan, used = state.get("type_plan"), Counter()
        for q in value["items"]:
            qtype = plan_item_type(state, q, used)
            try:
                gate = (self.gate(full, pack, q, state["date"], qtype, seed_only=True) if plan
                        else self.gate(full, pack, q, state["date"], qtype))
            except BudgetExceeded as exc:
                rewards.append(0.0)
                details.append({"status": "over_budget", "item": q, "reward": 0.0, **exc.details()})
                continue
            if gate["status"] != "accepted":
                r = rv2.rejection_reward_v2(gate, penalty)
                rewards.append(r)
                details.append({**gate, "reward": r, "stage": rv2.rejection_stage(gate)})
                continue
            q = gate["item"]
            key = question_key(q)
            if key in seen:
                r = rv2.duplicate_reward_v2(penalty)
                rewards.append(r)
                details.append({"status": "duplicate", "item": q, "reward": r, "duplicate": "exact"})
                continue
            related = [p for p in prior if set(p["E"]) & set(q["E"])][-8:]
            try:
                audit = self.evidence_audit(full, q, related)
            except BudgetExceeded as exc:
                rewards.append(0.0)
                details.append({"status": "over_budget", "item": q, "reward": 0.0, **exc.details()})
                continue
            if audit["duplicate_of"] is not None:
                r = rv2.duplicate_reward_v2(penalty)
                rewards.append(r)
                details.append({"status": "duplicate", "item": q, "reward": r, "duplicate": "semantic",
                                "duplicate_of": related[audit["duplicate_of"]]["q"], "audit": audit})
                continue
            seen.add(key)
            weight = 1.0
            if cfg.evidence_novelty:
                weight *= rv2.novelty_weight_v2(usage, q["E"])
            necessity = len(audit["necessary_rids"]) / len(q["E"])
            weight *= necessity
            if q["type"] != "single-session-assistant" and not audit["personal"]:
                weight *= cfg.impersonal_weight
            if gate.get("repaired"):
                weight *= cfg.attacker_repair_credit
            if cfg.require_reader_answerable:
                # 有效性包含“reader仅凭原文可答”：否则Builder永远修不好这道题，对抗训练会漂向这类题。
                defect, valid = self.validity_defect(full, state["M"], q)
                if not valid:
                    r = rv2.rejection_reward_v2({"reason": "reader_unanswerable"}, penalty)
                    rewards.append(r)
                    details.append({"status": "rejected", "reason": "reader_unanswerable", "item": q,
                                    "defect": defect, "reward": r, "stage": "reader_unanswerable"})
                    continue
                value_ = (0.5 + 0.5 * (defect["kind"] not in {"none", "unresolved"})) if state["M"] else 1.0
            elif state["M"]:
                defect = self.defect(full, state["M"], q)
                value_ = 0.5 + 0.5 * (defect["kind"] not in {"none", "unresolved"})
            else:
                # 空M下每道通过gate的题都是“missing”，defect项是常数；跳过它省下reader调用。
                defect, value_ = {"kind": "skipped_empty_memory"}, 1.0
            for rid in q["E"]:
                usage[rid] = usage.get(rid, 0) + 1
            prior.append({"q": q["q"], "E": list(q["E"])})
            r = value_ * weight
            rewards.append(r)
            details.append({"status": "accepted", "item": q, "defect": defect, "weight": weight,
                            "necessity": necessity, "audit": audit, "reward": r,
                            "repaired": gate.get("repaired", False)})
        reward = sum(rewards) / cfg.questions_per_pack
        return {"reward": reward, "effective_reward": reward, "legal": True, "items": details,
                "reward_version": "v2"}

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
