"""attacker pack的token预算构建与事实句邻居检索（pack_builder="token"）。

预算：用attacker真实tokenizer、按最长的system prompt估算种子部分的prompt长度。session装不下时
切成多个连续片段，每个片段单独成pack；单个round连同system都装不下时记入skipped_rounds，不截断原文、
不静默丢弃。邻居的空间由pack_neighbor_reserve预留，最终是否放得下由attacker_messages逐个裁剪。

邻居（neighbor_mode）：
- topical：与旧Sampler相同，用种子每条用户发言整段检索其他session的round。
- facts：从种子用户发言里挑出一人称事实句（"I bought a road bike for $900"），逐句检索其他session的
  用户句，RRF合并后映射回round；先每个session取1个round，名额有剩余再放宽到neighbor_session_cap。
  LongMemEval的证据多是任务型对话里顺带提到的个人事实，整段发言作查询会被对话主题主导。
- hybrid：先facts，名额不足再用topical补齐。

pack.rids = 种子round（原顺序）+ 邻居（按检索排名）。渲染时总按时间重排，所以这个顺序只决定超预算时
先去掉哪个邻居（末尾=排名最低）。不读取任何评测标签。
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from .audit_prompts import (ATTACKER_SCORING_NOTE, TYPE_GUIDANCE, TYPES, attacker_prompt, attacker_prompt_plan,
                            build_payload)
from .common import messages
from .data import resolve_type_policy

PLACEHOLDER_DATE = "2023/05/30 (Tue) 23:40"  # 与LongMemEval日期同长度，只用于估算
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_FIRST_PERSON = re.compile(
    r"(?<![\w.'’])i(?:['’](?:m|ve|d|ll))?(?![\w.'’])"
    r"|\b(?:my|mine|me|myself|we|we['’](?:re|ve|d|ll)|our|ours|us)\b", re.I)
_REQUEST = re.compile(
    r"^\s*(?:(?:can|could|would|will)\s+you\b|please\b|help\b|write\b|explain\b|give\b|tell\b|list\b"
    r"|create\b|generate\b|show\b|suggest\b|recommend\b|summari[sz]e\b|translate\b|describe\b"
    r"|i\s+want\s+you\s+to\b|act\s+as\b|pretend\b)", re.I)
_EVENT = re.compile(
    r"\b(?:i|we)\s+(?:\w+ly\s+)?(?:have|has|had|got|bought|own|owned|moved|started|finished|joined|adopted"
    r"|visited|went|took|made|met|lost|won|graduated|signed|ordered|booked|paid|spent|received|sold"
    r"|switched|upgraded|planted|ran|completed|am|was|were|used|tried|decided|planned|live|lived|work"
    r"|worked|attended|attend|just|recently)\b|\b(?:my|our)\s+\w+", re.I)
_I_FORMS = {"I", "I'm", "I've", "I'd", "I'll", "I’m", "I’ve", "I’d", "I’ll"}
_FIRST_PERSON_WORD = re.compile(r"(?:i|my|mine|me|myself|our|ours|us|we)(?:['’](?:m|ve|d|ll|re))?", re.I)
_EDGE = ",.;:!?\"'()[]{}“”‘’"


def split_sentences(text):
    return [" ".join(s.split()) for s in _SENTENCE_BREAK.split(text or "") if s and s.strip()]


def _first_person_windows(words, before=8, after=24):
    """超长、缺标点的段落：只取一人称词附近的窗口，避免把整段粘贴内容当查询。"""
    end = -1
    for i, word in enumerate(words):
        if i >= end and _FIRST_PERSON_WORD.fullmatch(word.strip(_EDGE)):
            start, end = max(0, i - before), min(len(words), i + after)
            yield " ".join(words[start:end])


def _segments(text, max_words):
    """查询侧的候选片段：正常句子原样；超过max_words的句子改用一人称词附近的窗口。"""
    for sentence in split_sentences(text):
        words = sentence.split()
        if len(words) <= max_words:
            yield sentence
        else:
            yield from _first_person_windows(words)


def _has_entity(sentence):
    tokens = re.findall(r"[A-Za-z][\w'’-]*", sentence)
    return any(t[0].isupper() and t not in _I_FORMS for t in tokens[1:])


def fact_sentences(texts, limit=8, min_words=4, max_words=50):
    """从用户发言里挑出最像个人事实的句子，作为邻居检索的查询。

    纯启发式、不调用模型；它只决定检索查询，不决定出什么题。必须含一人称；数字（金额、日期、数量）、
    专名、"I bought / my X"类事件加分；以"can you / write / explain"等请求开头的扣分。"""
    scored, seen = [], set()
    for order, sentence in enumerate(s for text in texts for s in _segments(text, max_words)):
        n = len(sentence.split())
        folded = sentence.casefold()
        if not min_words <= n <= max_words or folded in seen or not _FIRST_PERSON.search(sentence):
            continue
        seen.add(folded)
        score = (1.0 + (re.search(r"\d", sentence) is not None) + 0.5 * _has_entity(sentence)
                 + 0.5 * (_EVENT.search(sentence) is not None) - 0.75 * (_REQUEST.search(sentence) is not None))
        if score >= 1.0:
            scored.append((-score, order, sentence))
    return [sentence for _, _, sentence in sorted(scored)[:limit]]


def user_sentence_documents(full, document_cls, max_words=50, chunk_words=40):
    """每个用户句一个检索文档，id为"rid#k"；超长句按chunk_words切成连续片段。
    不带日期，避免日期token干扰句子匹配。"""
    docs = []
    for rid, r in full.rounds.items():
        k = 0
        for m in r.messages:
            if m.get("role") != "user":
                continue
            for sentence in split_sentences(m.get("content", "")):
                words = sentence.split()
                pieces = ([sentence] if len(words) <= max_words else
                          [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)])
                for piece in pieces:
                    if len(piece.split()) < 3:
                        continue
                    docs.append(document_cls(f"{rid}#{k}", piece, "", [r.session_id], [rid], []))
                    k += 1
    return docs


def _user_texts(full, rids):
    return [m["content"] for rid in rids for m in full.rounds[rid].messages if m.get("role") == "user"]


def topical_neighbors(full, seed_rids, seed_session, index, k):
    """旧Sampler的做法：种子每条用户发言整段检索，RRF合并（search_many）。"""
    queries = list(dict.fromkeys(_user_texts(full, seed_rids)))
    if index is None or k < 1 or not queries:
        return []
    hits = index.search_many(queries, k, exclude_sessions={seed_session}, expand=0)
    return [h.id for h in hits if h.id in full.rounds and full.rounds[h.id].session_id != seed_session]


def fact_neighbors(full, seed_rids, seed_session, index, cfg):
    """返回(按RRF排序的round, 使用的事实句查询)。"""
    queries = fact_sentences(_user_texts(full, seed_rids), cfg.fact_queries)
    if index is None or not queries:
        return [], queries
    fused, first_seen = defaultdict(float), {}
    for query in queries:
        rank, seen = 0, set()
        for hit in index.search(query, cfg.fact_query_k, exclude_sessions={seed_session}, expand=0):
            rid = hit.id.split("#", 1)[0]
            if rid not in full.rounds or full.rounds[rid].session_id == seed_session or rid in seen:
                continue
            seen.add(rid)
            rank += 1
            fused[rid] += 1.0 / (60 + rank)
            first_seen.setdefault(rid, len(first_seen))
    return sorted(fused, key=lambda rid: (-fused[rid], first_seen[rid])), queries


def diversify(ranked, session_of, k, cap):
    """按排名挑k个round：第一轮每个session最多1个，之后逐轮放宽到cap，尽量覆盖更多session。"""
    chosen, used = [], Counter()
    for limit in range(1, cap + 1):
        for rid in ranked:
            if len(chosen) >= k:
                return chosen
            session = session_of(rid)
            if rid in chosen or used[session] >= limit:
                continue
            chosen.append(rid)
            used[session] += 1
    return chosen


def choose_neighbors(full, seed_rids, seed_session, round_index, sentence_index, cfg, exclude=frozenset()):
    """exclude：单独都装不进attacker预算的round，作邻居也永远放不进去，不占名额。"""
    k = cfg.neighbors
    if k < 1:
        return [], []
    if cfg.neighbor_mode == "topical":
        topical = topical_neighbors(full, seed_rids, seed_session, round_index, k + len(exclude))
        return [rid for rid in topical if rid not in exclude][:k], []
    ranked, queries = fact_neighbors(full, seed_rids, seed_session, sentence_index, cfg)
    if cfg.neighbor_mode == "hybrid":
        known = set(ranked)
        ranked = ranked + [rid for rid in topical_neighbors(full, seed_rids, seed_session, round_index, 2 * k)
                           if rid not in known]
    ranked = [rid for rid in ranked if rid not in exclude]
    return diversify(ranked, lambda rid: full.rounds[rid].session_id, k, cfg.neighbor_session_cap), queries


def worst_case_system(cfg, counter):
    """最长的attacker system prompt（不含memory说明；超预算时它最先被去掉）。"""
    candidates = [attacker_prompt(t) for t in TYPES]
    if resolve_type_policy(cfg) == "balanced" and cfg.types_per_pack > 1:
        longest = sorted(TYPES, key=lambda t: -counter.count(TYPE_GUIDANCE[t]))
        candidates.append(attacker_prompt_plan({t: 1 for t in longest[:min(cfg.types_per_pack, len(TYPES))]}))
    if getattr(cfg, "attacker_reward", "v1") == "v2":
        candidates = [c + ATTACKER_SCORING_NOTE for c in candidates]
    return max(candidates, key=counter.count)


class _SeedCost:
    """种子片段prompt长度的估算与精确核对。估算=空payload基数+各round JSON的token数之和。"""

    def __init__(self, full, pack_cls, cfg, counter):
        self.full, self.pack_cls, self.cfg, self.counter = full, pack_cls, cfg, counter
        self.system = worst_case_system(cfg, counter)
        self.base = self.exact([])
        self.round_cost = {}

    def payload(self, rids):
        pack = self.pack_cls("estimate", self.full.fingerprint, "estimate", "session", 0, list(rids), list(rids))
        return build_payload(self.full, pack, "single-session-user", PLACEHOLDER_DATE, self.cfg.questions_per_pack,
                             {}, None, view=self.cfg.attacker_view, usage={})

    def exact(self, rids):
        return self.counter.prompt_count(messages(self.system, self.payload(rids)))

    def round(self, rid):
        if rid not in self.round_cost:
            entry = self.payload([rid])["rounds"][0]
            self.round_cost[rid] = self.counter.count(json.dumps(entry, ensure_ascii=False)) + 2
        return self.round_cost[rid]


def split_seed(rids, cost, seed_limit, hard_limit):
    """把一个session的round按顺序装成≤seed_limit的连续片段。单个round超过seed_limit但不超过
    hard_limit时独占一个片段（没有邻居空间）；连hard_limit都超过的round返回在too_big里。"""
    chunks, too_big, current, size = [], [], [], cost.base
    for rid in rids:
        c = cost.round(rid)
        if current and size + c > seed_limit:
            chunks.append(current)
            current, size = [], cost.base
        if not current and cost.base + c > seed_limit:
            tokens = cost.exact([rid])
            if tokens <= hard_limit:
                chunks.append([rid])
            else:
                too_big.append((rid, tokens))
            continue
        current.append(rid)
        size += c
    if current:
        chunks.append(current)
    verified = []
    for chunk in chunks:
        verified.extend(_verify(chunk, cost, seed_limit))
    return verified, too_big


def _verify(chunk, cost, limit):
    """估算只是近似：精确计数仍超出时二分，单round片段原样保留（它已按hard_limit检查过）。"""
    if len(chunk) == 1 or cost.exact(chunk) <= limit:
        return [chunk]
    middle = len(chunk) // 2
    return _verify(chunk[:middle], cost, limit) + _verify(chunk[middle:], cost, limit)


def token_pool(full, env):
    """返回(packs, report)。report列出被切分的session与装不下的round（中性sN/rid，不含原session ID）。"""
    cfg, counter = env.cfg, env.counter
    pack_cls = env.pack_module.Pack
    round_index = env.index(full.documents()) if cfg.neighbor_mode in {"topical", "hybrid"} else None
    sentence_index = None
    if cfg.neighbor_mode in {"facts", "hybrid"}:
        docs = user_sentence_documents(full, env.memory_module.Document)
        sentence_index = env.index(docs) if docs else None
    hard_limit = cfg.attacker_input_tokens
    seed_limit = max(1, int(hard_limit * (1 - cfg.pack_neighbor_reserve)))
    cost = _SeedCost(full, pack_cls, cfg, counter)
    packs, skipped, split_sessions, seeds = [], [], [], []
    for sid in full.chronological_ids:
        session_rids = list(full.sessions[sid].rids)
        neutral = session_rids[0].split(":")[0] if session_rids else None
        chunks, too_big = split_seed(session_rids, cost, seed_limit, hard_limit)
        skipped.extend({"rid": rid, "session": neutral, "tokens": tokens, "limit": hard_limit,
                        "reason": "round_over_attacker_budget"} for rid, tokens in too_big)
        if len(chunks) > 1:
            split_sessions.append({"session": neutral, "parts": len(chunks), "rounds": len(session_rids)})
        seeds.extend((sid, chunk, "session" if len(chunks) == 1 else "session_part") for chunk in chunks)
    unfit = frozenset(row["rid"] for row in skipped)
    for sid, chunk, kind in seeds:
        neighbors, queries = choose_neighbors(full, chunk, sid, round_index, sentence_index, cfg, unfit)
        packs.append(pack_cls(f"p{len(packs) + 1:05d}", full.fingerprint, sid, kind, 0, list(chunk),
                              list(chunk) + [r for r in neighbors if r not in chunk], [], list(queries)))
    report = {"skipped_rounds": skipped, "split_sessions": split_sessions,
              "seed_token_limit": seed_limit, "attacker_input_tokens": hard_limit}
    return packs, report
