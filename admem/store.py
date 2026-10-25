"""M 的纯函数编辑：先全部校验，再返回新副本；模型不能直接修改文件。"""
from __future__ import annotations

from copy import deepcopy

from .common import InvalidAction, digest


def text_size(memory, counter):
    return sum(counter.count(e["text"]) for e in memory)


def neutral_round(full, rid):
    r = full.rounds[rid]
    s = full.sessions[r.session_id]
    text = "\n".join(f"{m['role']}: {m['content']}" for m in r.messages)
    return {"rid": rid, "date": s.date, "text": text}


def raw_chunks(full, rids, counter, limit):
    result = []
    for rid in full.ordered(rids):
        source = neutral_round(full, rid)
        # source文本的每个字符都被保留；日期写入每个块便于独立检索。
        head = f"Recorded on {source['date']}. "
        budget = limit - counter.count(head) - 8
        if budget < 8:
            raise ValueError("Raw chunk budget is too small")
        for start, end, body in counter.split(source["text"], budget):
            text = head + body
            if counter.count(text) > limit:
                raise ValueError("Tokenizer boundary exceeded raw chunk budget")
            result.append({"id": "raw_" + digest([rid, start, end, text])[:20], "text": text,
                           "prov": [rid], "kind": "raw", "span": [start, end]})
    return result


def augment(memory, raw):
    """逐字回退只在推理/状态构建中使用；奖励始终基于模型原始提议。"""
    result = deepcopy(memory)
    seen = {(e["text"], tuple(sorted(e["prov"]))) for e in result}
    for e in raw:
        key = e["text"], tuple(sorted(e["prov"]))
        if key not in seen:
            result.append(deepcopy(e))
            seen.add(key)
    return result


def round_dates(full):
    """rid → 所属session的日期字符串，供卡片记录来源日期。"""
    return {rid: full.sessions[r.session_id].date for rid, r in full.rounds.items()}


def entry_dates(prov, dates):
    """卡片来源日期：prov里最早和最晚的session日期（LongMemEval日期为YYYY/MM/DD开头，按字符串排序即时间序）。"""
    values = sorted({dates[r] for r in prov if r in dates})
    return values[:1] + values[-1:] if len(values) > 1 else values


def apply(memory, action, *, mode, visible_ids, source_ids, full_ids, counter, limit, max_ops=16,
          round_dates=None, forbid_initial_noop=True):
    """并行语义：一次输出不能重复触碰同一旧ID，不能引用本次新建的未知ID。

    round_dates（可选，rid→日期）：给写入的卡片加dates字段，reader渲染时显示来源日期（builder_prompt=v2）。
    forbid_initial_noop=False（builder_prompt=v2）：允许第一个窗口NOOP；v2改用覆盖率奖励而不是硬规则。"""
    if mode not in {"stream", "patch", "refine"}:
        raise ValueError("Invalid edit mode")
    if not isinstance(action, dict) or set(action) != {"ops"} or not isinstance(action["ops"], list):
        raise InvalidAction("Expected exactly {ops: [...]}.")
    if len(action["ops"]) > max_ops:
        raise InvalidAction("Too many operations")
    index = {e["id"]: e for e in memory}
    if len(index) != len(memory):
        raise ValueError("Corrupt M: duplicate entry IDs")
    visible_ids, source_ids, full_ids = set(visible_ids), set(source_ids), set(full_ids)
    if not visible_ids <= index.keys() or not source_ids <= full_ids:
        raise ValueError("Corrupt edit context")
    # 允许 ADD 引用 M_old 里可见条目的既有来源：补存早期窗口漏掉的事实。
    visible_prov = {r for i in visible_ids for r in index[i]["prov"]}
    # A first stream window must not silently turn the entire memory into a
    # vacuous no-op.  The caller can fall back to the raw window instead.
    if forbid_initial_noop and mode == "stream" and not memory and source_ids and not action["ops"]:
        raise InvalidAction("NOOP is not allowed for a nonempty source and empty stream memory")
    used, removed, created, replaced = set(), set(), [], {}
    for pos, op in enumerate(action["ops"]):
        if not isinstance(op, dict):
            raise InvalidAction("Operation must be an object")
        typ = op.get("op")
        allowed = {"ADD": {"op", "text", "prov"}, "UPDATE": {"op", "id", "text", "prov"},
                   "MERGE": {"op", "ids", "text", "prov"}, "DELETE": {"op", "id"}}
        if typ not in allowed or set(op) != allowed[typ]:
            raise InvalidAction("Unknown operation or unexpected/missing fields")
        if mode == "refine" and typ == "ADD":
            raise InvalidAction("ADD is forbidden in refine mode")
        ids = [] if typ == "ADD" else op["ids"] if typ == "MERGE" else [op["id"]]
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
            raise InvalidAction("Invalid entry IDs")
        if len(set(ids)) != len(ids) or (typ == "MERGE" and len(ids) < 2):
            raise InvalidAction("MERGE requires distinct existing IDs")
        if not set(ids) <= visible_ids or used & set(ids):
            raise InvalidAction("Unknown/hidden/repeatedly edited entry ID")
        used.update(ids)
        if typ == "DELETE":
            removed.update(ids)
            continue
        text, prov = op["text"], op["prov"]
        
        # if not isinstance(text, str) or not text.strip() or counter.count(text) > limit:
        #     raise InvalidAction("Empty/non-string/overlong entry text")
        
        # Keep these failures separate: a single aggregate reason made it
        # impossible to tell whether the model omitted text or exceeded the
        # tokenizer budget.
        if not isinstance(text, str):
            raise InvalidAction("Entry text must be a string")
        if not text.strip():
            raise InvalidAction("Entry text is empty")
        text_tokens = counter.count(text)
        if text_tokens > limit:
            raise InvalidAction(f"Entry text exceeds {limit} tokens ({text_tokens})")
        
        if not isinstance(prov, list) or any(not isinstance(r, str) for r in prov) or len(set(prov)) != len(prov):
            raise InvalidAction("Invalid provenance list")
        parent_prov = {r for i in ids for r in index[i]["prov"]}
        # 新增来源必须在当前窗口或可见旧条目中可见；继承来源由执行器自动合并。
        if not set(prov) <= source_ids | visible_prov:
            raise InvalidAction("Provenance cites an unseen source")
        merged_prov = set(prov) | parent_prov
        if not merged_prov or not merged_prov <= full_ids:
            raise InvalidAction("Every written entry needs valid provenance")
        eid = op["id"] if typ == "UPDATE" else "m_" + digest([digest(memory), pos, op])[:20]
        entry = {"id": eid, "text": text, "prov": sorted(merged_prov), "kind": "card"}
        if round_dates is not None:
            entry["dates"] = entry_dates(merged_prov, round_dates)
        if typ == "UPDATE":
            replaced[eid] = entry
        else:
            if eid in index:
                raise InvalidAction("New ID collision")
            created.append(entry)
            removed.update(ids)
    result = [deepcopy(replaced.get(e["id"], e)) for e in memory if e["id"] not in removed]
    result.extend(created)
    changed = list(replaced.values()) + created
    return result, changed


def question_key(q):
    """不要按E去重：同一round可有多个属性。只做保守的规范问句去重。"""
    return digest([" ".join(q["q"].casefold().split()), q["question_date"], q["type"]])
