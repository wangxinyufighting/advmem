"""无损 full memory。构建阶段只读取历史白名单，不读取目标问题。"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dateutil.parser import parse

from common import digest, read_json, write_json

HISTORY_FIELDS = ("haystack_session_ids", "haystack_dates", "haystack_sessions")
# 标签只允许出现在评测侧；正文内恰好出现同名字样时绝不删除文字。
TURN_LABELS = {"has_answer", "answer_session_ids", "question_type", "question_id"}


def history_only(case: dict) -> dict:
    """提取历史并去掉消息级标签；其他原始消息属性原样保留。"""
    h = {k: copy.deepcopy(case[k]) for k in HISTORY_FIELDS}
    for session in h["haystack_sessions"]:
        for message in session:
            for key in TURN_LABELS:
                message.pop(key, None)
    return h


def date_key(value: str) -> str:
    """日期原串不变，另建排序键；固定 default，避免依赖运行日期。"""
    dt = parse(value, fuzzy=True, default=datetime(2000, 1, 1))
    # 数据的无时区时间只用于一致排序，不据此推断用户实际时区。
    return dt.replace(tzinfo=dt.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat()


@dataclass
class Round:
    rid: str
    session_id: str
    position: int
    message_indices: list[int]
    messages: list[dict]

    @property
    def text(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)


@dataclass
class Session:
    session_id: str
    node_id: str
    original_index: int
    date: str
    sort_date: str
    rids: list[str]


@dataclass
class Document:
    """统一检索接口：既支持原文 round，也支持未来 builder 的 M 条目。"""
    id: str
    text: str
    date: str
    session_ids: list[str]
    prov: list[str]
    neighbors: list[str]

    def render(self) -> str:
        # 原始 session ID（如 answer_*）是数据集标签，只留给评测/排除；模型视图用由 rid 推出的中性 sN。
        nodes = dict.fromkeys(rid.split(":", 1)[0] for rid in self.prov)
        return f"[{self.id} | {self.date} | sessions={','.join(nodes)}]\n{self.text}"


class FullMemory:
    def __init__(self, sessions: list[Session], rounds: list[Round]):
        self.sessions = {s.session_id: s for s in sessions}
        self.rounds = {r.rid: r for r in rounds}
        self.chronological_ids = [s.session_id for s in sorted(
            sessions, key=lambda s: (s.sort_date, s.original_index)
        )]
        self.fingerprint = digest(self.export_history())

    @classmethod
    def build(cls, case: dict) -> "FullMemory":
        h = history_only(case)
        ids, dates, histories = (h[k] for k in HISTORY_FIELDS)
        if not (len(ids) == len(dates) == len(histories)):
            raise ValueError("历史的 session IDs、dates、sessions 长度不一致")
        if len(set(ids)) != len(ids):
            raise ValueError("同一 case 的 session ID 不能重复")
        sessions, rounds = [], []
        for si, (sid, date, messages) in enumerate(zip(ids, dates, histories)):
            if not isinstance(sid, str) or not isinstance(date, str):
                raise ValueError("session ID 和日期必须是字符串")
            rids, mi = [], 0
            while mi < len(messages):
                msg = messages[mi]
                end = mi + 1
                # 正常情况为 user+紧随的 assistant；异常排列保留为单消息 round。
                if msg.get("role") == "user" and end < len(messages):
                    if messages[end].get("role") == "assistant":
                        end += 1
                group = messages[mi:end]
                if any(not isinstance(m.get("content"), str) or not isinstance(m.get("role"), str)
                       for m in group):
                    raise ValueError("本项目只接受 LongMemEval 的 role/content 文本消息")
                rid = f"s{si + 1}:r{len(rids) + 1}"
                rounds.append(Round(rid, sid, len(rids), list(range(mi, end)), group))
                rids.append(rid)
                mi = end
            sessions.append(Session(sid, f"s{si + 1}", si, date, date_key(date), rids))
        result = cls(sessions, rounds)
        if result.export_history() != h:
            raise AssertionError("无损性检验失败")
        return result

    def export_history(self) -> dict:
        sessions = sorted(self.sessions.values(), key=lambda s: s.original_index)
        return {
            "haystack_session_ids": [s.session_id for s in sessions],
            "haystack_dates": [s.date for s in sessions],
            "haystack_sessions": [[copy.deepcopy(m) for rid in s.rids
                                    for m in self.rounds[rid].messages] for s in sessions],
        }

    def ordered(self, rids) -> list[str]:
        return sorted(set(rids), key=lambda rid: (
            self.sessions[self.rounds[rid].session_id].sort_date,
            self.sessions[self.rounds[rid].session_id].original_index,
            self.rounds[rid].position,
        ))

    def structural_edges(self) -> list[tuple[str, str, str]]:
        """边从顺序索引确定性导出，不重复保存一份冗余图。"""
        edges = []
        for s in self.sessions.values():
            edges.extend((s.node_id, rid, "contains") for rid in s.rids)
            edges.extend((a, b, "next_round") for a, b in zip(s.rids, s.rids[1:]))
        edges.extend((self.sessions[a].node_id, self.sessions[b].node_id, "next_session")
                     for a, b in zip(self.chronological_ids, self.chronological_ids[1:]))
        return edges

    def documents(self) -> list[Document]:
        out = []
        for r in self.rounds.values():
            s = self.sessions[r.session_id]
            neighbors = s.rids[max(0, r.position - 1):r.position] + s.rids[r.position + 1:r.position + 2]
            out.append(Document(r.rid, r.text, s.date, [s.session_id], [r.rid], neighbors))
        return out

    def render(self, rids, marks: dict[str, list[str]] | None = None) -> str:
        """marks=None 不显示 M；{} 明确表示 M 为空，所有 round 都标为 ∉M。"""
        parts = []
        for rid in self.ordered(rids):
            r = self.rounds[rid]
            mark = ""
            if marks is not None:
                mids = marks.get(rid, [])
                mark = f"[∈M: {','.join(mids)}] " if mids else "[∉M] "
            s = self.sessions[r.session_id]
            parts.append(f"{mark}[{rid} | session={s.node_id} | {s.date}]\n{r.text}")
        return "\n\n".join(parts)

    def save(self, path: str | Path) -> None:
        write_json(path, {"schema": 1, "fingerprint": self.fingerprint,
                          "sessions": [asdict(s) for s in self.sessions.values()],
                          "rounds": [asdict(r) for r in self.rounds.values()],
                          "chronological_ids": self.chronological_ids})

    @classmethod
    def load(cls, path: str | Path) -> "FullMemory":
        obj = read_json(path)
        if obj.get("schema") != 1:
            raise ValueError("未知 full memory 格式")
        result = cls([Session(**s) for s in obj["sessions"]], [Round(**r) for r in obj["rounds"]])
        if result.fingerprint != obj["fingerprint"]:
            raise ValueError("full memory 内容与 fingerprint 不符")
        return result


def memory_documents(entries: list[dict], full: FullMemory) -> list[Document]:
    """M 的 provenance 只作出处映射，不把对应原文偷偷拼给 reader。"""
    docs = []
    for e in entries:
        if not isinstance(e.get("text"), str) or not e.get("id"):
            raise ValueError("M 条目必须有 id/text/prov")
        prov = list(dict.fromkeys(e.get("prov", [])))
        if not set(prov) <= full.rounds.keys():
            raise ValueError(f"M 的 {e['id']} 引用了未知 rid")
        sids = list(dict.fromkeys(full.rounds[r].session_id for r in prov))
        docs.append(Document(e["id"], e["text"], e.get("date", ""), sids, prov, []))
    return docs


def provenance_marks(entries: list[dict]) -> dict[str, list[str]]:
    marks = {}
    for e in entries:
        for rid in e.get("prov", []):
            marks.setdefault(rid, []).append(e["id"])
    return marks
