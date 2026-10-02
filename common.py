"""文件、哈希与轻量文本工具。无在线调用。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def iter_cases(path: str | Path) -> Iterator[dict]:
    """正式数据用 ijson 流式读取；没有 ijson 时，小文件仍可运行。"""
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    try:
        import ijson
    except ImportError:
        if path.stat().st_size > 50_000_000:
            raise RuntimeError("大 JSON 请先安装 ijson：pip install ijson")
        yield from read_json(path)
    else:
        with path.open("rb") as f:
            yield from ijson.items(f, "item", use_float=True)


def load_case(path: str | Path, case_index: int = 0, question_id: str | None = None) -> dict:
    for i, case in enumerate(iter_cases(path)):
        if (question_id is not None and case["question_id"] == question_id) or (
            question_id is None and i == case_index
        ):
            return case
    raise ValueError(f"找不到 case：index={case_index}, question_id={question_id}")


_STOP = set("a an the i me my we our you your he she it they their is are was were be been "
            "am do does did have has had of to in on at for from with and or but as this that "
            "these those please can could would should what which how when where who".split())


def tokens(text: str) -> list[str]:
    """LongMemEval 主要为英文；中文按单字索引，避免引入额外分词依赖。"""
    terms = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*|[\u4e00-\u9fff]", text.lower())
    return [t for t in terms if t not in _STOP]


def normalize_answer(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text).casefold()))
