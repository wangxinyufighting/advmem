#!/usr/bin/env python
"""按“用途”统计API调用：读取admem的API缓存，用请求的system提示词对上admem里的提示词常量
（gate的oracle/support/screen、reader、grade、faith、faith_locate、事实清单……），给出每类的
调用数和token。只统计真实发出且成功的请求（缓存命中不会写新文件）。

  python scripts/api_breakdown.py --cache .cache/admem/v3/api --since 2h
  python scripts/api_breakdown.py --cache .cache/admem/v3/api --since 2h --base-url commandcode

不改动任何奖励代码，不影响环境指纹。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_since(value):
    import re
    import time
    from datetime import datetime
    if not value:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", value.strip())
    if match:
        return time.time() - float(match[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]
    return datetime.fromisoformat(value).timestamp()


def usage_of(response):
    usage = (response or {}).get("usage") or {}
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return (int(usage.get("prompt_tokens") or 0), 0, int(usage.get("completion_tokens") or 0), int(reasoning or 0))


def known_prompts():
    """admem里所有较长的字符串常量 → 名字（用前200个字符匹配system提示词的开头）。"""
    import importlib
    table = {}
    for module in ("admem.audit_prompts", "admem.prompts", "admem.edit_checks"):
        try:
            mod = importlib.import_module(module)
        except Exception:  # 依赖缺失时退回按首行分组
            continue
        for name, value in vars(mod).items():
            if isinstance(value, str) and len(value) > 120 and name.isupper():
                table.setdefault(value[:200], f"{module.split('.')[-1]}.{name}")
    return table


def label_of(request, table):
    messages = request.get("messages") or []
    system = next((m.get("content") for m in messages if m.get("role") == "system"), None)
    if not isinstance(system, str):
        return "(no system prompt)"
    for prefix, name in table.items():
        if system.startswith(prefix):
            return name
    for prefix, name in table.items():  # attacker/builder提示词会在常量后拼接题型说明等
        if prefix[:120] in system:
            return name
    return "other: " + system.strip().splitlines()[0][:60]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=".cache/admem/api")
    p.add_argument("--since", help="例如 30m、2h、1d")
    p.add_argument("--base-url", help="只统计地址包含该子串的服务")
    a = p.parse_args(argv)
    since = parse_since(a.since)
    table = known_prompts()
    rows = defaultdict(lambda: [0, 0, 0, 0])  # calls, prompt, completion, reasoning
    for path in Path(a.cache).rglob("*.json"):
        try:
            if since and path.stat().st_mtime < since:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or "request" not in record or "response" not in record:
            continue
        url = str(record.get("base_url", ""))
        if a.base_url and a.base_url not in url:
            continue
        prompt, _, completion, reasoning = usage_of(record.get("response"))
        row = rows[(label_of(record["request"], table), record.get("model", ""))]
        row[0] += 1
        row[1] += prompt
        row[2] += completion
        row[3] += reasoning
    total = sum(r[0] for r in rows.values()) or 1
    print(f"{'calls':>7} {'share':>6} {'in_tok':>11} {'out_tok':>9} {'reason':>8}  prompt / model")
    for (label, model), (calls, pin, pout, preason) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        print(f"{calls:7d} {calls / total:6.1%} {pin:11,d} {pout:9,d} {preason:8,d}  {label} / {model}")


if __name__ == "__main__":
    main()
