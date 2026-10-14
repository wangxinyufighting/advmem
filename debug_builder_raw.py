"""从 LLM 缓存中提取 builder 在目标失败窗口的原始输出，不重新调用 API。

目的：区分"编造 provenance / 超预算卡片"到底是
  ① 模型真错（真编造 rid、真该拆没拆），还是
  ② 校验太脆/格式偏差（rid 多空格、token 数盲打）。

只读 .cache/admem/api 里已有的请求/响应缓存，以及 run 目录下的
checkpoint.json / full.json。请在跑过 longmemeval-eval 的机器上、
仓库根目录执行：

    python debug_builder_raw.py
    python debug_builder_raw.py outputs/xxx/run/c0000 .cache/admem/api

默认只打印两类硬错误：Provenance cites an unseen source / Entry text exceeds。
想打印全部失败窗口，把 main() 里的 targets 改成 None 即可。
"""
import json
import sys
from pathlib import Path

from admem.common import connect, digest
from admem.store import neutral_round


def find_full(run_dir, explicit):
    """--prepared 路径不把 full.json 复制进 run 目录，需去 prepared/cases/<key>/ 找。"""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise SystemExit(f"--full 指定的文件不存在: {p}")
    key = run_dir.name  # 例如 c0000
    for p in [run_dir / "full.json",
              Path("data/prepared_longmemeval") / "cases" / key / "full.json",
              Path("data") / "prepared_longmemeval" / "cases" / key / "full.json"]:
        if p.is_file():
            return p
    raise SystemExit(
        f"找不到 full.json（run 目录里没有，prepared/cases/{key}/ 里也没有）。\n"
        f"请显式传入：python debug_builder_raw.py <run_dir> <cache_dir> <full.json路径>"
    )


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/longmemeval_builder_test/run/c0000")
    cache_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".cache/admem/api")
    full_path = find_full(run_dir, sys.argv[3] if len(sys.argv) > 3 else None)

    # 1. 加载本 case 的 full memory，建立 rid -> 原文/顺序 映射
    memory_module, *_ = connect(".")
    full = memory_module.FullMemory.load(full_path)
    print(f"using full.json: {full_path}")
    round_order = {rid: i for i, rid in enumerate(full.ordered(full.rounds))}
    neutral = {rid: neutral_round(full, rid)["text"] for rid in full.rounds}

    # 2. checkpoint 里的失败窗口（window -> reason）
    state = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    failing = {}
    for e in state["events"]:
        if e.get("stage") != "build":
            continue
        c = e.get("checks") or {}
        reason = c.get("reason") or (c.get("faith") or {}).get("reason") or ""
        failing[e["window"]] = reason

    # 3. 扫缓存，收集属于本 case 的 builder(stream) 请求
    entries = []
    for f in cache_dir.glob("*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        req = rec.get("request") or {}
        msgs = req.get("messages") or []
        if len(msgs) < 2 or "memory editor" not in msgs[0].get("content", ""):
            continue
        try:
            payload = json.loads(msgs[1]["content"])
        except Exception:
            continue
        x = payload.get("x") or []
        if not x or payload.get("mode") != "stream":
            continue
        first = x[0]
        rid = first.get("rid")
        if rid not in neutral:
            continue  # 其它 run（probe 等）的合成 rid，跳过
        if first.get("text", "") not in neutral[rid]:
            continue  # x 文本不属于本 case 原文，跳过
        span0 = (first.get("span") or [0])[0]
        raw = rec["response"]["choices"][0]["message"]["content"]
        key = digest([payload.get("allowed_source_rids"), [s.get("text") for s in x]])
        entries.append((round_order[rid], span0, payload, raw, key, f.name))

    # 4. 去重（同 case 用不同 mode/nonce 重跑会产生相同 x 的条目），按窗口顺序排序
    seen, ordered = set(), []
    for e in sorted(entries, key=lambda t: (t[0], t[1])):
        if e[4] in seen:
            continue
        seen.add(e[4])
        ordered.append(e)

    print(f"matched {len(ordered)} stream windows from cache; "
          f"{len(failing)} build events in checkpoint")

    targets = ("Provenance cites an unseen source", "Entry text exceeds")
    for idx, (_, _, payload, raw, _, name) in enumerate(ordered):
        if idx not in failing:
            continue
        reason = failing[idx]
        if targets and not any(t in reason for t in targets):
            continue
        x = payload.get("x") or []
        print("=" * 90)
        print(f"window {idx} | reason: {reason}")
        print(f"allowed_source_rids={payload.get('allowed_source_rids')} "
              f"max_entry_tokens={payload.get('max_entry_tokens')}")
        print(f"M_old ids: {[e.get('id') for e in (payload.get('M_old') or [])]}")
        print("source in window (rid + text preview):")
        for s in x:
            print(f"  - rid={s.get('rid')!r}  text={s.get('text', '')[:80]!r}")
        print("-" * 90)
        print("RAW OUTPUT:")
        print(raw)
        print()


if __name__ == "__main__":
    main()
