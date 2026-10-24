"""调试：dump 某个 case 某个 pack 的 attacker 完整输入 / 原始输出 / gate 判定。

用法（从 repo 根目录运行）:
  python scripts/debug_bank_pack.py --config configs/type_aware_4b_nogate.json \
    --prepared data/prepared_longmemeval --key c0070 --pack 0 \
    [--attacker-view compact] [--neighbors 4] [--dry]

--dry 只打印 prompt，不调用模型。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter

from admem.audit_prompts import plan_item_type
from admem.common import Config, TokenBudget, Unknown, connect, read, parse
from admem.environment import Environment
from admem.pipeline import attacker_state, make_pool, propose


def pretty_payload(prompt):
    system = prompt[0]["content"]
    payload = json.loads(prompt[1]["content"])
    return system, payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--pack", type=int, default=0)
    p.add_argument("--attacker-view", choices=["full", "compact"])
    p.add_argument("--neighbors", type=int)
    p.add_argument("--attacker-role", default="ATTACKER")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()

    cfg = Config.load(a.config)
    if a.attacker_view:
        cfg.attacker_view = a.attacker_view
    if a.neighbors is not None:
        cfg.neighbors = a.neighbors
    counter = TokenBudget(cfg.tokenizer, cfg.tokenizer_revision)
    legacy = connect(cfg.project)
    env = Environment(cfg, counter, legacy)

    full_path = f"{a.prepared}/cases/{a.key}/full.json"
    context = read(f"{a.prepared}/cases/{a.key}/context.json")
    full = legacy[0].FullMemory.load(full_path)
    pool = make_pool(full, env)
    pack = pool[a.pack]

    state = attacker_state(full, full_path, context, [], pack, [], env, ["debug", a.pack])
    if state is None:
        print("type_infeasible: 该 pack 对 case 题型不可出题")
        return
    pack = env.pack_module.Pack(**state["pack"])  # gate 用 visible rids

    system, payload = pretty_payload(state["prompt"])
    print("=" * 90)
    print(f"key={a.key} pack={a.pack} qtype={state['qtype']} type_plan={state.get('type_plan')} "
          f"seed={state['pack']['seed_rids'][0].split(':')[0]}")
    print(f"visible_rids={state['pack']['rids']}")
    print("=" * 90)
    print("\n########## SYSTEM ##########\n")
    print(system)
    print("\n########## PAYLOAD ##########\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if a.dry:
        return

    print("\n########## ATTACKER 原始输出 ##########\n")
    raw = propose(state, env, a.attacker_role, f"debug:{a.key}:{a.pack}")
    print(raw)
    data = parse(raw)
    print("\n########## GATE 判定 ##########\n")
    used = Counter()
    for i, q in enumerate(data["items"]):
        qtype = plan_item_type(state, q, used)
        result = (env.gate(full, pack, q, state["date"], qtype, seed_only=True) if state.get("type_plan")
                  else env.gate(full, pack, q, state["date"], qtype))
        print(f"--- 题{i}: {q['q'][:120]}")
        print(f"    status={result.get('status')} reason={result.get('reason')}")
        print(f"    oracle={result.get('oracle')}")
        print(f"    support={result.get('support')}")
        print(f"    closed_book={result.get('closed_book')}")
        print(f"    screen={result.get('screen')}")


if __name__ == "__main__":
    main()
