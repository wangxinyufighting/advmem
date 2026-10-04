#!/usr/bin/env bash
# attacker消融：同一批case、同一attacker/judge，分别跑各配置的 `admem.cli bank`，再出对比表。
# 只用 prepared 的 context/full；报告不读 private_eval.jsonl。
#
# 用法（在仓库根目录）：
#   export ATTACKER_BASE_URL=http://localhost:8001/v1 ATTACKER_MODEL=qwen3-4b-v1 ATTACKER_API_KEY=EMPTY
#   export JUDGE_BASE_URL=... JUDGE_MODEL=... JUDGE_API_KEY=...      # gate用；不设则回落到 LLM_*
#   DATA=data/longmemeval_s.json bash scripts/eval_attacker.sh
#
# 可选环境变量：
#   PREPARED=data/prepared_longmemeval  不存在时用DATA自动prepare
#   SPLIT=val LIMIT=10 TYPES="multi-session temporal-reasoning"   case选择（TYPES为空=全部题型）
#   BASE_CONFIG=configs/type_aware.json  消融的基准配置（type_hidden.json 则测hidden设定）
#   TOKENIZER=...                        覆盖配置里的tokenizer路径（本地/服务器路径不同时）
#   VARIANTS="baseline compact filter neighbors4 all4b"   要跑的变体
#   HEAD_BASELINE=1                      另在git HEAD代码的worktree上跑一次未修改版本
#   PARALLEL=1                           各变体并行跑（注意API并发和MAX_API_CALLS）
#   OUT=runs/attacker_eval_<时间戳>
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
PREPARED=${PREPARED:-data/prepared_longmemeval}
SPLIT=${SPLIT:-val}
LIMIT=${LIMIT:-10}
TYPES=${TYPES:-}
BASE_CONFIG=${BASE_CONFIG:-configs/type_aware.json}
VARIANTS=${VARIANTS:-baseline compact filter neighbors4 all4b}
OUT=${OUT:-runs/attacker_eval_$(date +%Y%m%d_%H%M%S)}

: "${ATTACKER_MODEL:=${LLM_MODEL:-}}"
: "${JUDGE_MODEL:=${LLM_MODEL:-}}"
if [[ -z "$ATTACKER_MODEL" || -z "$JUDGE_MODEL" ]]; then
  echo "请设置 ATTACKER_MODEL/JUDGE_MODEL（或 LLM_MODEL）及对应 *_BASE_URL/*_API_KEY" >&2
  exit 1
fi
mkdir -p "$OUT/configs"

if [[ ! -f "$PREPARED/manifest.json" ]]; then
  : "${DATA:?PREPARED不存在，请设置 DATA=longmemeval_s.json 以便prepare}"
  "$PY" -m admem.cli prepare --config "$BASE_CONFIG" --data "$DATA" --out "$PREPARED" --sizes 300 50 150
fi

# 生成变体配置。baseline 把新开关全部关掉，等价于旧的attacker输入/奖励（题型指引文本已是新版）。
"$PY" - "$BASE_CONFIG" "$OUT/configs" "${TOKENIZER:-}" <<'EOF'
import json, sys
from pathlib import Path
base, out, tok = json.load(open(sys.argv[1])), Path(sys.argv[2]), sys.argv[3]
if tok:
    base["tokenizer"] = tok
off = {"attacker_view": "full", "attacker_type_filter": False, "attacker_memory_entries": 20,
       "attacker_context_items": 24, "evidence_novelty": False, "neighbors": 8,
       "attacker_input_tokens": 16384}
variants = {
    "baseline": {},
    "compact": {"attacker_view": "compact"},
    "filter": {"attacker_type_filter": True},
    "neighbors4": {"neighbors": 4},
    "all4b": {"attacker_view": "compact", "attacker_type_filter": True, "neighbors": 4,
              "attacker_memory_entries": 0, "attacker_context_items": 8, "evidence_novelty": True,
              "attacker_input_tokens": 12288},
}
for name, delta in variants.items():
    (out / f"{name}.json").write_text(json.dumps({**base, **off, **delta}, indent=2) + "\n")
EOF

run_bank() {  # $1=变体名 $2=代码目录
  local name=$1 code=${2:-.} dest
  dest=$(cd "$OUT" && pwd)/bank_$name
  local args=(-m admem.cli bank --config "$(cd "$OUT/configs" && pwd)/$name.json"
              --prepared "$(cd "$PREPARED" && pwd)" --split "$SPLIT" --limit "$LIMIT" --out "$dest")
  [[ -n "$TYPES" ]] && args+=(--question-types $TYPES)
  echo "== $name ($code) -> $dest"
  # tee保留完整日志，同时把进度条实时显示在终端（进度条写stderr，非TTY时逐行输出）。
  (cd "$code" && "$PY" "${args[@]}") 2>&1 | tee "$OUT/$name.log" \
    || { echo "!! $name 失败，见 $OUT/$name.log" >&2; return 1; }
}

runs=()
pids=()
for v in $VARIANTS; do
  [[ -f "$OUT/configs/$v.json" ]] || { echo "未知变体 $v" >&2; exit 1; }
  runs+=("$OUT/bank_$v")
  if [[ "${PARALLEL:-0}" == 1 ]]; then run_bank "$v" & pids+=($!); else run_bank "$v"; fi
done

if [[ "${HEAD_BASELINE:-0}" == 1 ]]; then
  WT=$(pwd)/.claude/worktrees/attacker_eval_head
  [[ -d "$WT" ]] || git worktree add --detach "$WT" HEAD >/dev/null
  # HEAD的Config不认识新字段，只能用原始基准配置（project "." 指向worktree自身）。
  "$PY" - "$BASE_CONFIG" "$OUT/configs/head.json" "${TOKENIZER:-}" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
new = {"attacker_view", "attacker_type_filter", "seed_min_personal", "attacker_memory_entries",
       "attacker_context_items", "evidence_novelty", "impersonal_weight"}
c = {k: v for k, v in c.items() if k not in new}
if sys.argv[3]:
    c["tokenizer"] = sys.argv[3]
open(sys.argv[2], "w").write(json.dumps(c, indent=2) + "\n")
EOF
  runs+=("$OUT/bank_head")
  if [[ "${PARALLEL:-0}" == 1 ]]; then run_bank head "$WT" & pids+=($!); else run_bank head "$WT"; fi
fi

failed=0
for pid in "${pids[@]:-}"; do [[ -n "$pid" ]] && { wait "$pid" || failed=1; }; done

existing=()
for r in "${runs[@]}"; do [[ -d "$r" ]] && existing+=("$r"); done
"$PY" scripts/attacker_report.py --prepared "$PREPARED" --json "$OUT/report.json" "${existing[@]}" | tee "$OUT/report.md"
echo "结果：$OUT/report.md  （逐题明细：$OUT/bank_*/<case>/bank.json, bank_log.json）"
exit $failed
