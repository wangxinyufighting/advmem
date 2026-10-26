#!/usr/bin/env bash
# 准备GRPO训练与验证所需的全部数据（详见 TRAINING.md 第4节）。各阶段可单独重跑：STAGES="..."。
#
#   CONFIG=configs/rl_hidden_4b_v2.json DATA=data/longmemeval_s.json OUT=data/rl bash scripts/prepare_rl_data.sh
#
# 角色与模型服务（OpenAI兼容接口；未单独设置时回落到LLM_*）：
#   BUILDER   生成训练状态时使用的Builder（起始/SFT模型）
#   ATTACKER  生成训练题库与attacker训练状态的Attacker
#   VAL_ATTACKER  一次性生成冻结val题库的更强模型（建议与JUDGE同级，不参与训练）
#   JUDGE / DEFENDER  gate、忠实度、reader与评分；应与被训练的模型不同
#
# 阶段（默认全部）：
#   prepare          LongMemEval → prepared目录（train/val/test，按相同历史分组）
#   builder_states   train/val：bank → run --mode build --collect → export builder parquet
#   val_bank         VAL_ATTACKER在val case上出题（冻结的Builder验证集）
#   attacker_states  参考Builder建记忆 → 针对M_build出题并收集attacker状态 → export attacker parquet
set -euo pipefail
: "${CONFIG:?set CONFIG}"
: "${OUT:?set OUT (e.g. data/rl)}"
PREPARED=${PREPARED:-data/prepared_longmemeval}
STAGES=${STAGES:-"prepare builder_states val_bank attacker_states"}
TRAIN_PER_TYPE=${TRAIN_PER_TYPE:-}   # 每种题型取几个train case（空=全部）
VAL_PER_TYPE=${VAL_PER_TYPE:-2}      # 验证成本控制：默认每种题型2个val case
PY=${PY:-python}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$OUT"

select_args() {  # $1=split
  local per_type=$TRAIN_PER_TYPE
  [[ "$1" == val ]] && per_type=$VAL_PER_TYPE
  if [[ -n "$per_type" ]]; then echo --split "$1" --per-type "$per_type"; else echo --split "$1"; fi
}
cli() { "$PY" -m admem.cli "$@" --config "$CONFIG"; }
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

if has_stage prepare && [[ ! -f "$PREPARED/manifest.json" ]]; then
  : "${DATA:?set DATA to longmemeval_s.json for the prepare stage}"
  cli prepare --data "$DATA" --out "$PREPARED" --sizes ${SIZES:-300 50 150}
fi

if has_stage builder_states; then
  for split in train val; do
    # 冷启动题库 → 用当前Builder流式建记忆并收集每个窗口的状态（含lookahead测试题与覆盖率评分标记）。
    cli bank --prepared "$PREPARED" $(select_args $split) --out "$OUT/bank_$split"
    cli run --prepared "$PREPARED" $(select_args $split) --mode build --bank "$OUT/bank_$split" \
      --builder-role BUILDER --collect "$OUT/collected_builder" --out "$OUT/build_$split"
    cli export --states "$OUT/collected_builder" --role builder --split $split \
      --out "$OUT/builder_$split.jsonl" --parquet
  done
fi

if has_stage val_bank; then
  # 冻结的留出问答集：之后所有checkpoint都在同一批题上比较。只用val case，绝不用test。
  cli bank --prepared "$PREPARED" $(select_args val) --attacker-role VAL_ATTACKER --out "$OUT/val_bank"
fi

if has_stage attacker_states; then
  for split in train val; do
    # 参考记忆：BUILDER建出的M_build（val上的这份也是attacker验证用的参考记忆）。
    if [[ ! -d "$OUT/build_$split" ]]; then
      cli run --prepared "$PREPARED" $(select_args $split) --mode build --bank "$OUT/bank_$split" \
        --builder-role BUILDER --out "$OUT/build_$split"
    fi
    cli bank --prepared "$PREPARED" $(select_args $split) --memory-run "$OUT/build_$split" \
      --memory-snapshot M_build --collect "$OUT/collected_attacker" --out "$OUT/attacker_bank_$split"
    cli export --states "$OUT/collected_attacker" --role attacker --split $split \
      --out "$OUT/attacker_$split.jsonl" --parquet
  done
fi
echo "done: parquet files in $OUT; see TRAINING.md for the training commands"
