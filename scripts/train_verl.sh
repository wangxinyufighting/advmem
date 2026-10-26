#!/usr/bin/env bash
# 单次verl GRPO训练（不做中间验证/早停）。需要早停与选最佳checkpoint时用 scripts/rl_loop.py。
# 参数都由 admem/verl_launch.py 生成；这里只把环境变量翻译成它的命令行参数。详见 TRAINING.md。
#
#   ROLE=builder CONFIG=configs/rl_hidden_4b_v2.json MODEL=/models/qwen3-4b-sft \
#   TRAIN=data/rl/builder_train.jsonl.parquet VAL=data/rl/builder_val.jsonl.parquet OUT=runs/grpo_builder \
#   bash scripts/train_verl.sh [extra.hydra=override ...]
set -euo pipefail
: "${MODEL:?set MODEL to a base or merged SFT checkpoint}"
: "${TRAIN:?set TRAIN to the exported train parquet}"
: "${OUT:?set OUT}"
: "${CONFIG:?set CONFIG to the admem config used to export the states}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
args=(--role "${ROLE:-builder}" --config "$CONFIG" --train "$TRAIN" --model "$MODEL" --out "$OUT"
      --steps "${STEPS:-200}" --save-freq "${SAVE_FREQ:-20}" --gpus "${GPUS:-4}" --batch "${BATCH:-8}"
      --group "${GROUP:-8}" --lr "${LR:-1e-6}" --kl-coef "${KL_COEF:-0.02}"
      --response-tokens "${RESPONSE_TOKENS:-3072}" --reward-workers "${REWARD_WORKERS:-8}"
      --estimator "${ESTIMATOR:-admem_grpo}" --std-floor "${STD_FLOOR:-0.1}" --reward-keys "${REWARD_KEYS:-new}"
      --log-file "${LOG_FILE:-$OUT/metrics.jsonl}")
[[ -n "${VAL:-}" ]] && args+=(--val "$VAL")
[[ -n "${PROMPT_TOKENS:-}" ]] && args+=(--prompt-tokens "$PROMPT_TOKENS")
[[ -n "${EXPERIMENT:-}" ]] && args+=(--experiment "$EXPERIMENT")
cd "$ROOT"
exec python -m admem.verl_launch "${args[@]}" ${DRY_RUN:+--dry-run} -- "$@"
