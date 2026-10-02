#!/usr/bin/env bash
# 在独立vLLM服务环境运行。每次换权重使用不同NAME，避免旧API缓存冒充新模型结果。
set -euo pipefail
: "${MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${NAME:?set NAME to a versioned served model name}"
: "${PORT:=8001}"
# Qwen3-Instruct-2507为非思考模型。只启用实际所需的上下文，不直接开262k。
vllm serve "$MODEL" \
  --served-model-name "$NAME" --port "$PORT" \
  --max-model-len "${MAX_MODEL_LEN:-24576}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --dtype bfloat16
