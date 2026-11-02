# 2×80GB 节点 + DeepSeek API 的角色环境模板（TRAINING.md §2.2）。用法：
#   export DEEPSEEK_API_KEY=sk-...        # 不要提交到git
#   source examples/env_2gpu_api.sh
# 已经导出的同名变量优先（便于覆盖）。本文件必须用 source，而不是直接执行。
# 本地Defender：source之后再 export DEFENDER_MODEL/BASE_URL/API_KEY 并 unset DEFENDER_EXTRA_BODY（§2.2）。

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "env_2gpu_api.sh: export DEEPSEEK_API_KEY first" >&2
  return 1 2>/dev/null || exit 1
fi

_API_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"       # 客户端会追加 /chat/completions
_NO_THINK='{"thinking": {"type": "disabled"}}'                  # 判官/读者只需要短JSON，不要推理token

# ---- API角色：判官、验证题库作者、（默认的）读者 ----
export JUDGE_MODEL="${JUDGE_MODEL:-deepseek-flash}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-$_API_URL}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-$DEEPSEEK_API_KEY}"
export JUDGE_EXTRA_BODY="${JUDGE_EXTRA_BODY:-$_NO_THINK}"

export VAL_ATTACKER_MODEL="${VAL_ATTACKER_MODEL:-deepseek-v4-pro}"
export VAL_ATTACKER_BASE_URL="${VAL_ATTACKER_BASE_URL:-$_API_URL}"
export VAL_ATTACKER_API_KEY="${VAL_ATTACKER_API_KEY:-$DEEPSEEK_API_KEY}"
export VAL_ATTACKER_EXTRA_BODY="${VAL_ATTACKER_EXTRA_BODY:-$_NO_THINK}"

export DEFENDER_MODEL="${DEFENDER_MODEL:-deepseek-flash}"
export DEFENDER_BASE_URL="${DEFENDER_BASE_URL:-$_API_URL}"
export DEFENDER_API_KEY="${DEFENDER_API_KEY:-$DEEPSEEK_API_KEY}"
if [ -z "${DEFENDER_EXTRA_BODY+x}" ]; then export DEFENDER_EXTRA_BODY="$_NO_THINK"; fi

# ---- 本地策略服务（数据准备阶段在GPU 1上起 scripts/serve_policy.sh，端口8001）----
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3-4b-start}"
export BUILDER_BASE_URL="${BUILDER_BASE_URL:-http://localhost:8001/v1}"
export BUILDER_API_KEY="${BUILDER_API_KEY:-EMPTY}"
export ATTACKER_MODEL="${ATTACKER_MODEL:-qwen3-4b-start}"
export ATTACKER_BASE_URL="${ATTACKER_BASE_URL:-http://localhost:8001/v1}"
export ATTACKER_API_KEY="${ATTACKER_API_KEY:-EMPTY}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"

# ---- 长任务稳健性 ----
export API_TIMEOUT="${API_TIMEOUT:-600}"          # 繁忙的公共API可能挂起请求数分钟
export API_RETRIES="${API_RETRIES:-6}"            # 429/5xx/断连重试；4xx（key/模型错误）立即失败
export API_BACKOFF="${API_BACKOFF:-2}"
export MAX_API_CALLS="${MAX_API_CALLS:-100000000}"  # 每进程每角色的上限；真正的预算是API余额
export EMBED_DEVICE="${EMBED_DEVICE:-cpu}"        # bge-m3跑CPU，不与vLLM/FSDP争显存

unset _API_URL _NO_THINK
