#!/usr/bin/env bash
# 可选的标准VERL后端：独立环境安装，勿与Code-A1 fork混装。
set -euo pipefail
: "${MODEL:?set MODEL to a base or merged SFT checkpoint}"
: "${TRAIN:?set TRAIN to exported train JSONL.parquet}"
: "${VAL:?set VAL to exported validation JSONL.parquet}"
: "${OUT:?set OUT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
  data.train_files="$TRAIN" data.val_files="$VAL" \
  data.train_batch_size="${BATCH:-8}" \
  data.max_prompt_length="${PROMPT_TOKENS:-8192}" data.max_response_length="${RESPONSE_TOKENS:-2048}" \
  data.truncation=error data.filter_overlong_prompts=false \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH:-8}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.02 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  custom_reward_function.path="$ROOT/admem/verl_reward.py" custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node="${GPUS:-4}" trainer.nnodes=1 \
  trainer.total_epochs="${EPOCHS:-1}" trainer.save_freq=10 trainer.test_freq=10 \
  trainer.project_name=adversarial_memory trainer.experiment_name="${EXPERIMENT:-type_aware}" \
  trainer.default_local_dir="$OUT" 'trainer.logger=[console]'
