"""verl自定义优势估计器 "admem_grpo"：GRPO + 组内非法样本整形 + std下限。

为什么需要它：verl按样本独立调用奖励函数，奖励函数看不到同组的其他样本。admem的非法输出奖励
（例如Builder v2约 −8.2）必须低于所有合法输出，否则策略可以用坏JSON逃避忠实/保留惩罚；但它会把
组内std撑大，使合法样本之间的差别（常常只有0.05量级）在标准GRPO归一化后几乎消失。这里在组内把
非法样本的奖励改成“最低合法奖励−合法std”，再用max(std, ADMEM_STD_FLOOR)归一化（见admem.rewards）。

启用方式（scripts/train_verl.sh与scripts/rl_loop.py已自动设置）：
  export VERL_USE_EXTERNAL_MODULES=admem.verl_adv   # verl在import时加载本模块，完成注册
  algorithm.adv_estimator=admem_grpo
  ADMEM_CONFIG=<admem config>  ADMEM_ROLE=builder|attacker   # 用于计算非法阈值
  ADMEM_STD_FLOOR=0.1（默认）
"""
from __future__ import annotations

import os
from functools import lru_cache

import torch

from admem.rewards import group_advantages, invalid_threshold


@lru_cache(maxsize=4)
def _threshold(config_path, role):
    if not config_path or not role:
        return None
    from admem.common import Config
    return invalid_threshold(Config.load(config_path), role)


def compute_admem_grpo_advantage(token_level_rewards, response_mask, index=None, epsilon=1e-6,
                                 config=None, **kwargs):
    """与verl的compute_grpo_outcome_advantage相同的输入输出约定：返回(advantages, returns)，
    形状均为(bs, response_length)，只在有效token上非零。"""
    if index is None:
        raise ValueError("admem_grpo needs the uid index to form GRPO groups")
    scores = token_level_rewards.sum(dim=-1)
    threshold = _threshold(os.getenv("ADMEM_CONFIG"), os.getenv("ADMEM_ROLE"))
    floor = float(os.getenv("ADMEM_STD_FLOOR", "0.1"))
    with torch.no_grad():
        values = group_advantages(scores.detach().float().cpu().tolist(), list(index), threshold, floor, epsilon)
        advantages = torch.tensor(values, dtype=scores.dtype, device=scores.device).unsqueeze(-1) * response_mask
    return advantages, advantages


def register():
    from verl.trainer.ppo import core_algos
    if core_algos.ADV_ESTIMATOR_REGISTRY.get("admem_grpo") is not compute_admem_grpo_advantage:
        core_algos.register_adv_est("admem_grpo")(compute_admem_grpo_advantage)


try:  # 在verl的import过程中被加载时完成注册；单独import（例如测试）且没有verl时跳过。
    register()
except ImportError:  # pragma: no cover - verl未安装
    pass
