"""单GPU参考实现：真在线采样GRPO + 冻结参考策略KL；不做离线加权SFT冒充RL。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Rollout:
    prompt_ids: list[int]
    completion_ids: list[int]
    text: str
    old_logp: torch.Tensor
    ref_logp: torch.Tensor
    reward: float
    detail: dict


def completion_logp(model, prompt_ids, completion_ids, device):
    """只对实际生成token计loss；prompt、padding均不参与策略目标。"""
    if not prompt_ids or not completion_ids:
        raise ValueError("Empty prompt/completion tokens")
    ids = torch.tensor([prompt_ids + completion_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor(completion_ids, dtype=torch.long, device=device)
    # Qwen3@transformers4.57.1支持logits_to_keep，避免计算整段prompt的巨大词表logits。
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                logits_to_keep=len(completion_ids))
    logits = out.logits[0, -len(completion_ids):].float()
    return logits.gather(-1, targets[:, None]).squeeze(-1) - logits.logsumexp(-1)


def advantages(rewards):
    x = torch.as_tensor(rewards, dtype=torch.float32)
    if len(x) < 2 or not torch.isfinite(x).all():
        raise ValueError("GRPO requires >=2 finite rewards")
    std = x.std(unbiased=False)
    return (x - x.mean()) / (std + 1e-6), float(std)


def grpo_loss(logp, old_logp, ref_logp, advantage, clip=0.2, beta=0.02):
    """逐token PPO clip，按实际completion长度归一；k3 KL显式锚定冻结reference。"""
    old_logp, ref_logp = old_logp.to(logp.device), ref_logp.to(logp.device)
    ratio = torch.exp(logp - old_logp)
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    delta = ref_logp - logp
    kl = torch.exp(delta) - delta - 1
    loss = (-objective + beta * kl).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite GRPO loss; do not update parameters")
    return loss


def optimize_groups(model, optimizer, groups, device, clip=0.2, beta=0.02):
    """groups在更新前全部由同一old policy采样/打分；累积梯度后只step一次。"""
    if not groups:
        return 0.0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_sum = 0.0
    for group in groups:
        adv, std = advantages([r.reward for r in group])
        if std <= 1e-8:
            raise ValueError("Zero-variance groups should have been skipped")
        for rollout, a in zip(group, adv):
            logp = completion_logp(model, rollout.prompt_ids, rollout.completion_ids, device)
            loss = grpo_loss(logp, rollout.old_logp, rollout.ref_logp, a.to(device), clip, beta)
            (loss / (len(groups) * len(group))).backward()
            loss_sum += float(loss.detach()) / (len(groups) * len(group))
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                   error_if_nonfinite=True)
    optimizer.step()
    return loss_sum


def sample_tokens(model, tokenizer, prompt_ids, max_new_tokens, device):
    from transformers import GenerationConfig
    # 采用温度1、无top-p/top-k过滤，使sampling分布与teacher-forcing logp一致。
    config = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                               max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                               bos_token_id=tokenizer.bos_token_id, use_cache=True)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config)
    completion = output[0, len(prompt_ids):].tolist()
    if tokenizer.eos_token_id in completion:
        completion = completion[:completion.index(tokenizer.eos_token_id) + 1]
    return completion
