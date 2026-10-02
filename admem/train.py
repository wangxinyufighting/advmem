"""SFT/GRPO参考训练入口。固定环境状态、在线生成动作；外部reader/judge始终冻结。"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from .common import Config, TokenBudget, Unknown, connect, digest, read, rows, write, write_rows
from .environment import Environment
from .learning import Rollout, advantages, completion_logp, optimize_groups, sample_tokens


def load_training_records(path, role, method):
    records = list(rows(path))
    selected = []
    for row in records:
        state = row["state"]
        if state["split"] != "train":
            raise ValueError("Training input contains non-train state")
        if state["role"] != role or row["prompt"] != state["prompt"]:
            raise ValueError("Policy role/prompt mismatch")
        if method == "sft" and not row.get("completion"):
            continue
        if method == "grpo" and role == "builder" and not state.get("tests"):
            continue
        selected.append(row)
    if not selected:
        raise ValueError("No eligible training states; collect verified examples first")
    return selected


def save_checkpoint(model, tokenizer, optimizer, out, step, attempts, rng, logs):
    target = Path(out) / f"step_{step:05d}_a{attempts:05d}"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(target / "adapter")
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "attempts": attempts,
                "random": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "logs": logs}, target / "trainer.pt")
    write(Path(out) / "latest.json", {"checkpoint": str(target.resolve())})
    write_rows(Path(out) / "train_log.jsonl", logs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True, help="export生成的states JSONL")
    p.add_argument("--role", choices=["builder", "attacker"], required=True)
    p.add_argument("--method", choices=["sft", "grpo"], required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="RL时推荐使用已merge的SFT checkpoint作为冻结base")
    p.add_argument("--revision")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--max-attempts", type=int, default=1000)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--groups-per-update", type=int, default=1)
    p.add_argument("--topvar", type=int, default=1, help="每次从候选group中选reward方差最大的几个")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-prompt-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    if min(a.steps, a.max_attempts, a.groups_per_update, a.topvar, a.lora_r, a.max_prompt_tokens,
           a.max_new_tokens, a.save_every) < 1 or a.group_size < 2 or a.topvar > a.groups_per_update:
        p.error("Invalid training budgets")
    if not 0 < a.clip < 1 or a.beta < 0 or a.lr <= 0:
        p.error("Invalid optimization parameters")
    cfg = Config.load(a.config)
    records = load_training_records(a.data, a.role, a.method)
    if any(r["state"].get("environment_fingerprint") != cfg.fingerprint() for r in records):
        raise ValueError("Collected environment differs from training config; do not silently change reward")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = []
    skipped = []
    for row in records:
        ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        if len(ids) > a.max_prompt_tokens:
            skipped.append({"id": row["id"], "reason": "overlong_prompt", "tokens": len(ids)})
        else:
            encoded.append((row, ids))
    if not encoded:
        raise ValueError("All prompts exceed budget; no silent truncation")
    out = Path(a.out)
    options = {k: v for k, v in vars(a).items() if k not in {"resume", "steps", "max_attempts"}}
    fingerprint = digest([options, cfg.fingerprint(), records])
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "train_config.json"
    if marker.exists() and (not a.resume or read(marker)["fingerprint"] != fingerprint):
        raise ValueError("Output exists or training configuration changed")
    write(marker, {"fingerprint": fingerprint, "options": options,
                   "note": "SFT=verified semantic demonstration; GRPO=online actions on frozen stage states"})
    write(out / "skipped_states.json", skipped)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run orchestration on Mac and training on a CUDA host")
    dtype = torch.bfloat16 if a.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=dtype,
                                               attn_implementation="sdpa", trust_remote_code=False)
    base.to(a.device)
    latest = read(out / "latest.json")["checkpoint"] if a.resume and (out / "latest.json").exists() else None
    if latest:
        model = PeftModel.from_pretrained(base, Path(latest) / "adapter", is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    step, attempts, logs = 0, 0, []
    if latest:
        # 只加载本程序生成、可信的本地checkpoint，不加载来源不明的pickle。
        saved = torch.load(Path(latest) / "trainer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        step, attempts, logs = saved["step"], saved["attempts"], saved["logs"]
        rng.setstate(saved["random"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    env = None
    if a.method == "grpo":
        legacy = connect(cfg.project)
        counter = TokenBudget(cfg.tokenizer, cfg.tokenizer_revision)
        env = Environment(cfg, counter, legacy)
    try:
        while step < a.steps and attempts < a.max_attempts:
            attempts += 1
            if a.method == "sft":
                row, ids = rng.choice(encoded)
                complete = tokenizer.encode(row["completion"], add_special_tokens=False) + [tokenizer.eos_token_id]
                if len(complete) > a.max_new_tokens:
                    logs.append({"attempt": attempts, "status": "skip_long_demonstration", "id": row["id"]})
                    continue
                optimizer.zero_grad(set_to_none=True)
                model.train()
                loss = -completion_logp(model, ids, complete, a.device).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("SFT loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1,
                                               error_if_nonfinite=True)
                optimizer.step()
                step += 1
                logs.append({"step": step, "loss": float(loss.detach()), "method": "sft", "state": row["id"]})
            else:
                candidates, group_logs = [], []
                for gi in range(a.groups_per_update):
                    row, ids = rng.choice(encoded)
                    state = row["state"]
                    full = env.memory_module.FullMemory.load(state["full_path"])
                    if full.fingerprint != state["full_hash"]:
                        raise ValueError("Reward full memory fingerprint mismatch")
                    group = []
                    try:
                        for _ in range(a.group_size):
                            completion = sample_tokens(model, tokenizer, ids, a.max_new_tokens, a.device)
                            text = tokenizer.decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            if not completion:
                                raise Unknown("Empty token sequence")
                            with torch.no_grad():
                                old = completion_logp(model, ids, completion, a.device).cpu()
                                # 每轮RL以--model为reference；建议它是上一阶段merge的SFT/RL权重。
                                with model.disable_adapter():
                                    ref = completion_logp(model, ids, completion, a.device).cpu()
                            reward_text = text
                            if len(completion) >= a.max_new_tokens and completion[-1] != tokenizer.eos_token_id:
                                reward_text += "\n[TRUNCATED_OUTPUT]"
                            detail = env.score(full, state, reward_text)
                            group.append(Rollout(ids, completion, text, old, ref, float(detail["reward"]), detail))
                    except Unknown as exc:
                        # 一条环境判分失败则整组不更新，防止只保留较易判分的动作。
                        group_logs.append({"state": row["id"], "status": "environment_error", "error": str(exc)})
                        write_rows(out / "last_group_errors.jsonl", group_logs)
                        raise  # fail-fast，网络修好后再resume，不烧完整轮API预算。
                    adv, std = advantages([r.reward for r in group])
                    group_logs.append({"state": row["id"], "std": std, "rewards": [r.reward for r in group],
                                       "responses": [r.text for r in group]})
                    if std > 1e-8:
                        candidates.append((std, group))
                chosen = [g for _, g in sorted(candidates, key=lambda x: -x[0])[:a.topvar]]
                if chosen:
                    loss = optimize_groups(model, optimizer, chosen, a.device, a.clip, a.beta)
                    step += 1
                else:
                    loss = None
                logs.append({"step": step, "attempt": attempts, "method": "grpo", "loss": loss,
                             "selected_groups": len(chosen), "groups": group_logs})
            print(f"{a.role} {a.method} step={step}/{a.steps} attempt={attempts} loss={logs[-1].get('loss')}", flush=True)
            if step and step % a.save_every == 0:
                save_checkpoint(model, tokenizer, optimizer, out, step, attempts, rng, logs)
    finally:
        save_checkpoint(model, tokenizer, optimizer, out, step, attempts, rng, logs)
    if step < a.steps:
        raise RuntimeError("Stopped at max-attempts before requested updates; inspect zero-variance/overlong rates")
    return 0


if __name__ == "__main__":
    main()
