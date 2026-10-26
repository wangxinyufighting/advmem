"""构造并运行admem的verl GRPO命令。scripts/train_verl.sh（单次训练）与scripts/rl_loop.py（带验证与早停的
分段训练）共用这里的参数，避免两份命令行漂移。

verl的配置键在版本间有变化：
- 新版（2026年的0.10.x，main_ppo走reward loop）：reward.custom_reward_function.*、reward.num_workers；
  顶层custom_reward_function.*在main_ppo里不再迁移，写了也会被忽略，奖励会退回verl默认打分并在
  data_source="admem_builder"上报错。--reward-keys new（默认）。
- 旧版（顶层custom_reward_function、reward_model.reward_manager）：--reward-keys legacy。
本文件针对verl提交3c6d9b4（0.10.0.dev，2026-10-23）核对过配置键；其他版本请先跑一次--dry-run检查。

  python -m admem.verl_launch --role builder --config configs/rl_hidden_4b_v2.json \\
      --train data/rl/builder_train.jsonl.parquet --val data/rl/builder_val.jsonl.parquet \\
      --model /path/to/Qwen3-4B-Instruct-2507 --out runs/grpo_builder --steps 200 [--dry-run] [-- extra=override ...]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LaunchOptions:
    role: str
    config: str
    train: str
    model: str
    out: str
    val: str | None = None
    steps: int = 200
    save_freq: int = 20
    gpus: int = 4
    nnodes: int = 1
    batch: int = 8                 # 每步多少个prompt（状态）
    group: int = 8                 # 每个prompt采样几条（GRPO组大小）
    lr: float = 1e-6
    kl_coef: float = 0.02
    prompt_tokens: int | None = None   # None：由admem config的输入预算推导
    response_tokens: int = 3072
    rollout_gpu_util: float = 0.45
    tensor_parallel: int = 1
    reward_workers: int = 8
    estimator: str = "admem_grpo"      # admem_grpo（整形+std下限）| grpo（verl原版）
    std_floor: float = 0.1
    reward_keys: str = "new"           # new | legacy
    experiment: str | None = None
    log_file: str | None = None
    max_ckpt_to_keep: int | None = 1
    extra: list = field(default_factory=list)


def prompt_budget(config_path, role):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    key = "builder_input_tokens" if role == "builder" else "attacker_input_tokens"
    return int(cfg.get(key, 16384)) + 256


def verl_overrides(o: LaunchOptions) -> list[str]:
    if o.role not in {"builder", "attacker"} or o.estimator not in {"admem_grpo", "grpo"}:
        raise ValueError("role must be builder|attacker and estimator admem_grpo|grpo")
    if o.reward_keys not in {"new", "legacy"}:
        raise ValueError("reward_keys must be new or legacy")
    prompt = o.prompt_tokens or prompt_budget(o.config, o.role)
    reward_path = str(ROOT / "admem" / "verl_reward.py")
    if o.reward_keys == "new":
        reward = [f"reward.custom_reward_function.path={reward_path}",
                  "reward.custom_reward_function.name=compute_score",
                  f"reward.num_workers={o.reward_workers}"]
    else:
        reward = [f"custom_reward_function.path={reward_path}", "custom_reward_function.name=compute_score",
                  "reward_model.reward_manager=naive"]
    loggers = "['console','file']" if o.log_file else "['console']"
    overrides = [
        f"algorithm.adv_estimator={o.estimator}", "algorithm.use_kl_in_reward=False",
        f"data.train_files={o.train}", f"data.val_files={o.val or o.train}",
        f"data.train_batch_size={o.batch}", f"data.max_prompt_length={prompt}",
        f"data.max_response_length={o.response_tokens}", "data.truncation=error",
        "data.filter_overlong_prompts=False",
        f"actor_rollout_ref.model.path={o.model}", "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.external_lib=admem.verl_adv",
        f"actor_rollout_ref.actor.optim.lr={o.lr}", "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={o.batch}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=True", f"actor_rollout_ref.actor.kl_loss_coef={o.kl_coef}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.rollout.name=vllm", f"actor_rollout_ref.rollout.n={o.group}",
        "actor_rollout_ref.rollout.temperature=1.0", "actor_rollout_ref.rollout.top_p=1.0",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={o.tensor_parallel}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={o.rollout_gpu_util}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        *reward,
        f"trainer.n_gpus_per_node={o.gpus}", f"trainer.nnodes={o.nnodes}",
        # 步数由total_training_steps控制；epochs给足，避免数据轮数先用完。
        "trainer.total_epochs=1000", f"trainer.total_training_steps={o.steps}",
        f"trainer.save_freq={o.save_freq}", "trainer.test_freq=-1", "trainer.val_before_train=False",
        "trainer.resume_mode=auto", f"trainer.default_local_dir={o.out}",
        "trainer.project_name=admem", f"trainer.experiment_name={o.experiment or Path(o.out).name}",
        f"trainer.logger={loggers}",
    ]
    if o.max_ckpt_to_keep:
        overrides.append(f"trainer.max_actor_ckpt_to_keep={o.max_ckpt_to_keep}")
    return overrides + list(o.extra)


def verl_env(o: LaunchOptions, base=None) -> dict:
    """verl进程（以及它启动的本机Ray worker）继承的环境变量。多机训练需另行通过
    ray_kwargs.ray_init.runtime_env.env_vars下发同样的变量。"""
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    modules = [m for m in env.get("VERL_USE_EXTERNAL_MODULES", "").split(",") if m]
    if "admem.verl_adv" not in modules:
        modules.append("admem.verl_adv")
    env["VERL_USE_EXTERNAL_MODULES"] = ",".join(modules)
    env["ADMEM_CONFIG"] = str(Path(o.config).resolve())
    env["ADMEM_ROLE"] = o.role
    env["ADMEM_STD_FLOOR"] = str(o.std_floor)
    if o.log_file:
        env["VERL_FILE_LOGGER_PATH"] = str(Path(o.log_file).resolve())
    return env


def verl_command(o: LaunchOptions) -> list[str]:
    return [sys.executable, "-m", "verl.trainer.main_ppo", *verl_overrides(o)]


def run(o: LaunchOptions, dry_run=False, runner=subprocess.run):
    command, env = verl_command(o), verl_env(o)
    if o.log_file:
        Path(o.log_file).parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(shlex.quote(c) for c in command), flush=True)
    if dry_run:
        return 0
    return runner(command, env=env, cwd=str(ROOT), check=True).returncode


def add_arguments(p):
    p.add_argument("--role", choices=["builder", "attacker"], required=True)
    p.add_argument("--config", required=True, help="admem配置（必须与导出状态时的配置相同）")
    p.add_argument("--train", required=True, help="admem.cli export --parquet 生成的训练parquet")
    p.add_argument("--val", help="验证parquet（只用于verl数据加载；早停验证见scripts/rl_loop.py）")
    p.add_argument("--model", required=True, help="起始模型（HF目录；建议先做SFT再merge）")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--prompt-tokens", type=int)
    p.add_argument("--response-tokens", type=int, default=3072)
    p.add_argument("--rollout-gpu-util", type=float, default=0.45)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--reward-workers", type=int, default=8)
    p.add_argument("--estimator", choices=["admem_grpo", "grpo"], default="admem_grpo")
    p.add_argument("--std-floor", type=float, default=0.1)
    p.add_argument("--reward-keys", choices=["new", "legacy"], default="new")
    p.add_argument("--experiment")


def options_from(a, **override) -> LaunchOptions:
    values = dict(role=a.role, config=a.config, train=a.train, val=a.val, model=a.model, gpus=a.gpus,
                  nnodes=a.nnodes, batch=a.batch, group=a.group, lr=a.lr, kl_coef=a.kl_coef,
                  prompt_tokens=a.prompt_tokens, response_tokens=a.response_tokens,
                  rollout_gpu_util=a.rollout_gpu_util, tensor_parallel=a.tensor_parallel,
                  reward_workers=a.reward_workers, estimator=a.estimator, std_floor=a.std_floor,
                  reward_keys=a.reward_keys, experiment=a.experiment, extra=list(getattr(a, "extra", []) or []))
    values.update(override)
    return LaunchOptions(**values)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--save-freq", type=int, default=20)
    p.add_argument("--log-file")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.extra = extra
    return run(options_from(a, out=a.out, steps=a.steps, save_freq=a.save_freq, log_file=a.log_file,
                            max_ckpt_to_keep=None), dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
