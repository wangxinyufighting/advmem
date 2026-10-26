#!/usr/bin/env python3
"""GRPO训练 + 周期性验证 + 早停（详见 TRAINING.md 第6节）。

循环（每一步都可中断后用同一命令续跑，状态在 <out>/loop_state.json）：
  0. 起始模型评估一次，作为护栏基线与初始“最佳”。
  1. verl训练到 step+eval_every（trainer.resume_mode=auto，从上一段的checkpoint继续）。
  2. 把 global_step_N/actor 合并成HF模型（verl.model_merger）。
  3. 用vLLM临时起服务，运行 python -m admem.validation（val split、冻结题库/参考记忆），关闭服务。
  4. EarlyStopper：护栏退化→停止；主指标与最佳做配对bootstrap，连续patience次无显著提升→停止。
最后写 <out>/summary.json（最佳checkpoint、停止原因、全部历史）。训练与验证轮流占用同一批GPU。

  python scripts/rl_loop.py --role builder --config configs/rl_hidden_4b_v2.json \\
      --train data/rl/builder_train.jsonl.parquet --val data/rl/builder_val.jsonl.parquet \\
      --model /models/qwen3-4b-sft --prepared data/prepared_longmemeval \\
      --val-bank data/rl/val_bank --out runs/rl_builder --eval-every 20 --max-steps 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admem.common import read, write  # noqa: E402
from admem.validation import ATTACKER_GUARDS, BUILDER_GUARDS, EarlyStopper  # noqa: E402
from admem.verl_launch import add_arguments, options_from, run as run_verl, verl_env  # noqa: E402

DEFAULT_MERGE = ("{python} -m verl.model_merger merge --backend fsdp "
                 "--local_dir {actor} --target_dir {target}")
DEFAULT_SERVE = ("vllm serve {model} --served-model-name {name} --port {port} "
                 "--max-model-len {max_model_len} --gpu-memory-utilization 0.85 --dtype bfloat16")


def train_metrics(log_file):
    """从verl的file logger（每段一个jsonl）取最后若干步的KL与训练奖励均值，只作记录/KL护栏。"""
    path = Path(log_file)
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tail = rows[-5:]

    def pick(suffix):
        values = [v for r in tail for k, v in r.get("data", {}).items()
                  if k.endswith(suffix) and isinstance(v, (int, float))]
        return sum(values) / len(values) if values else None

    return {"kl_loss": pick("kl_loss"), "train_reward": pick("critic/score/mean"),
            "response_clip_ratio": pick("response_length/clip_ratio")}


class Loop:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.state_path = self.out / "loop_state.json"
        state = read(self.state_path) if self.state_path.exists() else {}
        guards = [dict(g) for g in (BUILDER_GUARDS if a.role == "builder" else ATTACKER_GUARDS)]
        tolerances = {"legal_rate": a.legal_tol, "faith_rate": a.faith_tol, "noop_rate": a.noop_tol,
                      "fallback_rate": a.fallback_tol, "mean_m_tokens": a.size_ratio,
                      "invalid_rate": a.invalid_tol, "accept_rate": a.accept_tol}
        for g in guards:
            g["tolerance"] = tolerances[g["name"]]
        self.stopper = EarlyStopper(guards, a.patience, a.confidence, a.min_delta, a.guard_patience,
                                    a.max_kl, state=state.get("stopper"))
        self.step = state.get("step", 0)

    # ---------- 外部步骤（测试里替换） ----------
    def train_chunk(self, target):
        options = options_from(self.a, out=str(self.out / "verl"), steps=target, save_freq=self.a.eval_every,
                               log_file=str(self.out / "logs" / f"train_to_{target:05d}.jsonl"),
                               max_ckpt_to_keep=1)
        run_verl(options, dry_run=self.a.dry_run)
        return options.log_file

    def merge(self, target):
        actor = self.out / "verl" / f"global_step_{target}" / "actor"
        merged = self.out / "merged" / f"step_{target:05d}"
        if (merged / "config.json").exists() or self.a.dry_run:
            return merged
        if not actor.exists():
            raise RuntimeError(f"verl did not write {actor}; check the training log")
        command = self.a.merge_cmd.format(python=self.a.verl_python or sys.executable, actor=actor, target=merged)
        subprocess.run(shlex.split(command), check=True, cwd=str(ROOT), env=verl_env(options_from(
            self.a, out=str(self.out / "verl"))))
        return merged

    def serve(self, model, name):
        command = self.a.serve_cmd.format(model=model, name=name, port=self.a.serve_port,
                                          max_model_len=self.a.max_model_len)
        env = dict(os.environ)
        if self.a.serve_gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.a.serve_gpus
        log = open(self.out / "logs" / f"serve_{name}.log", "w", encoding="utf-8")
        process = subprocess.Popen(shlex.split(command), env=env, stdout=log, stderr=subprocess.STDOUT)
        url = f"http://127.0.0.1:{self.a.serve_port}/v1/models"
        deadline = time.time() + self.a.serve_timeout
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"model server exited early; see {log.name}")
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        return process
            except OSError:
                time.sleep(5)
        process.terminate()
        raise RuntimeError(f"model server did not come up within {self.a.serve_timeout}s; see {log.name}")

    def stop_server(self, process):
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()

    def validate(self, model, tag):
        """起服务→运行admem.validation→关服务；返回metrics。"""
        eval_dir = self.out / "eval" / tag
        if (eval_dir / "metrics.json").exists():
            return read(eval_dir / "metrics.json")
        role = "BUILDER_EVAL" if self.a.role == "builder" else "ATTACKER_EVAL"
        # 服务名含权重路径的hash：不同权重绝不同名（API缓存按模型名区分），同一权重重跑可复用缓存。
        name = f"{self.out.name}-{tag}-{hashlib.sha256(str(Path(model).resolve()).encode()).hexdigest()[:10]}"
        command = [sys.executable, "-m", "admem.validation", self.a.role, "--config", self.a.config,
                   "--prepared", self.a.prepared, "--split", "val", "--out", str(eval_dir), "--role", role]
        if self.a.role == "builder":
            command += ["--val-bank", self.a.val_bank] if self.a.val_bank else []
            command += ["--max-bank-questions", str(self.a.max_bank_questions)] if self.a.max_bank_questions else []
        else:
            command += ["--memory-run", self.a.memory_run, "--snapshot", self.a.snapshot,
                        "--packs-per-case", str(self.a.packs_per_case)]
        command += ["--limit", str(self.a.val_limit)] if self.a.val_limit else []
        command += ["--per-type", str(self.a.val_per_type)] if self.a.val_per_type else []
        if self.a.dry_run:
            print("[dry-run] serve", model, "as", name, "\n[dry-run]", " ".join(command), flush=True)
            return {"primary": 0.0, "guards": {}, "per_question": {}}
        server = self.serve(model, name)
        try:
            env = dict(os.environ, **{f"{role}_MODEL": name,
                                      f"{role}_BASE_URL": f"http://127.0.0.1:{self.a.serve_port}/v1",
                                      f"{role}_API_KEY": "EMPTY"})
            env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            subprocess.run(command, check=True, cwd=str(ROOT), env=env)
        finally:
            self.stop_server(server)
        return read(eval_dir / "metrics.json")

    # ---------- 主循环 ----------
    def save(self):
        write(self.state_path, {"step": self.step, "stopper": self.stopper.state, "args": vars(self.a)})

    def run(self):
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        if not self.stopper.state["history"]:
            metrics = self.validate(self.a.model, "step_00000")
            self.report(self.stopper.update("step_00000", 0, metrics, self.a.model))
            self.save()
        while self.step < self.a.max_steps and not self.stopper.state["stopped"]:
            target = min(self.step + self.a.eval_every, self.a.max_steps)
            log_file = self.train_chunk(target)
            merged = self.merge(target)
            tag = f"step_{target:05d}"
            metrics = dict(self.validate(str(merged), tag))
            metrics["train"] = train_metrics(log_file)
            self.step = target
            decision = self.stopper.update(tag, target, metrics, str(merged))
            self.report(decision)
            self.cleanup()
            self.save()
        return self.finish()

    def report(self, decision):
        keys = ("tag", "primary", "decision", "improved", "prob_better", "mean_diff", "violations", "best_tag",
                "reason")
        print("[rl_loop] " + json.dumps({k: decision.get(k) for k in keys}, ensure_ascii=False), flush=True)

    def cleanup(self):
        """--keep-merged best：只保留最佳与最新的合并模型，其余删除以节省磁盘。"""
        if self.a.keep_merged != "best" or self.a.dry_run:
            return
        keep = {self.stopper.state["best"].get("model_path")}
        history = self.stopper.state["history"]
        if history:
            keep.add(history[-1].get("model_path"))
        for path in (self.out / "merged").glob("step_*"):
            if str(path) not in keep:
                shutil.rmtree(path, ignore_errors=True)

    def finish(self):
        state = self.stopper.state
        reason = state["reason"] or (f"reached --max-steps {self.a.max_steps}" if self.step >= self.a.max_steps
                                     else None)
        summary = {"role": self.a.role, "best": {k: v for k, v in state["best"].items() if k != "per_question"},
                   "stopped_early": state["stopped"], "reason": reason, "steps_trained": self.step,
                   "history": state["history"],
                   "note": "Selected on the val split only. Report test once with admem.cli longmemeval-eval."}
        write(self.out / "summary.json", summary)
        print(f"[rl_loop] done: best={summary['best'].get('tag')} ({summary['best'].get('model_path')}); "
              f"reason={reason}", flush=True)
        return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument("--out", required=True)
    p.add_argument("--prepared", required=True, help="admem prepared目录（验证用其val split）")
    p.add_argument("--eval-every", type=int, default=20, help="每训练多少步验证一次")
    p.add_argument("--max-steps", type=int, default=300)
    # 停止准则
    p.add_argument("--patience", type=int, default=3, help="连续多少次验证没有显著提升就停止")
    p.add_argument("--confidence", type=float, default=0.9, help="配对bootstrap中P(优于当前最佳)的门槛")
    p.add_argument("--min-delta", type=float, default=0.0, help="主指标均值至少提升多少才算提升")
    p.add_argument("--guard-patience", type=int, default=1, help="连续多少次护栏退化就停止")
    p.add_argument("--max-kl", type=float, help="训练日志里kl_loss超过它视为护栏退化")
    p.add_argument("--legal-tol", type=float, default=0.05)
    p.add_argument("--faith-tol", type=float, default=0.05)
    p.add_argument("--noop-tol", type=float, default=0.15)
    p.add_argument("--fallback-tol", type=float, default=0.10)
    p.add_argument("--size-ratio", type=float, default=1.5, help="平均记忆token数相对基线的上限倍数")
    p.add_argument("--invalid-tol", type=float, default=0.10)
    p.add_argument("--accept-tol", type=float, default=0.10)
    # 验证数据
    p.add_argument("--val-bank", help="builder：冻结attacker生成的val题库目录（主指标）")
    p.add_argument("--max-bank-questions", type=int, default=30)
    p.add_argument("--memory-run", help="attacker：参考记忆run目录")
    p.add_argument("--snapshot", default="M_build")
    p.add_argument("--packs-per-case", type=int, default=8)
    p.add_argument("--val-limit", type=int)
    p.add_argument("--val-per-type", type=int, help="每种题型取几个val case（控制验证成本）")
    # 外部命令
    p.add_argument("--verl-python", help="verl环境的python（默认当前解释器）")
    p.add_argument("--merge-cmd", default=DEFAULT_MERGE)
    p.add_argument("--serve-cmd", default=DEFAULT_SERVE)
    p.add_argument("--serve-port", type=int, default=8100)
    p.add_argument("--serve-gpus", help="验证时vLLM使用的CUDA_VISIBLE_DEVICES（默认继承）")
    p.add_argument("--serve-timeout", type=int, default=900)
    p.add_argument("--max-model-len", type=int, default=24576)
    p.add_argument("--keep-merged", choices=["best", "all"], default="best")
    p.add_argument("--dry-run", action="store_true", help="只打印每一步要执行的命令")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    a = parser().parse_args(argv)
    a.extra = extra
    if a.role == "builder" and not a.val_bank:
        print("[rl_loop] WARNING: no --val-bank; the primary metric falls back to one official question per "
              "val case, which is too noisy to select checkpoints reliably", file=sys.stderr)
    if a.role == "attacker" and not a.memory_run:
        raise SystemExit("--role attacker needs --memory-run (reference memories for the val cases)")
    return Loop(a).run()


if __name__ == "__main__":
    main()
