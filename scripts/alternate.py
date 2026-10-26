#!/usr/bin/env python3
"""Attacker/Builder交替训练（TRAINING.md 第9节）。每轮k，当前Builder B与Attacker A：

  1. memories        用B重建train/val记忆（admem.cli run --mode build）。
  2. attacker phase  A针对“记忆池”出题并收集状态：最新记忆（B的）占--pool-share，其余来自较早轮次的记忆；
                     导出后用scripts/rl_loop.py训练，验证对象是B的val记忆 → A'。
  3. adversarial bank A'针对B的train记忆出题，只保留暴露可修复缺陷的题（--defects-only），并入累计题库
                     （相当于Code-A1的Mistake Book：旧题一直留在题库里作为回放测试）。
  4. builder phase   用累计题库给B的流式状态配测试题（Builder调用命中缓存，只重算测试题），
                     用rl_loop训练，验证集是冻结的val题库（VAL_ATTACKER生成，从不训练）→ B'。
停止：某一轮两个阶段都没有选出比起点更好的checkpoint（Attacker在B的记忆上找不到更多缺陷，且Builder
在冻结val题库上没有显著提升），或达到--rounds。--start builder时第1轮跳过attacker阶段。

每一步完成后写 <out>/alternate_state.json；同一命令重跑会跳过已完成的步骤。

  python scripts/alternate.py --config configs/rl_hidden_4b_v2.json --prepared data/prepared_longmemeval \\
      --builder /models/builder_sft --attacker /models/attacker_sft --val-bank data/rl/val_bank \\
      --out runs/alternate --rounds 4 --phase-steps 60 --eval-every 20 --gpus 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admem.common import read, write  # noqa: E402

DEFAULT_SERVE = ("vllm serve {model} --served-model-name {name} --port {port} "
                 "--max-model-len {max_model_len} --gpu-memory-utilization 0.85 --dtype bfloat16")


class Alternation:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.state_path = self.out / "alternate_state.json"
        self.state = read(self.state_path) if self.state_path.exists() else {
            "round": 1, "builder": a.builder, "attacker": a.attacker, "bank": a.initial_bank,
            "memories": [], "done": {}, "history": [], "stopped": False, "reason": None}

    # ---------------- 外部动作（测试里替换） ----------------
    def run_cmd(self, command, env=None):
        print("[alternate] " + " ".join(shlex.quote(str(c)) for c in command), flush=True)
        if self.a.dry_run:
            return
        subprocess.run([str(c) for c in command], check=True, cwd=str(ROOT), env=env)

    def serve(self, model, name):
        if self.a.dry_run:
            print(f"[alternate] serve {model} as {name}", flush=True)
            return None
        command = self.a.serve_cmd.format(model=model, name=name, port=self.a.serve_port,
                                          max_model_len=self.a.max_model_len)
        env = dict(os.environ)
        if self.a.serve_gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.a.serve_gpus
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        log = open(self.out / "logs" / f"serve_{name}.log", "w", encoding="utf-8")
        process = subprocess.Popen(shlex.split(command), env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + self.a.serve_timeout
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"model server exited early; see {log.name}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.a.serve_port}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        return process
            except OSError:
                time.sleep(5)
        process.terminate()
        raise RuntimeError(f"model server did not come up; see {log.name}")

    def stop(self, process):
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()

    def phase_summary(self, out_dir):
        if self.a.dry_run and not (Path(out_dir) / "summary.json").exists():
            return {"best": {"tag": "step_00000", "model_path": None}, "reason": "dry-run"}
        return read(Path(out_dir) / "summary.json")

    # ---------------- 辅助 ----------------
    def served_name(self, role, model):
        # 同一权重同一名字：API缓存可复用；不同权重名字不同，缓存不会混用。
        return f"{self.out.name}-{role}-{hashlib.sha256(str(model).encode()).hexdigest()[:10]}"

    def with_model(self, role, model, commands):
        """临时起服务，把<ROLE>_MODEL/_BASE_URL指向它，运行若干admem.cli命令。"""
        name = self.served_name(role.lower(), model)
        server = self.serve(model, name)
        try:
            env = dict(os.environ, **{f"{role}_MODEL": name,
                                      f"{role}_BASE_URL": f"http://127.0.0.1:{self.a.serve_port}/v1",
                                      f"{role}_API_KEY": "EMPTY"})
            for command in commands:
                self.run_cmd(command, env)
        finally:
            self.stop(server)

    def cli(self, *args):
        return [sys.executable, "-m", "admem.cli", *args, "--config", self.a.config]

    def select(self, split):
        per_type = self.a.train_per_type if split == "train" else self.a.val_per_type
        return ["--prepared", self.a.prepared, "--split", split] + (["--per-type", str(per_type)] if per_type else [])

    def rl_loop(self, role, model, train, out, extra):
        command = [sys.executable, str(ROOT / "scripts" / "rl_loop.py"), "--role", role, "--config", self.a.config,
                   "--prepared", self.a.prepared, "--model", model, "--train", train, "--out", out,
                   "--eval-every", str(self.a.eval_every), "--max-steps", str(self.a.phase_steps),
                   "--patience", str(self.a.patience), "--gpus", str(self.a.gpus),
                   "--serve-port", str(self.a.serve_port + 1), "--serve-cmd", self.a.serve_cmd,
                   "--max-model-len", str(self.a.max_model_len)]
        if self.a.serve_gpus:
            command += ["--serve-gpus", self.a.serve_gpus]
        if self.a.val_per_type:
            command += ["--val-per-type", str(self.a.val_per_type)]
        command += extra + shlex.split(self.a.rl_args or "")
        self.run_cmd(command)

    def save(self):
        write(self.state_path, self.state)

    def step(self, name, fn):
        key = f"round_{self.state['round']}"
        done = self.state["done"].setdefault(key, [])
        if name in done:
            return
        fn()
        done.append(name)
        self.save()

    # ---------------- 一轮 ----------------
    def run_round(self):
        k = self.state["round"]
        r = self.out / f"round_{k}"
        builder, attacker, bank = self.state["builder"], self.state["attacker"], self.state["bank"]
        bank_args = ["--bank", bank] if bank else []
        record = self.state.setdefault("current", {"round": k, "builder_in": builder, "attacker_in": attacker})

        def memories():
            self.with_model("BUILDER", builder, [
                self.cli("run", *self.select(split), "--mode", "build", *bank_args, "--builder-role", "BUILDER",
                         "--out", str(r / f"build_{split}")) for split in ("train", "val")])
        self.step("memories", memories)

        skip_attacker = k == 1 and self.a.start == "builder"
        if not skip_attacker:
            def attacker_states():
                older = [m for m in self.state["memories"]][-self.a.pool_size:] if self.a.pool_size else []
                pool = (["--memory-pool", *older, "--memory-latest-share", str(self.a.pool_share)] if older else [])
                self.with_model("ATTACKER", attacker, [
                    self.cli("bank", *self.select("train"), "--memory-run", str(r / "build_train"),
                             "--memory-snapshot", "M_build", *pool, "--collect", str(r / "collected_attacker"),
                             "--out", str(r / "attacker_states_bank")),
                    self.cli("export", "--states", str(r / "collected_attacker"), "--role", "attacker",
                             "--split", "train", "--out", str(r / "attacker_train.jsonl"), "--parquet")])
            self.step("attacker_states", attacker_states)

            def attacker_phase():
                self.rl_loop("attacker", attacker, str(r / "attacker_train.jsonl.parquet"), str(r / "rl_attacker"),
                             ["--memory-run", str(r / "build_val"), "--snapshot", "M_build",
                              "--packs-per-case", str(self.a.packs_per_case)] + shlex.split(self.a.attacker_args or ""))
            self.step("attacker_phase", attacker_phase)
            summary = self.phase_summary(r / "rl_attacker")
            record["attacker_improved"] = summary["best"]["tag"] != "step_00000"
            record["attacker_out"] = attacker = summary["best"]["model_path"] or attacker
            record["attacker_summary"] = {k2: summary.get(k2) for k2 in ("best", "reason", "stopped_early")}
        else:
            record["attacker_improved"], record["attacker_out"] = None, attacker

        def adversarial_bank():
            base = ["--base-bank", bank] if bank else []
            self.with_model("ATTACKER", attacker, [
                self.cli("bank", *self.select("train"), "--memory-run", str(r / "build_train"),
                         "--memory-snapshot", "M_build", *base, "--defects-only", "--out", str(r / "bank"))])
        self.step("adversarial_bank", adversarial_bank)
        bank = str(r / "bank")

        def builder_states():
            # 与第1步同一个Builder、同样的窗口：Builder输出命中API缓存，只按新题库重算测试题。
            self.with_model("BUILDER", builder, [
                self.cli("run", *self.select("train"), "--mode", "build", "--bank", bank, "--builder-role", "BUILDER",
                         "--collect", str(r / "collected_builder"), "--out", str(r / "build_train_states")),
                self.cli("export", "--states", str(r / "collected_builder"), "--role", "builder", "--split", "train",
                         "--out", str(r / "builder_train.jsonl"), "--parquet")])
        self.step("builder_states", builder_states)

        def builder_phase():
            self.rl_loop("builder", builder, str(r / "builder_train.jsonl.parquet"), str(r / "rl_builder"),
                         ["--val-bank", self.a.val_bank, "--max-bank-questions", str(self.a.max_bank_questions)]
                         + shlex.split(self.a.builder_args or ""))
        self.step("builder_phase", builder_phase)
        summary = self.phase_summary(r / "rl_builder")
        record["builder_improved"] = summary["best"]["tag"] != "step_00000"
        record["builder_out"] = summary["best"]["model_path"] or builder
        record["builder_summary"] = {k2: summary.get(k2) for k2 in ("best", "reason", "stopped_early")}

        # 本轮结束：更新模型、题库与记忆池，判断是否停止。
        self.state["memories"].append(str(r / "build_train"))
        self.state.update(builder=record["builder_out"], attacker=record["attacker_out"], bank=bank)
        self.state["history"].append(record)
        self.state.pop("current")
        if record["attacker_improved"] is False and not record["builder_improved"]:
            self.state["stopped"] = True
            self.state["reason"] = (f"round {k}: neither the attacker nor the builder improved on its frozen "
                                    "validation (no new defects found, no significant val-bank gain)")
        self.state["round"] = k + 1
        self.save()

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        if not Path(self.a.val_bank).exists() and not self.a.dry_run:
            raise SystemExit(f"--val-bank {self.a.val_bank} does not exist; create it once with "
                             "STAGES=val_bank scripts/prepare_rl_data.sh (VAL_ATTACKER, never trained)")
        while not self.state["stopped"] and self.state["round"] <= self.a.rounds:
            self.run_round()
        if not self.state["stopped"]:
            self.state["reason"] = f"completed --rounds {self.a.rounds}"
        summary = {"builder": self.state["builder"], "attacker": self.state["attacker"], "bank": self.state["bank"],
                   "rounds": len(self.state["history"]), "stopped_early": self.state["stopped"],
                   "reason": self.state["reason"], "history": self.state["history"]}
        write(self.out / "summary.json", summary)
        self.save()
        print(f"[alternate] done after {summary['rounds']} round(s): builder={summary['builder']} "
              f"attacker={summary['attacker']}; {summary['reason']}", flush=True)
        return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--builder", required=True, help="起始Builder（HF目录，建议先SFT）")
    p.add_argument("--attacker", required=True, help="起始Attacker（HF目录，建议先SFT）")
    p.add_argument("--val-bank", required=True, help="冻结的val题库（VAL_ATTACKER生成，从不训练）")
    p.add_argument("--initial-bank", help="可选：已有的train题库目录，作为累计题库的起点")
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--start", choices=["attacker", "builder"], default="attacker")
    p.add_argument("--phase-steps", type=int, default=60, help="每个阶段最多训练多少步（短阶段，避免过拟合冻结的对手）")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--pool-share", type=float, default=0.5, help="attacker状态中针对最新Builder记忆的pack比例")
    p.add_argument("--pool-size", type=int, default=3, help="记忆池最多包含多少个较早轮次")
    p.add_argument("--packs-per-case", type=int, default=8)
    p.add_argument("--max-bank-questions", type=int, default=30)
    p.add_argument("--train-per-type", type=int)
    p.add_argument("--val-per-type", type=int, default=2)
    p.add_argument("--rl-args", help="追加给两个阶段rl_loop的参数（引号括起的字符串），例如 \"--batch 8 --lr 1e-6\"")
    p.add_argument("--attacker-args", help="只追加给attacker阶段rl_loop的参数")
    p.add_argument("--builder-args", help="只追加给builder阶段rl_loop的参数")
    p.add_argument("--serve-cmd", default=DEFAULT_SERVE)
    p.add_argument("--serve-port", type=int, default=8200, help="数据步骤用此端口，rl_loop验证用此端口+1")
    p.add_argument("--serve-gpus")
    p.add_argument("--serve-timeout", type=int, default=900)
    p.add_argument("--max-model-len", type=int, default=24576)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if not 0 <= a.pool_share <= 1 or a.rounds < 1 or a.phase_steps < 1:
        raise SystemExit("invalid --pool-share/--rounds/--phase-steps")
    return Alternation(a).run()


if __name__ == "__main__":
    main()
