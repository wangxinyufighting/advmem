"""python -m admem.cli：check / prepare / bank / run / probe-suite / probe / export / evaluate / merge。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .common import Config, TokenBudget, Unknown, connect, digest, read, rows, write, write_rows
from .data import contexts, prepare
from .environment import Environment
from .pipeline import Collector, generate_bank, run_case
from .probes import make_suite, run_probe
from .bootstrap import bootstrap
from .store import neutral_round, text_size


def export_states(collected, output, role, split, config_path, parquet=False):
    selected = []
    for path in sorted((Path(collected) / split / role).glob("*.json")):
        row = read(path)
        if row["state"]["split"] != split or row["state"]["role"] != role:
            raise ValueError("State partition mismatch")
        row["state_path"] = str(path.resolve())
        selected.append(row)
    if not selected:
        raise ValueError("No collected states in this partition")
    write_rows(output, selected)
    if parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq
        records = []
        for row in selected:
            state = row["state"]
            if role == "builder" and not state["tests"]:
                continue
            records.append({"data_source": "admem_" + role, "ability": "memory_editing",
                "prompt": row["prompt"], "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"state_path": row["state_path"], "config": str(Path(config_path).resolve()), "split": split}})
        if not records:
            raise ValueError("No reward-bearing states for parquet")
        pq.write_table(pa.Table.from_pylist(records), str(output) + ".parquet")
    return len(selected)


def relocate_states(input_path, prepared, output, core_memory):
    """Mac准备的状态复制到GPU后，仅迁移full_path；内容/hash/prompt不变。"""
    migrated = []
    for row in rows(input_path):
        state = row["state"]
        path = Path(prepared) / "cases" / state["case_key"] / "full.json"
        full = core_memory.FullMemory.load(path)
        if full.fingerprint != state["full_hash"]:
            raise ValueError("Relocated full memory content differs")
        if row["prompt"] != state["prompt"]:
            raise ValueError("State/prompt mismatch")
        state["full_path"] = str(path.resolve())
        # 原state_path是可选VERL侧文件，不直接把旧绝对路径带到新机器。
        sidecar = Path(str(output) + ".states") / (row["id"] + ".json")
        row["state_path"] = str(sidecar.resolve())
        write(sidecar, row)
        migrated.append(row)
    if not migrated:
        raise ValueError("No states to relocate")
    write_rows(output, migrated)
    return len(migrated)


def evaluate(prepared, run_dir, split, snapshot, env, out, limit, keys, all_memory=False):
    # 只有此独立命令读取官方q/a；结果不能反馈给同一test case的memory编辑。
    golds = {q["key"]: q for q in rows(Path(prepared) / "private_eval.jsonl")}
    details = []
    for context, full_path in contexts(prepared, split, limit, keys):
        key = context["key"]
        q = golds[key]
        try:
            if snapshot == "full":
                full = env.memory_module.FullMemory.load(full_path)
                memory = [{"id": r, "text": f"Recorded on {s['date']}. {s['text']}", "prov": [r], "kind": "raw"}
                          for r in full.ordered(full.rounds) for s in [neutral_round(full, r)]]
            else:
                memory = read(Path(run_dir) / key / (snapshot + ".json"))
            result = env.answer(memory, q, all_memory=all_memory, abstention=q["abstention"])
            details.append({"key": key, "question_id": q["question_id"], "type": q["type"],
                "group": "abstention" if q["abstention"] else q["type"], "status": "ok",
                "M_tokens": text_size(memory, env.counter), "entries": len(memory),
                "raw_entries": sum(e.get("kind") == "raw" for e in memory),
                "raw_token_fraction": text_size([e for e in memory if e.get("kind") == "raw"], env.counter) / max(1, text_size(memory, env.counter)), **result})
        except (Unknown, OSError, ValueError) as exc:
            details.append({"key": key, "question_id": q["question_id"], "group": "abstention" if q["abstention"] else q["type"],
                            "status": "error", "error": str(exc)})
        write_rows(Path(out) / "cases.jsonl", details)
    grouped = defaultdict(list)
    for r in details:
        grouped[r["group"]].append(r)
        grouped["ALL_MICRO"].append(r)
    summary = []
    for typ, cases in grouped.items():
        judged = [r for r in cases if r["status"] == "ok"]
        correct = sum(r["correct"] for r in judged)
        unknown = len(cases) - len(judged)
        summary.append({"group": typ, "selected": len(cases), "judged": len(judged), "correct": correct,
            "unknown": unknown, "accuracy_completed": correct / len(judged) if judged else None,
            "lower_all": correct / len(cases), "upper_all": (correct + unknown) / len(cases),
            "mean_M_tokens": sum(r["M_tokens"] for r in judged) / len(judged) if judged else None})
    write(Path(out) / "summary.json", {"hint_mode": env.cfg.hint_mode, "snapshot": snapshot,
          "all_memory": all_memory, "judge": "admem diagnostic; not official execution", "groups": summary})
    write_rows(Path(out) / "predictions.jsonl", [{"question_id": r["question_id"], "hypothesis": r["answer"]}
                                                for r in details if r["status"] == "ok"])
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["check", "prepare", "bank", "run", "bootstrap", "probe-suite", "probe", "export", "relocate", "evaluate", "merge"])
    p.add_argument("--config", default="configs/type_aware.json")
    p.add_argument("--data")
    p.add_argument("--prepared")
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--sizes", nargs=3, type=int, default=[300, 50, 150])
    p.add_argument("--limit", type=int)
    p.add_argument("--keys", nargs="+")
    p.add_argument("--mode", choices=["build", "closed_loop"], default="closed_loop")
    p.add_argument("--builder-role", default="BUILDER")
    p.add_argument("--attacker-role", default="ATTACKER")
    p.add_argument("--collect", help="保存状态和通过验证的模型示范")
    p.add_argument("--bank", help="bank命令输出根目录，用于stream训练状态的自生成问答奖励")
    p.add_argument("--role", choices=["builder", "attacker"], default="builder")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--variants", type=int, default=5)
    p.add_argument("--states", help="probe的状态JSONL或export的collector根目录")
    p.add_argument("--parquet", action="store_true")
    p.add_argument("--run-dir")
    p.add_argument("--snapshot", default="M_final", help="M_build / M_final / M_audit_0 / full")
    p.add_argument("--all-memory", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--adapter", help="merge时adapter目录；可从训练latest.json获取")
    a = p.parse_args(argv)
    if a.limit is not None and a.limit < 1 or a.variants < 1:
        p.error("limit/variants must be positive")
    cfg = Config.load(a.config)
    if a.command == "merge":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        if not a.adapter:
            p.error("merge requires --adapter")
        base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cpu")
        model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
        model.save_pretrained(a.out, safe_serialization=True)
        AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
        return
    legacy = connect(cfg.project)
    if a.command == "check":
        import inspect
        required = {"FullMemory": legacy[0].FullMemory, "Sampler": legacy[1].Sampler,
                    "Retriever": legacy[2].Retriever, "Client.post": legacy[3].Client.post}
        report = {name: str(inspect.signature(item)) for name, item in required.items()}
        report["config"] = cfg.fingerprint()
        write(Path(a.out) / "compatibility.json", report)
        print(report)
        return
    if a.command == "prepare":
        if not a.data:
            p.error("prepare requires --data")
        print(prepare(a.data, a.out, legacy[0], a.sizes, cfg.seed))
        return
    if a.command == "relocate":
        if not a.states or not a.prepared:
            p.error("relocate requires --states exported JSONL and --prepared")
        print(relocate_states(a.states, a.prepared, a.out, legacy[0]))
        return
    if a.command == "export":
        if not a.states:
            p.error("export requires --states collector root")
        print(export_states(a.states, a.out, a.role, a.split, a.config, a.parquet))
        return
    counter = TokenBudget(cfg.tokenizer, cfg.tokenizer_revision)
    env = Environment(cfg, counter, legacy)
    if a.command == "probe-suite":
        print(len(make_suite(env, a.out, a.variants)))
        return
    if a.command == "probe":
        if not a.states or a.samples < 1:
            p.error("probe requires --states and positive --samples")
        print(run_probe(a.states, env, a.out, a.samples, a.builder_role, a.limit))
        return
    if not a.prepared:
        p.error("This command requires --prepared")
    if a.command == "evaluate":
        print(evaluate(a.prepared, a.run_dir, a.split, a.snapshot, env, a.out, a.limit, a.keys, a.all_memory))
        return
    reports = []
    for context, full_path in contexts(a.prepared, a.split, a.limit, a.keys):
        key = context["key"]
        print(f"{a.command} {key} split={context['split']} type_hint={context['question_type'] if cfg.hint_mode == 'target_type' else 'hidden'}", flush=True)
        full = legacy[0].FullMemory.load(full_path)
        try:
            if a.command == "bootstrap":
                if not a.bank or not a.collect or not a.run_dir:
                    p.error("bootstrap requires --bank, --collect and --run-dir")
                memory = read(Path(a.run_dir) / key / (a.snapshot + ".json"))
                bank = read(Path(a.bank) / key / "bank.json")
                result = {"status": "ok", **bootstrap(full, full_path, context, memory, bank, env,
                          Collector(a.collect), a.builder_role, cfg.replay_questions)}
            elif a.command == "bank":
                n = generate_bank(full, full_path, context, env, Path(a.out) / key, a.attacker_role)
                result = {"status": "ok", "questions": n}
            else:
                bank = read(Path(a.bank) / key / "bank.json") if a.bank else []
                result = run_case(full, full_path, context, env, Path(a.out) / key, mode=a.mode, bank=bank,
                    builder_role=a.builder_role, attacker_role=a.attacker_role,
                    collect=Collector(a.collect) if a.collect else None)
            reports.append({"key": key, **result})
            write(Path(a.out) / "summary.json", reports)
        except Unknown as exc:
            reports.append({"key": key, "status": "unknown", "error": str(exc)})
            write(Path(a.out) / "summary.json", reports)
            raise  # API/TLS错误不继续烧后面案例预算。


if __name__ == "__main__":
    main()
