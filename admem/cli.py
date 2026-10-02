"""python -m admem.cli：check / prepare / bank / run / bootstrap / probe-suite / probe / export / evaluate / longmemeval-eval / merge。"""
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


def _builder_case_metrics(case_out):
    """Summarize Builder-only checks recorded by run_case(mode='build')."""
    checkpoint = Path(case_out) / "checkpoint.json"
    if not checkpoint.exists():
        return {"status": "missing_checkpoint", "windows": 0}
    state = read(checkpoint)
    events = [event for event in state.get("events", []) if event.get("stage") == "build"]
    legal = 0
    faithful = 0
    fallback = 0
    for event in events:
        checks = event.get("checks") or {}
        if checks.get("legal") is True:
            legal += 1
            faith = checks.get("faith")
            if isinstance(faith, dict) and faith.get("faithful") is True:
                faithful += 1
        if event.get("fallback") is True:
            fallback += 1
    total = len(events)
    return {
        "status": "ok",
        "windows": total,
        "json_valid": legal,
        "faithful": faithful,
        "fallback_windows": fallback,
        "builder_json_rate": legal / total if total else None,
        "builder_faith_rate": faithful / total if total else None,
    }


def _json_case_list(path):
    """Read a LongMemEval split JSON produced by prepare_longmemeval_cases.py."""
    value = read(path)
    if isinstance(value, dict):
        for key in ("data", "cases", "instances"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("--cases must be a JSON list of LongMemEval case objects")
    required = {"haystack_session_ids", "haystack_dates", "haystack_sessions"}
    if any(not required <= set(row) for row in value):
        raise ValueError(
            "--cases must contain raw LongMemEval cases with haystack_session_ids, "
            "haystack_dates and haystack_sessions; do not pass *_builder.jsonl"
        )
    return value


def _external_label_index(path):
    """Index labels without exposing them to the Builder phase."""
    result = {}
    for row in rows(path):
        if not isinstance(row, dict):
            raise ValueError("--labels must contain JSON objects, one per line")
        qid = row.get("question_id") or row.get("case_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError("Each --labels row needs question_id or case_id")
        if qid in result:
            raise ValueError(f"Duplicate label for {qid}")
        result[qid] = row
    if not result:
        raise ValueError("--labels is empty")
    return result


def _history_for_builder(case):
    """Whitelist history and disambiguate duplicate source session IDs.

    Some LongMemEval exports reuse a session ID inside one case. FullMemory
    needs unique internal keys, so only the duplicate key is suffixed; dates,
    messages, ordering, and all message text remain unchanged.
    """
    seen = {}
    used = set()
    session_ids = []
    for value in case["haystack_session_ids"]:
        base = str(value)
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        candidate = base if occurrence == 0 else f"{base}__duplicate_{occurrence}"
        while candidate in used:
            occurrence += 1
            seen[base] = occurrence + 1
            candidate = f"{base}__duplicate_{occurrence}"
        used.add(candidate)
        session_ids.append(candidate)
    return {"haystack_session_ids": session_ids,
            "haystack_dates": case["haystack_dates"],
            "haystack_sessions": case["haystack_sessions"]}


def _external_question(case, label):
    """Convert custom split labels to Environment's private question schema."""
    case_qid = str(case.get("question_id", ""))
    label_qid = str(label.get("question_id") or label.get("case_id") or "")
    if label_qid != case_qid:
        raise ValueError(f"Label question_id {label_qid!r} does not match case {case_qid!r}")
    question = label.get("q", label.get("question"))
    answer = label.get("a", label.get("answer"))
    qtype = label.get("type", label.get("question_type"))
    qdate = label.get("question_date")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Missing question label for {case_qid}")
    if not isinstance(qtype, str) or not qtype.strip() or not isinstance(qdate, str) or not qdate.strip():
        raise ValueError(f"Invalid question type/date label for {case_qid}")
    if qtype != case.get("question_type") or qdate != case.get("question_date"):
        raise ValueError(f"Label type/date does not match case {case_qid}")
    # Native preparation uses str(answer), including for list/dict answers.
    if not isinstance(answer, str):
        answer = str(answer)
    evidence = label.get("E", [])
    if not isinstance(evidence, list):
        evidence = []
    return {"q": question, "a": answer, "type": qtype, "question_date": qdate,
            "E": evidence, "abstention": bool(label.get("abstention", case_qid.endswith("_abs"))),
            "question_id": case_qid}


def _evaluate_external(cases, labels, run_dir, snapshot, env, out, all_memory=False):
    """Evaluate memories built from an external split using an external labels JSONL."""
    details = []
    out = Path(out)
    for record in cases:
        key, full_path = record["key"], Path(record["full_path"])
        qid = record["question_id"]
        label = labels.get(qid)
        if label is None:
            # Some older splitters used case_id as the sole identifier.
            label = labels.get(str(record.get("case_id", "")))
        if label is None:
            raise ValueError(f"No label for case {qid}")
        q = _external_question(record, label)
        try:
            if snapshot == "full":
                full = env.memory_module.FullMemory.load(full_path)
                memory = [{"id": r, "text": f"Recorded on {s['date']}. {s['text']}",
                           "prov": [r], "kind": "raw"}
                          for r in full.ordered(full.rounds) for s in [neutral_round(full, r)]]
            else:
                memory = read(Path(run_dir) / key / (snapshot + ".json"))
            result = env.answer(memory, q, all_memory=all_memory, abstention=q["abstention"])
            details.append({"key": key, "question_id": q["question_id"], "type": q["type"],
                "group": "abstention" if q["abstention"] else q["type"], "status": "ok",
                "M_tokens": text_size(memory, env.counter), "entries": len(memory),
                "raw_entries": sum(e.get("kind") == "raw" for e in memory),
                "raw_token_fraction": text_size([e for e in memory if e.get("kind") == "raw"], env.counter) /
                                      max(1, text_size(memory, env.counter)), **result})
        except (Unknown, OSError, ValueError) as exc:
            details.append({"key": key, "question_id": q["question_id"], "group":
                            "abstention" if q["abstention"] else q["type"],
                            "status": "error", "error": str(exc)})
    write_rows(out / "cases.jsonl", details)
    grouped = defaultdict(list)
    for row in details:
        grouped[row["group"]].append(row)
        grouped["ALL_MICRO"].append(row)
    summary = []
    for typ, group in grouped.items():
        judged = [row for row in group if row["status"] == "ok"]
        correct = sum(row["correct"] for row in judged)
        unknown = len(group) - len(judged)
        summary.append({"group": typ, "selected": len(group), "judged": len(judged),
            "correct": correct, "unknown": unknown,
            "accuracy_completed": correct / len(judged) if judged else None,
            "lower_all": correct / len(group), "upper_all": (correct + unknown) / len(group),
            "mean_M_tokens": sum(row["M_tokens"] for row in judged) / len(judged) if judged else None})
    write(out / "summary.json", {"snapshot": snapshot, "all_memory": all_memory,
          "judge": "admem diagnostic; not official execution", "groups": summary})
    write_rows(out / "predictions.jsonl", [{"question_id": row["question_id"],
                                             "hypothesis": row["answer"]}
                                            for row in details if row["status"] == "ok"])
    return summary


def longmemeval_eval_external(cases_path, labels_path, split, snapshot, env, out, limit,
                              keys=None, builder_role="BUILDER", all_memory=False):
    """Run Builder on a raw split JSON, then score it with a separate labels JSONL.

    ``case`` is reduced to FullMemory before it reaches ``run_case``.  The case-level
    question/answer and all message labels therefore remain outside the Builder prompt.
    """
    out = Path(out)
    if (out / "summary.json").exists():
        raise ValueError("Output directory already has summary.json; use a new directory")
    case_values = _json_case_list(cases_path)
    labels = _external_label_index(labels_path)
    run_dir, eval_dir = out / "run", out / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    for index, case in enumerate(case_values):
        qid = case.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"Case {index} has no question_id")
        if qid in seen:
            raise ValueError(f"Duplicate case question_id {qid}")
        seen.add(qid)
        history = _history_for_builder(case)
        # Keep q/a and message-level evaluation labels outside the object passed
        # to the Builder pipeline. FullMemory.build also strips these fields,
        # but making the boundary explicit prevents future prompt leakage.
        full = env.memory_module.FullMemory.build(history)
        key = f"c{index:04d}"
        case_dir = run_dir / key
        full_path = case_dir / "full.json"
        full.save(full_path)
        context = {"key": key, "question_type": case.get("question_type", ""),
                   "question_date": case.get("question_date", ""), "split": split,
                   "full_hash": full.fingerprint}
        records.append({"key": key, "question_id": qid,
                        "case_id": str(case.get("case_id", "")),
                        "question_type": case.get("question_type", ""),
                        "question_date": case.get("question_date", ""),
                        "context": context, "full": full,
                        "full_path": str(full_path.resolve())})
    missing = sorted(record["question_id"] for record in records
                     if record["question_id"] not in labels
                     and record.get("case_id", "") not in labels)
    if missing:
        raise ValueError(f"--labels missing {len(missing)} case(s), first: {missing[0]}")
    valid_label_keys = set(seen)
    valid_label_keys.update(record["case_id"] for record in records if record.get("case_id"))
    extra = sorted(set(labels) - valid_label_keys)
    if extra:
        raise ValueError(f"--labels contains {len(extra)} unknown case(s), first: {extra[0]}")
    if keys:
        wanted = set(keys)
        available = {value for record in records for value in
                     (record["key"], record["question_id"], record.get("case_id", "")) if value}
        unknown = sorted(wanted - available)
        if unknown:
            raise ValueError(f"Unknown --keys value for --cases: {unknown[0]}")
    selected = [record for record in records if not keys or
                record["key"] in keys or record["question_id"] in keys or
                record.get("case_id", "") in keys]
    if limit:
        selected = selected[:limit]
    # Validate label joins before spending Builder/API calls. The labels are
    # still kept out of the history/context objects passed to run_case.
    for record in selected:
        label = labels.get(record["question_id"]) or labels.get(record.get("case_id", ""))
        if label is None:
            raise ValueError(f"No label for case {record['question_id']}")
        _external_question(record, label)
    reports, builder_cases = [], []
    for record in selected:
        key, full, context = record["key"], record["full"], record["context"]
        print(f"longmemeval-eval {key} split={split} type_hint="
              f"{context['question_type'] if env.cfg.hint_mode == 'target_type' else 'hidden'}", flush=True)
        try:
            result = run_case(full, record["full_path"], context, env, run_dir / key,
                              mode="build", bank=[], builder_role=builder_role)
            reports.append({"key": key, "question_id": record["question_id"], **result})
            builder_cases.append({"key": key, "question_id": record["question_id"],
                                  "question_type": context["question_type"], **_builder_case_metrics(run_dir / key)})
        except Unknown as exc:
            report = {"key": key, "question_id": record["question_id"],
                      "status": "unknown", "error": str(exc)}
            reports.append(report)
            builder_cases.append({"key": key, "question_id": record["question_id"],
                                  "question_type": context["question_type"], "status": "unknown",
                                  "error": str(exc)})
            write(run_dir / "summary.json", reports)
            raise
        write(run_dir / "summary.json", reports)
    total_windows = sum(row.get("windows", 0) for row in builder_cases)
    total_legal = sum(row.get("json_valid", 0) for row in builder_cases)
    total_faithful = sum(row.get("faithful", 0) for row in builder_cases)
    builder_summary = {"split": split, "builder_role": builder_role, "cases": len(builder_cases),
        "windows": total_windows, "json_valid": total_legal, "faithful": total_faithful,
        "fallback_windows": sum(row.get("fallback_windows", 0) for row in builder_cases),
        "builder_json_rate": total_legal / total_windows if total_windows else None,
        "builder_faith_rate": total_faithful / total_windows if total_windows else None,
        "by_case": builder_cases}
    write(out / "builder_summary.json", builder_summary)
    qa_summary = _evaluate_external(selected, labels,
                                    run_dir, snapshot, env, eval_dir, all_memory)
    summary = {"split": split, "builder": builder_summary, "qa": qa_summary,
               "run_dir": str(run_dir), "eval_dir": str(eval_dir), "snapshot": snapshot,
               "all_memory": all_memory, "cases": str(Path(cases_path).resolve()),
               "labels": str(Path(labels_path).resolve())}
    write(out / "summary.json", summary)
    return summary


def longmemeval_eval(prepared, split, snapshot, env, out, limit, keys,
                     builder_role="BUILDER", all_memory=False):
    """Run the existing Builder over a prepared LongMemEval split, then score QA.

    The Builder phase only receives the prepared context/full history. Official
    question/answer labels are read later by ``evaluate`` from private_eval.jsonl,
    so they cannot leak into the Builder prompt. ``mode='build'`` deliberately
    disables the attacker/audit loop and evaluates the selected Builder policy.
    """
    out = Path(out)
    prepared = Path(prepared)
    required = [prepared / "manifest.json", prepared / "private_eval.jsonl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(
            "--prepared must point to a native admem prepared directory "
            "(missing: " + ", ".join(missing) + "). Run `python -m admem.cli prepare "
            "--data <LongMemEval.json> --out <prepared>` first; the "
            "prepare_longmemeval_cases.py SFT split is not an evaluation input."
        )
    run_dir = out / "run"
    eval_dir = out / "eval"
    if (out / "summary.json").exists():
        raise ValueError("Output directory already has summary.json; use a new directory")
    run_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    builder_cases = []
    for context, full_path in contexts(prepared, split, limit, keys):
        key = context["key"]
        print(f"longmemeval-eval {key} split={split} "
              f"type_hint={context['question_type'] if env.cfg.hint_mode == 'target_type' else 'hidden'}",
              flush=True)
        full = env.memory_module.FullMemory.load(full_path)
        case_out = run_dir / key
        try:
            result = run_case(full, full_path, context, env, case_out,
                              mode="build", bank=[], builder_role=builder_role)
            metrics = _builder_case_metrics(case_out)
            reports.append({"key": key, **result})
            builder_cases.append({"key": key, "question_type": context["question_type"], **metrics})
        except Unknown as exc:
            report = {"key": key, "status": "unknown", "error": str(exc)}
            reports.append(report)
            builder_cases.append({"key": key, "question_type": context["question_type"],
                                  "status": "unknown", "error": str(exc)})
            write(run_dir / "summary.json", reports)
            raise
        write(run_dir / "summary.json", reports)

    total_windows = sum(row.get("windows", 0) for row in builder_cases)
    total_legal = sum(row.get("json_valid", 0) for row in builder_cases)
    total_faithful = sum(row.get("faithful", 0) for row in builder_cases)
    builder_summary = {
        "split": split,
        "builder_role": builder_role,
        "cases": len(builder_cases),
        "windows": total_windows,
        "json_valid": total_legal,
        "faithful": total_faithful,
        "fallback_windows": sum(row.get("fallback_windows", 0) for row in builder_cases),
        "builder_json_rate": total_legal / total_windows if total_windows else None,
        "builder_faith_rate": total_faithful / total_windows if total_windows else None,
        "by_case": builder_cases,
    }
    write(out / "builder_summary.json", builder_summary)

    qa_summary = evaluate(prepared, run_dir, split, snapshot, env, eval_dir, limit, keys, all_memory)
    summary = {"split": split, "builder": builder_summary, "qa": qa_summary,
               "run_dir": str(run_dir), "eval_dir": str(eval_dir), "snapshot": snapshot,
               "all_memory": all_memory}
    write(out / "summary.json", summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["check", "prepare", "bank", "run", "bootstrap", "probe-suite", "probe", "export", "relocate", "evaluate", "longmemeval-eval", "merge"])
    p.add_argument("--config", default="configs/type_aware.json")
    p.add_argument("--data")
    p.add_argument("--prepared")
    p.add_argument("--cases", help="longmemeval-eval raw split JSON (with --labels)")
    p.add_argument("--labels", help="longmemeval-eval labels JSONL (with --cases)")
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["train", "val", "valid", "test"], default="train")
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
    if a.split == "valid":
        a.split = "val"
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
    if a.command == "longmemeval-eval" and a.cases:
        if a.prepared:
            p.error("longmemeval-eval accepts either --prepared or --cases/--labels, not both")
        if not a.labels:
            p.error("longmemeval-eval with --cases requires --labels")
        external_split = a.split
        stem = Path(a.cases).stem.lower()
        if a.split == "train" and stem in {"test", "valid", "val"}:
            external_split = "val" if stem == "valid" else stem
        print(longmemeval_eval_external(a.cases, a.labels, external_split, a.snapshot, env, a.out,
                                        a.limit, a.keys, builder_role=a.builder_role,
                                        all_memory=a.all_memory))
        return
    if a.command == "longmemeval-eval" and a.labels:
        p.error("longmemeval-eval with --labels also requires --cases")
    if not a.prepared:
        p.error("This command requires --prepared")
    if a.command == "evaluate":
        print(evaluate(a.prepared, a.run_dir, a.split, a.snapshot, env, a.out, a.limit, a.keys, a.all_memory))
        return
    if a.command == "longmemeval-eval":
        print(longmemeval_eval(a.prepared, a.split, a.snapshot, env, a.out, a.limit, a.keys,
                               builder_role=a.builder_role, all_memory=a.all_memory))
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
