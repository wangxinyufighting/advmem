#!/usr/bin/env python
"""诊断建记忆阶段：为什么fallback这么多、记忆为什么这么大、faith为什么不过。

只读：run目录里每个case的checkpoint.json / M_final.json，和（可选）API缓存里的Builder与faith请求。

  python scripts/diagnose_builder.py --run runs/test_base_smoke_builder8k_per_type_v1/run \\
      --cache .cache/admem/api --since 2d --out runs/test_base_smoke_builder8k_per_type_v1/diagnosis.json

输出（终端摘要 + --out的完整JSON）：
* 窗口去向：非法输出（按原因归类）/ 合法NOOP / 写入并通过faith / 写入但faith不过；
* 记忆从哪里长大：fallback窗口与接受窗口各贡献多少token，M_final里raw块与卡片各占多少；
* 非法原因与faith原因的Top-N（数字已归一，便于合并同类）；
* --cache：Builder实际看到了多少条可编辑M_old（预算是否把它挤没了）、输出是否被截断、
  prov引用错在哪里；faith逐条判定的通过率与原因关键词。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUILDER_MARKERS = ("You are a memory editor", "You maintain a long-term memory")
FAITH_MARKER = "Check proposed memory entries against their original cited history."
FAITH_KEYWORDS = {
    "speaker/attribution": r"assistant|attribut|speaker|suggest|recommend",
    "date/time": r"\bdate|\btime|yesterday|week|month|year|temporal|chronolog",
    "number/quantity": r"number|quantit|amount|count|\bunit|price|\$",
    "not in cited source": r"not (?:mention|support|found|present|stated)|no mention|absent|does not (?:say|state|mention)|unsupported",
    "certainty/status": r"plan|intend|consider|might|uncertain|status|hedg|future|completed",
    "added detail": r"add(?:s|ed)? |extra|exclusiv|invent|fabricat|embellish",
}


def norm(reason):
    text = re.sub(r"\d+", "N", str(reason or ""))
    return re.sub(r"'[^']*'|\"[^\"]*\"", "'…'", text)[:160]


def parse_since(value):
    if not value:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", value.strip())
    if match:
        return time.time() - float(match[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]
    return datetime.fromisoformat(value).timestamp()


def share(part, whole):
    return round(part / whole, 4) if whole else None


# ------------------------------------------------------------------ run directory

def window_outcome(event):
    checks = event.get("checks") or {}
    if not checks.get("legal"):
        return "illegal"
    if event.get("ops") == 0:
        return "noop"
    faith = checks.get("faith")
    if faith is None:
        return "written_unchecked"
    return "written_faithful" if faith.get("faithful") else "written_unfaithful"


def analyze_run(run):
    cases = sorted(Path(run).rglob("checkpoint.json"))
    outcome, illegal, unfaithful, ops = Counter(), Counter(), Counter(), Counter()
    growth = defaultdict(list)
    examples = defaultdict(list)
    memory = {"raw_entries": 0, "card_entries": 0, "raw_chars": 0, "card_chars": 0}
    per_case = []
    for path in cases:
        state = json.loads(path.read_text(encoding="utf-8"))
        events = [e for e in state.get("events", []) if e.get("stage") == "build"]
        previous = 0
        counts = Counter()
        for event in events:
            kind = window_outcome(event)
            outcome[kind] += 1
            counts[kind] += 1
            checks = event.get("checks") or {}
            if kind == "illegal":
                illegal[norm(checks.get("reason"))] += 1
            if kind == "written_unfaithful":
                reason = (checks.get("faith") or {}).get("reason", "")
                unfaithful[norm(reason)] += 1
                if len(examples["unfaithful"]) < 12:
                    examples["unfaithful"].append({"case": path.parent.name, "window": event.get("window"),
                                                   "reason": reason})
            if isinstance(event.get("ops"), int):
                ops[min(event["ops"], 20)] += 1
            size = event.get("M_tokens")
            if isinstance(size, (int, float)):
                bucket = "fallback" if event.get("fallback") else kind
                growth[bucket].append(size - previous)
                previous = size
        final = path.parent / "M_final.json"
        if not final.exists():
            final = path.parent / "M_build.json"
        if final.exists():
            for entry in json.loads(final.read_text(encoding="utf-8")):
                key = "raw" if entry.get("kind") == "raw" else "card"
                memory[f"{key}_entries"] += 1
                memory[f"{key}_chars"] += len(entry.get("text", ""))
        per_case.append({"case": path.parent.name, "windows": len(events), **counts,
                         "M_tokens": previous})
    windows = sum(outcome.values())
    added = {k: {"windows": len(v), "tokens_added": sum(v), "mean_per_window": round(statistics.mean(v), 1)}
             for k, v in growth.items() if v}
    total_added = sum(v["tokens_added"] for v in added.values()) or 0
    for value in added.values():
        value["share_of_memory"] = share(value["tokens_added"], total_added)
    chars = memory["raw_chars"] + memory["card_chars"]
    memory["raw_char_share"] = share(memory["raw_chars"], chars)
    return {"cases": len(cases), "windows": windows,
            "outcome": {k: {"windows": n, "share": share(n, windows)} for k, n in outcome.most_common()},
            "illegal_reasons": illegal.most_common(15), "faith_reasons": unfaithful.most_common(15),
            "ops_per_parsed_window": dict(sorted(ops.items())), "memory_growth": added,
            "final_memory": memory, "examples": dict(examples), "per_case": per_case}


# ------------------------------------------------------------------ API cache

def iter_cache(cache, since):
    for path in Path(cache).glob("*.json"):
        try:
            if since and path.stat().st_mtime < since:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and isinstance(record.get("request"), dict):
            yield record


def builder_payload(messages):
    if not messages or not any(m in str(messages[0].get("content", "")) for m in BUILDER_MARKERS):
        return None
    try:
        payload = json.loads(messages[-1]["content"])
    except (ValueError, KeyError, TypeError):
        return None
    return payload if isinstance(payload, dict) and "mode" in payload else None


def classify_output(content, finish):
    if finish == "length":
        return "truncated (finish_reason=length)"
    text = (content or "").strip()
    if not text:
        return "empty"
    try:
        value = json.loads(text)
    except ValueError:
        if text.startswith("```"):
            return "markdown fence"
        if "<think>" in text:
            return "thinking block"
        return "not JSON"
    if not isinstance(value, dict) or set(value) != {"ops"} or not isinstance(value["ops"], list):
        return "JSON, wrong top level"
    return "JSON {ops}"


def prov_errors(payload, value):
    """ADD/UPDATE/MERGE的prov里不可引用的rid来自哪里。"""
    allowed = set(payload.get("allowed_source_rids") or [])
    visible = {r for e in payload.get("M_old") or [] for r in e.get("prov", [])}
    catalog = {item.get("id") for item in payload.get("stored_catalog") or []}
    errors = Counter()
    for op in value.get("ops", []) if isinstance(value, dict) else []:
        if not isinstance(op, dict):
            continue
        prov = op.get("prov")
        if prov is None:
            continue
        for rid in prov if isinstance(prov, list) else [prov]:
            if rid in allowed or rid in visible:
                continue
            if rid in catalog:
                errors["cites a catalog entry id"] += 1
            elif isinstance(rid, str) and re.fullmatch(r"s\d+:r\d+", rid):
                errors["round id outside the window"] += 1
            else:
                errors["invented/other"] += 1
        extra = set(op) - {"ADD": {"op", "text", "prov"}, "UPDATE": {"op", "id", "text", "prov"},
                           "MERGE": {"op", "ids", "text", "prov"}, "DELETE": {"op", "id"}}.get(op.get("op"), set(op))
        for key in extra:
            errors[f"extra field '{key}'"] += 1
        if op.get("op") not in {"ADD", "UPDATE", "MERGE", "DELETE"}:
            errors[f"unknown op {op.get('op')!r}"] += 1
    return errors


def analyze_cache(cache, since, model=None):
    calls, shapes, prov, faith_reasons = 0, Counter(), Counter(), Counter()
    editable, catalog, completion, prompt_tokens, ops = [], [], [], [], []
    starved = 0
    faith = Counter()
    for record in iter_cache(cache, since):
        request, response = record["request"], record.get("response") or {}
        messages = request.get("messages") or []
        choice = (response.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        usage = response.get("usage") or {}
        system = str(messages[0].get("content", "")) if messages else ""
        if system.startswith(FAITH_MARKER) or FAITH_MARKER in system[:200]:
            try:
                value = json.loads(content)
            except (TypeError, ValueError):
                faith["unparsed"] += 1
                continue
            ok = value.get("faithful") is True
            faith["faithful" if ok else "unfaithful"] += 1
            if not ok:
                reason = str(value.get("reason", "")).lower()
                hits = [k for k, pattern in FAITH_KEYWORDS.items() if re.search(pattern, reason)]
                faith_reasons.update(hits or ["other"])
            continue
        payload = builder_payload(messages)
        if payload is None:
            continue
        if model and request.get("model") != model:
            continue
        calls += 1
        n_old, n_catalog = len(payload.get("M_old") or []), len(payload.get("stored_catalog") or [])
        editable.append(n_old)
        catalog.append(n_catalog)
        if n_old == 0 and n_catalog > 0:
            starved += 1
        shape = classify_output(content, choice.get("finish_reason"))
        shapes[shape] += 1
        if usage.get("completion_tokens") is not None:
            completion.append(usage["completion_tokens"])
        if usage.get("prompt_tokens") is not None:
            prompt_tokens.append(usage["prompt_tokens"])
        if shape == "JSON {ops}":
            value = json.loads(content)
            ops.append(len(value["ops"]))
            prov.update(prov_errors(payload, value))

    def dist(values):
        if not values:
            return None
        values = sorted(values)
        return {"n": len(values), "mean": round(statistics.mean(values), 1), "p50": values[len(values) // 2],
                "p90": values[int(len(values) * 0.9)], "max": values[-1]}

    judged = faith["faithful"] + faith["unfaithful"]
    return {"builder_calls": calls, "output_shape": shapes.most_common(),
            "editable_entries_shown": dist(editable), "catalog_items_shown": dist(catalog),
            "calls_with_no_editable_entry_but_nonempty_memory": starved,
            "starved_share": share(starved, calls), "prompt_tokens": dist(prompt_tokens),
            "completion_tokens": dist(completion), "ops_per_output": dist(ops),
            "schema_problems_in_json_outputs": prov.most_common(),
            "faith_entries": {"judged": judged, "faithful_share": share(faith["faithful"], judged),
                              "unparsed": faith["unparsed"]},
            "faith_reason_keywords": faith_reasons.most_common()}


# ------------------------------------------------------------------ report

def explain(run, cache):
    notes = []
    outcome = {k: v["share"] for k, v in run["outcome"].items()}
    if outcome.get("illegal", 0) > 0.2:
        notes.append(f"{outcome['illegal']:.0%} of windows are illegal outputs; every one becomes a whole-window "
                     "raw fallback. See illegal_reasons; salvage=true keeps the legal operations.")
    if outcome.get("written_unfaithful", 0) > 0.2:
        notes.append(f"{outcome['written_unfaithful']:.0%} of windows wrote entries that failed faith; one failed "
                     "entry discards the whole window. salvage=true drops only that entry and re-checks it "
                     "against the whole window (wrong provenance is common).")
    growth = run["memory_growth"].get("fallback")
    if growth and (growth.get("share_of_memory") or 0) > 0.5:
        notes.append(f"{growth['share_of_memory']:.0%} of the memory tokens are raw fallback text: QA mostly "
                     "measures retrieval over raw chunks, not the Builder's cards.")
    if cache and (cache.get("starved_share") or 0) > 0.3:
        notes.append(f"In {cache['starved_share']:.0%} of Builder calls M_old was empty while memory was not: "
                     "the prompt budget went to the catalog, so UPDATE/MERGE/DELETE were impossible. "
                     "Raise builder_input_tokens and/or set budget_priority=editable.")
    shapes = dict(cache.get("output_shape", [])) if cache else {}
    if shapes.get("truncated (finish_reason=length)"):
        notes.append(f"{shapes['truncated (finish_reason=length)']} Builder outputs were truncated; raise "
                     "BUILDER_MAX_TOKENS or lower max_ops/entry_tokens.")
    return notes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="longmemeval-eval/run 输出里的run目录（含各case的checkpoint.json）")
    p.add_argument("--cache", help="config.cache下的api目录，例如 .cache/admem/api")
    p.add_argument("--since", help="只看这段时间内写入的缓存，例如 2d、12h 或ISO时间")
    p.add_argument("--builder-model", help="只统计这个served name的Builder请求")
    p.add_argument("--out", help="完整结果写到这个JSON")
    a = p.parse_args(argv)
    report = {"run": analyze_run(a.run)}
    if a.cache:
        report["cache"] = analyze_cache(a.cache, parse_since(a.since), a.builder_model)
    report["notes"] = explain(report["run"], report.get("cache"))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    brief = {k: v for k, v in report["run"].items() if k not in {"per_case", "examples"}}
    print(json.dumps({"run": brief, "cache": report.get("cache"), "notes": report["notes"]},
                     indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
