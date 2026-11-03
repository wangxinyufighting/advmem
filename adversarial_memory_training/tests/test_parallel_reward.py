"""ADMEM_PARALLEL_CALLS只改变等待时间：并发与顺序计算的奖励逐字相同，且并发时确实同时调用。"""
import json
import threading
import time

import pytest

from admem import environment as envmod
from admem.common import pmap, run_together
from test_reward_prompt_v2 import ADD, bstate, builder_case, q

DELAY = 0.05


def slow_world(env, monkeypatch):
    """answer/faith/保留与覆盖检查都换成带延迟的确定性替身，并记录最大并发数。"""
    active, peak, lock = [0], [0], threading.Lock()

    def slow(value):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(DELAY)
        with lock:
            active[0] -= 1
        return value

    env.cfg.builder_reward = "v2"
    monkeypatch.setattr(env, "answer", lambda memory, question, **kw: slow(
        {"correct": any(question["a"] in e["text"] for e in memory), "visible_ids": [], "answer": ""}))
    monkeypatch.setattr(env, "_faith_one", lambda full, entry: slow({"faithful": "Biscuit" in entry["text"],
                                                                     "reason": "stub"}))
    monkeypatch.setattr(envmod, "assess_edits", lambda *a, **k: slow({
        "retention_pass": True, "information_loss": 0.0, "redundant_ops": [], "redundancy_ratio": 0.0,
        "retention_checks": []}))
    monkeypatch.setattr(envmod, "source_coverage", lambda *a, **k: slow({"facts": 2, "new": 2, "stored": 1,
                                                                         "ratio": 0.5}))
    return peak


def score(env, monkeypatch, workers):
    monkeypatch.setenv("ADMEM_PARALLEL_CALLS", str(workers))
    full = builder_case()
    tests = [q("What is my dog's name?", ("s2:r1",), a="Biscuit")] + \
            [q(f"Old {i}?", ("s1:r1",), a=f"Old fact {i}") for i in range(7)]
    action = json.dumps({"ops": json.loads(ADD)["ops"] + [
        {"op": "ADD", "text": "The user has a cat.", "prov": ["s2:r1"]}]})
    started = time.time()
    result = env.builder_score(full, bstate(full, tests), action)
    return result, time.time() - started


def test_parallel_reward_equals_sequential_and_overlaps_calls(env, monkeypatch):
    peak = slow_world(env, monkeypatch)
    sequential, slow_time = score(env, monkeypatch, 1)
    assert peak[0] == 1
    parallel, fast_time = score(env, monkeypatch, 8)
    drop = {"proposed"}
    assert {k: v for k, v in parallel.items() if k not in drop} == {k: v for k, v in sequential.items() if k not in drop}
    assert peak[0] > 2 and fast_time < slow_time / 2


def test_pmap_keeps_order_and_raises_the_first_error_in_order(monkeypatch):
    monkeypatch.setenv("ADMEM_PARALLEL_CALLS", "4")
    assert pmap(lambda x: x * x, range(10)) == [x * x for x in range(10)]

    def fail(x):
        time.sleep(0.01 * (5 - x))
        if x in (1, 3):
            raise ValueError(f"bad {x}")
        return x

    with pytest.raises(ValueError, match="bad 1"):
        pmap(fail, range(5))
    assert run_together(lambda: 1, lambda: 2) == [1, 2]


def test_case_parallel_validation_gives_the_same_metrics(case, env, tmp_path):
    from admem.validation import builder_metrics
    from test_rl_training import prepared_val
    prepared = prepared_val(case, env, tmp_path)
    env.clients["BUILDER_EVAL"] = env.clients["BUILDER"]
    env.cfg.faith_gate = True
    one = builder_metrics(env, prepared, "val", tmp_path / "e1", val_bank=tmp_path / "val_bank", workers=1)
    two = builder_metrics(env, prepared, "val", tmp_path / "e2", val_bank=tmp_path / "val_bank", workers=2)
    assert one == two


def test_identical_concurrent_requests_are_sent_once(tmp_path, monkeypatch):
    from admem.common import Remote

    class CachingClient:
        model, base_url, max_calls, calls = "m", "http://x", 100, 0

        def __init__(self):
            self.cache, self.sent, self.lock = {}, 0, threading.Lock()

        def post(self, endpoint, body, nonce=""):
            key = json.dumps([body, nonce], sort_keys=True)
            with self.lock:
                if key in self.cache:
                    return self.cache[key]
            time.sleep(0.05)
            with self.lock:
                self.sent += 1
                self.cache[key] = {"choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]}
                return self.cache[key]

    client = CachingClient()
    core = type("Core", (), {"from_env": classmethod(lambda cls, role, cache: client)})
    remote = Remote("JUDGE", core, str(tmp_path))
    prompt = [{"role": "user", "content": "say json"}]
    monkeypatch.setenv("ADMEM_PARALLEL_CALLS", "8")
    answers = pmap(lambda i: remote.complete(prompt, "same-nonce"), range(8))
    assert answers == ['{"ok": true}'] * 8 and client.sent == 1
    pmap(lambda i: remote.complete(prompt, f"nonce-{i}"), range(4))
    assert client.sent == 5


def test_rl_loop_reports_verl_stage_timings(tmp_path):
    import importlib.util
    from pathlib import Path
    from admem.common import write_rows
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("rl_loop", root / "scripts" / "rl_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log = tmp_path / "log.jsonl"
    write_rows(log, [{"step": i, "data": {"timing_s/gen": 100.0 + i, "timing_s/update_actor": 30.0}}
                     for i in range(1, 3)])
    assert module.train_metrics(log)["timing_s_mean"] == {"gen": 101.5, "update_actor": 30.0}


def test_probe_workers_give_identical_records_and_overlap_samples(env, tmp_path, monkeypatch):
    """--workers：同一状态的样本先并发生成、并发打分（写入缓存），记录与逐个执行时完全相同。"""
    from admem.common import rows, write_rows
    from admem.probes import make_suite, run_probe
    from test_probe_reward_diagnostics import verdict

    records = make_suite(env, tmp_path / "suite", variants=1)[:2]
    states = tmp_path / "states.jsonl"
    write_rows(states, records)
    generated, scored, lock = {}, {}, threading.Lock()

    def complete(prompt, nonce, temperature=0.0, **kw):  # 模拟带缓存的API：同一nonce只真正生成一次
        with lock:
            if nonce in generated:
                return generated[nonce]
        time.sleep(DELAY)
        text = json.dumps({"ops": [], "n": int(nonce[-2:], 16) % 3})
        with lock:
            generated[nonce] = text
        return text

    def score(full, state, raw):  # 模拟带缓存的judge
        with lock:
            if raw in scored:
                return dict(scored[raw])
        time.sleep(DELAY)
        value = verdict(reward=float(json.loads(raw)["n"]), effective_reward=float(json.loads(raw)["n"]))
        with lock:
            scored[raw] = value
        return dict(value)

    monkeypatch.setattr(env.policy("BUILDER"), "complete", complete)
    monkeypatch.setattr(env, "builder_score", score)
    timing = {"generation_seconds", "scoring_seconds"}

    def run(workers, out):
        generated.clear(), scored.clear()
        started = time.time()
        run_probe(states, env, tmp_path / out, samples=6, workers=workers)
        return [{k: v for k, v in r.items() if k not in timing} for r in rows(tmp_path / out / "samples.jsonl")], \
            time.time() - started

    sequential, slow = run(1, "seq")
    parallel, fast = run(6, "par")
    assert parallel == sequential and len(sequential) == 12
    assert fast < slow / 2
