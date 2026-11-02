#!/usr/bin/env python3
"""开训前逐个角色发一条很短的JSON模式请求，走与奖励/验证相同的调用路径（admem.common.Remote：
*_EXTRA_BODY、JSON模式、token预算、重试），把错误的模型名/key/端口/不被接受的字段在几秒内暴露出来，
而不是训练数小时后才发现。只打印状态、耗时和实际请求（不含messages与key）。不读写API缓存。

  python scripts/check_roles.py --config configs/rl_hidden_4b_v2.json [--roles JUDGE DEFENDER]

BUILDER/ATTACKER指向本地策略服务，服务没起时会报FAILED。有任何FAILED时退出码为1。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admem.common import Config, Remote, connect, messages, parse_json_object, role_models, self_judging  # noqa: E402

ROLES = ("BUILDER", "ATTACKER", "DEFENDER", "JUDGE", "VAL_ATTACKER")
SYSTEM = 'Reply with exactly this JSON object and nothing else: {"ok": true}'


def check_role(role, core, nonce=None):
    """返回 {role, status, model, base_url, latency_s, request, reply|error}。"""
    base = {"role": role, "model": os.getenv(f"{role}_MODEL", os.getenv("LLM_MODEL", "")),
            "base_url": os.getenv(f"{role}_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))}
    remote = None
    started = time.time()
    try:
        with tempfile.TemporaryDirectory() as cache:  # 空缓存：必须是真实请求
            remote = Remote(role, core, cache)
            raw = remote.complete(messages(SYSTEM, {"check": nonce or str(started)}), nonce=f"check:{started}")
            reply = parse_json_object(raw)
        return {**base, "status": "OK", "latency_s": round(time.time() - started, 2),
                "request": _request(remote), "reply": reply}
    except Exception as exc:  # noqa: BLE001 - 任何失败都要报告给用户
        return {**base, "status": "FAILED", "latency_s": round(time.time() - started, 2),
                "request": _request(remote), "error": f"{type(exc).__name__}: {exc}"[:600]}


def _request(remote):
    request = getattr(remote, "last_request", None)
    return {k: v for k, v in request.items() if k != "messages"} if request else None


def warnings(config, env=None):
    env = os.environ if env is None else env
    found = []
    if int(env.get("MAX_API_CALLS", "10000")) < 1_000_000:
        found.append("MAX_API_CALLS is small (default 10000, per process and role): a long run can stop "
                     "mid-way; export MAX_API_CALLS=100000000")
    if int(env.get("API_TIMEOUT", "120")) < 300:
        found.append("API_TIMEOUT < 300s: a busy public API can hold a request longer; export API_TIMEOUT=600")
    if int(env.get("API_RETRIES", "3")) < 4:
        found.append("API_RETRIES < 4: transient 429/5xx will fail the sample; export API_RETRIES=6")
    if config.embedding == "local" and env.get("EMBED_DEVICE") != "cpu":
        found.append("EMBED_DEVICE is not cpu: bge-m3 will compete with vLLM/FSDP for GPU memory")
    if not env.get("CUDA_VISIBLE_DEVICES"):
        found.append("CUDA_VISIBLE_DEVICES is unset: on the 2-GPU layout verl/vLLM would land on GPU 0 (Defender)")
    if env.get("LLM_JSON_MODE", "1") != "1":
        found.append("LLM_JSON_MODE != 1: judge replies are no longer constrained to a JSON object")
    for risky in self_judging():
        found.append(f"self-judging: {risky} (same model and URL); use separate services")
    for role in ("DEFENDER", "JUDGE", "VAL_ATTACKER"):
        url = env.get(f"{role}_BASE_URL", env.get("LLM_BASE_URL", ""))
        body = env.get(f"{role}_EXTRA_BODY", env.get("LLM_EXTRA_BODY", ""))
        if "thinking" in body and ("localhost" in url or "127.0.0.1" in url):
            found.append(f"{role}_EXTRA_BODY has a 'thinking' field but {role} is a local vLLM server; "
                         "unset it (DeepSeek-only field)")
    models = role_models(("DEFENDER", "VAL_ATTACKER"))
    if models["DEFENDER"][0] and models["DEFENDER"] == models["VAL_ATTACKER"]:
        found.append("VAL_ATTACKER and DEFENDER are the same model: val questions would be written by the "
                     "model that answers them")
    return found


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    a = p.parse_args(argv)
    config = Config.load(a.config)
    *_, llm = connect(config.project)
    failed = 0
    for role in a.roles:
        result = check_role(role, llm.Client)
        failed += result["status"] != "OK"
        head = f"[{result['status']}] {role}: {result['model'] or '<no model>'} @ {result['base_url']} " \
               f"({result['latency_s']}s)"
        print(head if result["status"] == "OK" else f"{head}\n    FAILED: {result['error']}")
        print("    request: " + json.dumps(result["request"], ensure_ascii=False))
    for text in warnings(config):
        print(f"[WARN] {text}")
    print(f"{len(a.roles) - failed}/{len(a.roles)} roles OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
