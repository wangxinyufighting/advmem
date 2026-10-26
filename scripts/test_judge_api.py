#!/usr/bin/env python3
"""独立诊断 JUDGE 接口；仅使用标准库，不加载模型、不读写项目缓存。"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request


def setting(name, default=""):
    return os.getenv("JUDGE_" + name, os.getenv("LLM_" + name, default))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--input-chars", type=int, default=0,
                        help="增加合成输入字符数，排查长输入；字符数不等于 token 数")
    args = parser.parse_args()
    if args.timeout <= 0 or args.input_chars < 0:
        parser.error("timeout 必须大于 0，input-chars 不能为负")
    model = setting("MODEL")
    base = setting("BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = setting("API_KEY", "EMPTY")
    if not model:
        parser.error("请先设置 JUDGE_MODEL 或 LLM_MODEL")
    try:
        extras = json.loads(setting("EXTRA_BODY", "{}"))
        budget = int(setting("MAX_TOKENS", "4096"))
        if not isinstance(extras, dict) or budget <= 0:
            raise ValueError("EXTRA_BODY 必须是 JSON 对象，MAX_TOKENS 必须为正数")
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    # 和项目 Remote.complete 一样：null 删除参数，其余覆盖默认值。
    configured = {}
    if os.getenv("LLM_JSON_MODE", "1") == "1":
        configured["response_format"] = {"type": "json_object"}
    configured.update(extras)
    tests = [
        ("基础请求", {}),
        ("仅关闭思考", {"thinking": {"type": "disabled"}}),
        ("仅 JSON 模式", {"response_format": {"type": "json_object"}}),
        ("关闭思考 + JSON 模式", {
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }),
        ("当前项目环境配置", configured),
    ]
    prompt = 'Return only this JSON object: {"ok": true}'
    if args.input_chars:
        filler = ("This is synthetic context for an API diagnostic. " *
                  (args.input_chars // 48 + 1))[:args.input_chars]
        prompt = "Context:\n" + filler + "\n\n" + prompt

    def safe(value):
        text = str(value)
        return text.replace(key, "[REDACTED]") if key else text

    print(safe(f"model={model}\nbase_url={base}"))
    print(f"max_tokens={budget}; synthetic_input_chars={args.input_chars}")
    results = []
    for name, overrides in tests:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": budget}
        for field, value in overrides.items():
            if value is None:
                body.pop(field, None)
            else:
                body[field] = value
        req = urllib.request.Request(
            base + "/chat/completions", data=json.dumps(body).encode(),
            # 与 llm.Client.post 保持一致，避免客户端标识差异干扰对照。
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0"},
        )
        print(f"\n=== {name} ===", flush=True)
        start = time.monotonic()
        passed = False
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                data = json.load(response)
            if data.get("error"):
                raise ValueError("接口返回错误：" + json.dumps(data["error"], ensure_ascii=False))
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content")
            finish = choice.get("finish_reason")
            print("finish_reason:", finish)
            print("content:", safe(content)[:1000])
            print("usage:", safe(json.dumps(data.get("usage"), ensure_ascii=False)))
            print("reasoning_content 非空:", bool(message.get("reasoning_content")))
            try:
                parsed = json.loads(content) if isinstance(content, str) else None
                passed = isinstance(parsed, dict) and parsed.get("ok") is True and finish != "length"
            except json.JSONDecodeError:
                pass
            if not passed:
                print("请求已返回，但未得到完整的预期 JSON。")
        except urllib.error.HTTPError as exc:
            print("HTTP", exc.code, safe(exc.read().decode(errors="replace"))[:2000])
        except Exception as exc:
            print(type(exc).__name__ + ":", safe(exc)[:2000])
        print(f"结果={'PASS' if passed else 'FAIL'}; 耗时={time.monotonic() - start:.1f}s", flush=True)
        results.append((name, passed))
    print("\n=== 汇总 ===")
    for name, passed in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print("PASS 仅说明本次请求返回了预期 JSON，不保证真实长请求正常或思考参数实际生效。")
    return 0 if all(passed for _, passed in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
