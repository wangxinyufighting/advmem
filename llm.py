"""OpenAI-compatible HTTP 客户端：缓存、重试、角色隔离；无需 SDK。"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common import digest, read_json, write_json


class ModelError(RuntimeError):
    """模型/网络错误必须记为 ERROR，不能算成答错或空列表。"""


class Client:
    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY",
                 cache_dir: str | Path = ".cache/api", max_calls: int = 10000):
        if not model:
            raise ValueError("请配置模型名，例如 LLM_MODEL 或 ATTACKER_MODEL")
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.cache_dir = Path(cache_dir)
        self.max_calls, self.calls, self.cache_hits = max_calls, 0, 0

    @classmethod
    def from_env(cls, role: str, cache_dir: str | Path = ".cache/api") -> "Client":
        # 各角色可用不同模型/服务；没有覆盖时共用 LLM_*。
        return cls(os.getenv(f"{role}_MODEL", os.getenv("LLM_MODEL", "")),
                   os.getenv(f"{role}_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")),
                   os.getenv(f"{role}_API_KEY", os.getenv("LLM_API_KEY", "EMPTY")), cache_dir,
                   int(os.getenv("MAX_API_CALLS", "10000")))

    def post(self, endpoint: str, payload: dict, nonce: str = "") -> dict:
        # nonce 区分同一 prompt 的独立采样，但不修改发给模型的内容。
        key = digest([self.base_url, endpoint, payload, nonce])
        path = self.cache_dir / (key + ".json")
        if path.exists():
            self.cache_hits += 1
            return read_json(path)["response"]
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base_url + endpoint, raw, method="POST", headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"}) 
        
        last = None
        for attempt in range(3):
            if self.calls >= self.max_calls:
                raise ModelError(f"{self.model} 达到 API 调用预算 {self.max_calls}")
            self.calls += 1
            try:
                with urllib.request.urlopen(req, timeout=int(os.getenv("API_TIMEOUT", "120"))) as res:
                    result = json.loads(res.read())
                # 缓存含实验原文及响应，但绝不含 API key。
                write_json(path, {"model": self.model, "base_url": self.base_url,
                                  "request": payload, "response": result})
                return result
            except urllib.error.HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")[:1000]
                last = f"HTTP {exc.code}: {message}"
                if exc.code != 429 and exc.code < 500:
                    raise ModelError(last) from exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last = str(exc)
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise ModelError(f"模型请求失败：{last}")

    def json(self, system: str, data: Any, *, temperature: float = 0, nonce: str = "") -> dict:
        content = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        payload = {"model": self.model, "messages": [
            {"role": "system", "content": system + '\nReturn exactly one valid JSON object.'},
            {"role": "user", "content": content}], "temperature": temperature,
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096"))}
        if os.getenv("LLM_JSON_MODE", "1") == "1":
            payload["response_format"] = {"type": "json_object"}
        # 某些服务使用 max_completion_tokens 或额外参数，可显式覆盖/删掉默认字段。
        for key, value in json.loads(os.getenv("LLM_EXTRA_BODY", "{}")).items():
            if value is None:
                payload.pop(key, None)
            else:
                payload[key] = value
        for attempt in range(2):
            result = self.post("/chat/completions", payload, nonce=f"{nonce}:json{attempt}")
            try:
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("输出被长度上限截断")
                text = choice["message"]["content"].strip()
                if text.startswith("```"):
                    text = "\n".join(text.splitlines()[1:-1])
                obj = json.loads(text)
                if not isinstance(obj, dict):
                    raise ValueError("顶层必须为 JSON object")
                return obj
            except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
                if attempt == 1:
                    raise ModelError(f"模型 JSON 不合法：{exc}") from exc
        raise AssertionError("unreachable")

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self.post("/embeddings", {"model": self.model, "input": texts,
                                          "encoding_format": "float"})
        try:
            rows = sorted(result["data"], key=lambda row: row["index"])
            if [r["index"] for r in rows] != list(range(len(texts))):
                raise ValueError("embedding 行号/数量不匹配")
            return [r["embedding"] for r in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelError(f"embedding 响应不合法：{exc}") from exc
