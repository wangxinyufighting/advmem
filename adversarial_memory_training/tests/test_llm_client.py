import json

import urllib.request

from common import digest, read_json
from llm import Client


class _Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


def _request_cache_path(client, tmp_path, payload, nonce):
    return tmp_path / (digest([client.base_url, "/chat/completions", payload, nonce]) + ".json")


def test_post_ignores_cached_provider_error(tmp_path, monkeypatch):
    client = Client("fake", "http://fake/v1", "EMPTY", tmp_path)
    payload = {"model": "fake", "messages": [], "max_tokens": 8}
    nonce = "case"
    path = _request_cache_path(client, tmp_path, payload, nonce)
    path.write_text(json.dumps({"response": {
        "error": {"message": "Provider returned an empty response", "code": 502}
    }}))
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: _Response({
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
    }))

    result = client.post("/chat/completions", payload, nonce)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert client.cache_hits == 0
    assert read_json(path)["response"]["choices"]


def test_post_retries_http_200_provider_error_without_caching(tmp_path, monkeypatch):
    client = Client("fake", "http://fake/v1", "EMPTY", tmp_path)
    payload = {"model": "fake", "messages": [], "max_tokens": 8}
    responses = [
        {"error": {"message": "upstream unavailable", "code": 502,
                    "metadata": {"error_type": "provider_unavailable"}}},
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    ]
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout: _Response(responses.pop(0)))
    monkeypatch.setattr("llm.time.sleep", lambda seconds: None)

    result = client.post("/chat/completions", payload, "case")

    assert result["choices"]
    assert client.calls == 2
    assert list(tmp_path.glob("*.json"))
