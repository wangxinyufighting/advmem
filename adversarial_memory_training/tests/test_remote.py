import json

import pytest

from admem.common import Remote, Unknown


class _FakeClient:
    model = "fake-judge"
    base_url = "http://fake"
    max_calls = 10

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.calls = 0

    def post(self, endpoint, body, nonce=""):
        self.calls += 1
        self.requests.append((endpoint, body, nonce))
        return self.responses.pop(0)


class _FakeCore:
    client = None

    @classmethod
    def from_env(cls, role, cache):
        return cls.client


def _clear_llm_env(monkeypatch):
    for name in [
        "JUDGE_MAX_TOKENS", "LLM_MAX_TOKENS", "JUDGE_EXTRA_BODY",
        "LLM_EXTRA_BODY", "JUDGE_JSON_RETRY_MAX_TOKENS",
        "LLM_JSON_RETRY_MAX_TOKENS", "LLM_JSON_MODE",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_json_retry_expands_truncated_judge_budget(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    client = _FakeClient([
        {"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": '{"correct": true}'}}]},
    ])
    _FakeCore.client = client

    result = Remote("JUDGE", _FakeCore, tmp_path / "api").json(
        [{"role": "user", "content": "judge"}], "case")

    assert result == {"correct": True}
    assert client.requests[0][1]["max_tokens"] == 4096
    assert client.requests[1][1]["max_tokens"] == 8192


def test_complete_failure_reports_prompt_tokens(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    client = _FakeClient([{"choices": []}])
    _FakeCore.client = client

    with pytest.raises(Unknown) as exc:
        Remote("JUDGE", _FakeCore, tmp_path / "api").complete(
            [{"role": "user", "content": "judge"}], "case", prompt_tokens=12345)

    message = str(exc.value)
    assert "prompt_tokens=12345" in message and "max_tokens=4096" in message


def test_trace_request_uses_effective_body_and_clears_stale_request(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    effective = [{"role": "user", "content": "overridden input"}]
    monkeypatch.setenv("JUDGE_EXTRA_BODY", json.dumps({"messages": effective, "temperature": 0.3}))
    client = _FakeClient([{"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}])
    _FakeCore.client = client
    remote = Remote("JUDGE", _FakeCore, tmp_path / "api")
    remote.complete([{"role": "user", "content": "original input"}], "first")
    assert remote.last_request == client.requests[0][1]
    assert remote.last_request["messages"][1:] == effective
    assert "JSON" in remote.last_request["messages"][0]["content"]
    client.calls = client.max_calls
    with pytest.raises(Unknown, match="MAX_API_CALLS"):
        remote.complete([], "second")
    assert remote.last_request is None


@pytest.mark.parametrize("extra,prompt_text,adds_reminder", [
    ({}, "Judge the answer.", True),
    ({}, "Return Json.", False),
    ({"response_format": None}, "Judge the answer.", False),
])
def test_json_mode_message_requirement(tmp_path, monkeypatch, extra, prompt_text, adds_reminder):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("JUDGE_EXTRA_BODY", json.dumps(extra))
    client = _FakeClient([{"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}])
    _FakeCore.client = client
    prompt = [{"role": "user", "content": prompt_text}]
    Remote("JUDGE", _FakeCore, tmp_path / "api").complete(prompt, "case")
    sent = client.requests[0][1]["messages"]
    assert prompt == [{"role": "user", "content": prompt_text}]
    assert len(sent) == len(prompt) + int(adds_reminder)
    if adds_reminder:
        assert "JSON" in sent[0]["content"]
        assert sent[1:] == prompt
    else:
        assert sent == prompt


def test_json_retry_when_reasoning_returns_null_content(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    client = _FakeClient([
        {"choices": [{"finish_reason": "length", "message": {"content": None}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": '{"correct": true}'}}]},
    ])
    _FakeCore.client = client

    result = Remote("JUDGE", _FakeCore, tmp_path / "api").json(
        [{"role": "user", "content": "judge"}], "case")

    assert result == {"correct": True}
    assert client.requests[0][1]["max_tokens"] == 4096
    assert client.requests[1][1]["max_tokens"] == 8192


def test_json_retry_expands_max_completion_tokens_without_adding_max_tokens(
    tmp_path, monkeypatch
):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv(
        "JUDGE_EXTRA_BODY",
        json.dumps({"max_tokens": None, "max_completion_tokens": 2048}),
    )
    client = _FakeClient([
        {"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": '{"faithful": true}'}}]},
    ])
    _FakeCore.client = client

    result = Remote("JUDGE", _FakeCore, tmp_path / "api").json(
        [{"role": "user", "content": "judge"}], "case")

    assert result == {"faithful": True}
    assert "max_tokens" not in client.requests[0][1]
    assert client.requests[0][1]["max_completion_tokens"] == 2048
    assert "max_tokens" not in client.requests[1][1]
    assert client.requests[1][1]["max_completion_tokens"] == 4096
