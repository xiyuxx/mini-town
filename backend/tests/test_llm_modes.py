import asyncio
from types import SimpleNamespace

from backend.town.llm import LLMProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def post(self, url, headers, json):
        self.requests.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.responses.pop(0))


class FakeRegistry:
    async def execute(self, _tool_call):
        return SimpleNamespace(content="工具已完成")

    def consume_submitted_action(self):
        return None


def run(coro):
    return asyncio.run(coro)


def make_provider(client):
    provider = LLMProvider()
    provider.fallback = False
    provider._client = client
    return provider


def test_chat_mode_disables_thinking_and_keeps_temperature():
    async def scenario():
        client = FakeClient([{"choices": [{"message": {"content": "ok"}}]}])
        provider = make_provider(client)
        await provider._chat_raw(
            [{"role": "user", "content": "hello"}],
            temperature=0.4,
            mode="chat",
            task="dialogue",
        )
        body = client.requests[0]["json"]
        assert body["thinking"] == {"type": "disabled"}
        assert body["temperature"] == 0.4
        assert "reasoning_effort" not in body
        assert provider.recent_request_metadata(1)[0]["task"] == "dialogue"
        assert provider.recent_request_metadata(1)[0]["mode"] == "chat"

    run(scenario())


def test_thinking_mode_enables_reasoning_and_omits_temperature():
    async def scenario():
        client = FakeClient([{"choices": [{"message": {"content": "[]"}}]}])
        provider = make_provider(client)
        await provider._chat_raw(
            [{"role": "user", "content": "reflect"}],
            temperature=0.1,
            mode="thinking",
            reasoning_effort="max",
            task="reflection",
        )
        body = client.requests[0]["json"]
        assert body["thinking"] == {"type": "enabled"}
        assert body["reasoning_effort"] == "max"
        assert "temperature" not in body
        metadata = provider.recent_request_metadata(1)[0]
        assert metadata["task"] == "reflection"
        assert metadata["reasoningEffort"] == "max"

    run(scenario())


def test_function_call_loop_preserves_reasoning_content_after_tool_call():
    async def scenario():
        first_message = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "先查询工具结果再回答",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }],
        }
        client = FakeClient([
            {"choices": [{"message": first_message}]},
            {"choices": [{"message": {"content": '{"ok": true}'}}]},
        ])
        provider = make_provider(client)
        result = await provider.function_call_loop(
            "system", "user", [{"type": "function"}], FakeRegistry(),
            max_calls=2, mode="chat", task="dialogue_turn",
        )
        assert result == {"ok": True}
        second_messages = client.requests[1]["json"]["messages"]
        assistant_messages = [
            message for message in second_messages if message.get("role") == "assistant"
        ]
        assert assistant_messages[0]["reasoning_content"] == "先查询工具结果再回答"
        assert client.requests[0]["json"]["thinking"] == {"type": "disabled"}

    run(scenario())


def test_invalid_llm_mode_is_rejected_before_request():
    async def scenario():
        client = FakeClient([])
        provider = make_provider(client)
        try:
            await provider._chat_raw(
                [{"role": "user", "content": "hello"}], mode="unknown"
            )
        except ValueError as exc:
            assert "unsupported LLM mode" in str(exc)
        else:
            raise AssertionError("invalid mode should fail")
        assert client.requests == []

    run(scenario())
