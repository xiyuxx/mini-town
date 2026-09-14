"""
Step 1: 最小可用的 agent loop

核心骨架只有三件事:
  1. messages: 一份不断增长的对话历史(list[dict])
  2. 调 LLM，拿到 response
  3. 判断 response 里有没有"工具调用"；
     - 有 -> 执行工具，把结果塞回 messages，继续循环
     - 没有 -> 这是最终答案，跳出循环

这一步先不管错误处理、最大轮数、并发——先把"循环长什么样"看清楚。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 1. 消息格式：沿用 OpenAI 的 chat message 结构，这是目前最通用的事实标准
#    role: "system" | "user" | "assistant" | "tool"
#    assistant 消息里可能带 tool_calls 字段
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------------------
# 2. 一个"假的" LLM：真实项目里这里是 openai.chat.completions.create(...)
#    这里用一个简单规则模拟："问天气就调 get_weather 工具，否则直接回答"
# ---------------------------------------------------------------------------

def fake_llm_call(messages: list[dict[str, Any]]) -> LLMResponse:
    last_user_msg = next(
        (m for m in reversed(messages) if m["role"] == "user"), None
    )
    has_tool_result = any(m["role"] == "tool" for m in messages)

    if last_user_msg and "天气" in last_user_msg["content"] and not has_tool_result:
        # 模型决定调用工具，而不是直接回答
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"city": "杭州"})
            ],
        )

    if has_tool_result:
        # 工具结果已经在历史里了，模型给出最终答案
        tool_msg = next(m for m in messages if m["role"] == "tool")
        return LLMResponse(content=f"根据查询结果：{tool_msg['content']}")

    return LLMResponse(content="你好，我能帮你查天气，试着问我「今天天气怎么样」")


# ---------------------------------------------------------------------------
# 3. 工具实现：一个普通函数，接收参数、返回字符串结果
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    return f"{city}今天晴，26℃"


TOOLS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
}


# ---------------------------------------------------------------------------
# 4. 核心 loop
# ---------------------------------------------------------------------------

def run_agent(user_input: str, max_iterations: int = 5) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": user_input},
    ]

    for _ in range(max_iterations):
        response = fake_llm_call(messages)

        if not response.has_tool_calls:
            # 没有工具调用 -> 这就是最终回答，退出循环
            return response.content or ""

        # 有工具调用：先把 assistant 这条消息（包含 tool_calls）存进历史
        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ],
        })

        # 逐个执行工具调用，把结果作为 role="tool" 的消息接回去
        for tc in response.tool_calls:
            fn = TOOLS.get(tc.name)
            result = fn(**tc.arguments) if fn else f"Error: unknown tool {tc.name}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": result,
            })

        # 不 return，继续下一轮循环，把工具结果喂给模型

    return "达到最大轮数，仍未得到最终答案。"


if __name__ == "__main__":
    print(run_agent("你好"))
    print(run_agent("今天天气怎么样"))
