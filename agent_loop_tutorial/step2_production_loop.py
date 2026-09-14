"""
Step 2: 更接近生产级的 agent loop

在 Step 1 基础上加的东西，对应 nanobot / 大多数成熟 agent 框架的通用做法：

  1. Tool 抽象成类（带 name/description/parameters/execute），而不是裸函数
     -> 方便生成 function-calling schema 给模型，也方便做参数校验
  2. ToolRegistry 管理工具的注册/查找/执行，执行失败不抛异常，返回错误字符串
     -> 错误要能"喂回模型"让它看到、能重试，而不是让整个 loop 崩掉
  3. Hook 机制：在关键节点（迭代开始、工具执行前后、结束）留出扩展点
     -> 日志、限流、埋点、审批都通过 hook 加，不用改核心 loop
  4. 显式的停止原因（stop_reason）：completed / max_iterations / error
     -> 上层（比如 API 返回、UI 展示）需要知道 agent 是怎么结束的
  5. LLM 调用失败的重试与兜底
  6. 达到 max_iterations 时，做一次"不带工具"的收尾请求，让模型总结现状
     而不是直接甩出一句死板提示
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ===========================================================================
# 1. 消息/响应的数据结构（和 Step 1 相同，抽出来方便复用）
# ===========================================================================

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "error" | "length"

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ===========================================================================
# 2. Tool 抽象：这是和 Step 1 最大的区别
# ===========================================================================

class Tool(ABC):
    """所有工具的基类。子类只需要实现 name/description/parameters/execute。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema，描述这个工具接受什么参数。"""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """执行工具逻辑，返回字符串结果。抛异常会被 ToolRegistry 捕获转成错误文本。"""
        ...

    def to_schema(self) -> dict[str, Any]:
        """转成 OpenAI function-calling 格式的 schema，直接传给模型。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class GetWeatherTool(Tool):
    name = "get_weather"
    description = "查询指定城市的当前天气"
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名"}},
        "required": ["city"],
    }

    def execute(self, city: str) -> str:
        return f"{city}今天晴，26℃"


class DivideTool(Tool):
    """故意留一个会报错的工具，用来演示错误处理路径。"""

    name = "divide"
    description = "两个数相除"
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }

    def execute(self, a: float, b: float) -> str:
        return str(a / b)  # b=0 时会抛 ZeroDivisionError，用来演示错误处理


class ToolRegistry:
    """管理工具的注册、查找、schema 导出、安全执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_definitions(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """执行一个工具调用。返回 (结果文本, 是否出错)。

        关键点：任何异常都在这里被捕获，转成一段模型能读懂的错误文本，
        绝不让异常向上冒泡打断整个 agent loop。工具调用失败是"正常事件"，
        不是"系统故障"——应该让模型看到错误、决定要不要换个方式重试。
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: tool '{name}' not found. Available: {list(self._tools)}", True
        try:
            return tool.execute(**arguments), False
        except Exception as exc:
            return f"Error executing {name}: {type(exc).__name__}: {exc}", True


# ===========================================================================
# 3. Hook：在循环的关键节点留扩展点，核心 loop 代码不需要为了加日志/埋点而改
# ===========================================================================

class AgentHook:
    """默认什么都不做；子类按需覆写想关心的钩子。"""

    def on_iteration_start(self, iteration: int, messages: list[dict[str, Any]]) -> None:
        pass

    def on_llm_response(self, response: LLMResponse) -> None:
        pass

    def on_before_tool(self, tool_call: ToolCall) -> None:
        pass

    def on_after_tool(self, tool_call: ToolCall, result: str, is_error: bool) -> None:
        pass

    def on_finish(self, stop_reason: str, final_content: str | None) -> None:
        pass


class LoggingHook(AgentHook):
    """一个具体的 hook 实现：只是把关键事件打出来。"""

    def on_iteration_start(self, iteration: int, messages: list[dict[str, Any]]) -> None:
        print(f"[iter {iteration}] messages so far: {len(messages)}")

    def on_before_tool(self, tool_call: ToolCall) -> None:
        print(f"  -> calling tool: {tool_call.name}({tool_call.arguments})")

    def on_after_tool(self, tool_call: ToolCall, result: str, is_error: bool) -> None:
        tag = "ERROR" if is_error else "ok"
        print(f"  <- [{tag}] {tool_call.name}: {result[:80]}")

    def on_finish(self, stop_reason: str, final_content: str | None) -> None:
        print(f"[finish] stop_reason={stop_reason}")


# ===========================================================================
# 4. LLM 调用封装：真实项目里这里是 openai/anthropic 的 SDK 调用 + 重试逻辑
# ===========================================================================

class LLMCallError(Exception):
    pass


def call_llm_with_retry(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_retries: int = 2,
) -> LLMResponse:
    """真实实现：调用 provider SDK，网络错误/限流错误做指数退避重试。

    这里用 fake_llm 模拟，同时演示"重试"骨架长什么样。
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _fake_llm(messages, tools)
        except LLMCallError as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(0)  # 真实场景这里是 time.sleep(backoff)
                continue
    # 重试全部耗尽，返回一个 finish_reason="error" 的响应，而不是抛异常
    return LLMResponse(content=f"LLM调用失败: {last_exc}", finish_reason="error")


def _fake_llm(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
    """模拟 LLM：根据最后一条 user 消息决定要不要调用工具。"""
    last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
    has_tool_result = any(m["role"] == "tool" for m in messages)
    text = (last_user or {}).get("content", "")

    if "天气" in text and not has_tool_result:
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "杭州"})],
            finish_reason="tool_calls",
        )
    if "除" in text and not has_tool_result:
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="call_2", name="divide", arguments={"a": 10, "b": 0})],
            finish_reason="tool_calls",
        )
    if has_tool_result:
        last_tool = next(m for m in reversed(messages) if m["role"] == "tool")
        if "Error" in last_tool["content"]:
            return LLMResponse(content=f"抱歉，操作失败了：{last_tool['content']}")
        return LLMResponse(content=f"根据查询结果：{last_tool['content']}")

    return LLMResponse(content="你好，我可以查天气或做除法运算。")


# ===========================================================================
# 5. AgentRunner: 核心循环本身
# ===========================================================================

@dataclass
class AgentRunResult:
    final_content: str | None
    messages: list[dict[str, Any]]
    stop_reason: str  # "completed" | "max_iterations" | "error"


class AgentRunner:
    def __init__(self, tools: ToolRegistry, hook: AgentHook | None = None) -> None:
        self.tools = tools
        self.hook = hook or AgentHook()

    def run(self, messages: list[dict[str, Any]], max_iterations: int = 8) -> AgentRunResult:
        for iteration in range(max_iterations):
            self.hook.on_iteration_start(iteration, messages)

            response = call_llm_with_retry(messages, self.tools.get_definitions())
            self.hook.on_llm_response(response)

            # --- LLM 本身调用失败：结束循环，把错误信息当最终答案返回 ---
            if response.finish_reason == "error":
                self.hook.on_finish("error", response.content)
                messages.append({"role": "assistant", "content": response.content})
                return AgentRunResult(response.content, messages, "error")

            # --- 没有工具调用：这是最终答案 ---
            if not response.has_tool_calls:
                messages.append({"role": "assistant", "content": response.content})
                self.hook.on_finish("completed", response.content)
                return AgentRunResult(response.content, messages, "completed")

            # --- 有工具调用：先记录 assistant 消息(带 tool_calls) ---
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

            # --- 逐个执行工具，结果作为 role="tool" 消息接回历史 ---
            for tc in response.tool_calls:
                self.hook.on_before_tool(tc)
                result, is_error = self.tools.execute(tc.name, tc.arguments)
                self.hook.on_after_tool(tc, result, is_error)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result,
                })
            # 不返回，继续下一轮，把工具结果喂给模型

        # --- 到这里说明达到了 max_iterations，做一次不带工具的收尾请求 ---
        final = self._finalize_after_max_iterations(messages)
        self.hook.on_finish("max_iterations", final)
        return AgentRunResult(final, messages, "max_iterations")

    def _finalize_after_max_iterations(self, messages: list[dict[str, Any]]) -> str:
        """达到最大轮数时，不要直接甩一句"超时了"，而是让模型基于现有信息总结一下。

        做法：追加一条系统提示，明确告诉模型"预算用完了，不要再调用工具，
        基于目前已知信息给出最好的回答"，再调一次不带 tools 参数的请求。
        """
        wrap_up_messages = list(messages) + [{
            "role": "user",
            "content": (
                "你已经用完了可用的操作次数。不要再尝试调用任何工具，"
                "请基于目前已经获得的信息，直接给出你能给出的最佳回答。"
            ),
        }]
        response = call_llm_with_retry(wrap_up_messages, tools=[])  # 不传 tools，模型不会再调用工具
        return response.content or "抱歉，未能在限定步骤内完成任务。"


# ===========================================================================
# 6. 组装 + 演示
# ===========================================================================

def build_agent() -> AgentRunner:
    registry = ToolRegistry()
    registry.register(GetWeatherTool())
    registry.register(DivideTool())
    return AgentRunner(registry, hook=LoggingHook())


def run_agent(user_input: str) -> str:
    agent = build_agent()
    messages = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": user_input},
    ]
    result = agent.run(messages)
    return result.final_content or ""


if __name__ == "__main__":
    print("=== 场景1: 普通问答 ===")
    print(run_agent("你好"))

    print("\n=== 场景2: 需要调用工具 ===")
    print(run_agent("今天天气怎么样"))

    print("\n=== 场景3: 工具执行报错，模型看到错误并回应 ===")
    print(run_agent("帮我算一下除法"))
