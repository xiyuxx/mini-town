"""Tool system: base class, registry, and all tool implementations.

Tools are callable by LLM via function-calling. Each tool
returns raw facts, not interpretations — interpretation is the LLM's job.
"""

from dataclasses import dataclass
from typing import Any, Callable, Coroutine


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict  # JSON Schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str


class ToolRegistry:
    def __init__(self):
        self._handlers: dict[str, tuple[ToolDef, Callable]] = {}
        self._submitted_action: dict | None = None

    def register(self, name: str, description: str, parameters: dict, handler: Callable):
        self._handlers[name] = (ToolDef(name, description, parameters), handler)

    def get_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {"name": td.name, "description": td.description, "parameters": td.parameters},
            }
            for td, _ in self._handlers.values()
        ]

    def submit_action(self, action: dict):
        """Store the one world-changing intent produced during this decision."""
        self._submitted_action = action

    def consume_submitted_action(self) -> dict | None:
        action = self._submitted_action
        self._submitted_action = None
        return action

    async def execute(self, call: ToolCall) -> ToolResult:
        if call.name not in self._handlers:
            return ToolResult(call_id=call.id, name=call.name, content=f"Error: unknown tool '{call.name}'")
        _, handler = self._handlers[call.name]
        try:
            result = await handler(**call.arguments)
            return ToolResult(call_id=call.id, name=call.name, content=str(result))
        except Exception as e:
            return ToolResult(call_id=call.id, name=call.name, content=f"Error: {e}")
