"""Full tool registry factory for agent decision-making.

Each agent gets its own bound ToolRegistry with handlers that
close over agent_id and engine_ref so tools return agent‑scoped data.
"""

from typing import TYPE_CHECKING

from .tools import ToolRegistry, ToolCall
from .world import LOCATION_MAP, location_name, agents_at_location
from .memory import memories_to_text, auto_importance
from ..config import config

if TYPE_CHECKING:
    from .engine import SimulationEngine


def make_registry_for(agent_id: str, engine_ref: "SimulationEngine") -> ToolRegistry:
    """Build a ToolRegistry with agent‑scoped async handlers."""
    registry = ToolRegistry()

    # ── 1. get_weather ────────────────────────────────────────
    registry.register(
        name="get_weather",
        description="查询当前世界的天气状态与结构化出行成本",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_make_get_weather(engine_ref),
    )

    # ── 2. get_nearby_agents ──────────────────────────────────
    registry.register(
        name="get_nearby_agents",
        description="查询当前地点有哪些其他角色",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_make_get_nearby_agents(agent_id, engine_ref),
    )

    # ── 3. get_schedule ───────────────────────────────────────
    registry.register(
        name="get_schedule",
        description="查询自己当前的日程安排",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_make_get_schedule(agent_id, engine_ref),
    )

    # ── 4. get_location_info ──────────────────────────────────
    registry.register(
        name="get_location_info",
        description="查询某个地点的详细信息（名称、类型、描述）",
        parameters={
            "type": "object",
            "properties": {
                "location_id": {"type": "string", "description": "WorldPack中声明的地点ID"}
            },
            "required": ["location_id"],
        },
        handler=_make_get_location_info(),
    )

    # ── 5. get_relationship ───────────────────────────────────
    registry.register(
        name="get_relationship",
        description="查询自己和另一个角色的关系（熟悉度、好感度、重要事件等）",
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "目标角色的名字"}
            },
            "required": ["target_name"],
        },
        handler=_make_get_relationship(agent_id, engine_ref),
    )

    # ── 6. list_known_agents ──────────────────────────────────
    registry.register(
        name="list_known_agents",
        description="列出当前世界中已知角色的基本信息",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_make_list_known_agents(agent_id, engine_ref),
    )

    # ── 7. recall_memories ────────────────────────────────────
    registry.register(
        name="recall_memories",
        description="回忆近期记忆，可以按关键词过滤",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "可选，按关键词筛选记忆"}
            },
            "required": [],
        },
        handler=_make_recall_memories(agent_id, engine_ref),
    )

    # ── 8. remember ───────────────────────────────────────────
    registry.register(
        name="remember",
        description="记录一条新的记忆（在决策过程中想到的重要信息）",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要记录的内容"},
                "importance": {"type": "integer", "description": "重要性 1-10，默认5"}
            },
            "required": ["content"],
        },
        handler=_make_remember(agent_id, engine_ref),
    )

    # ── 9. do_activity ─────────────────────────────────────────
    registry.register(
        name="do_activity",
        description="开始一项会消耗时间的活动（如观察、等待、吃饭、工作、读书、睡觉）。任何外显行为都应该调用这个工具；时间到了系统会自动通知你完成。",
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "活动描述，如'吃早餐'、'在咖啡馆工作'、'午休'"
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "持续时间（分钟）。短动作至少5分钟；普通活动15-60分钟。"
                },
                "location_id": {
                    "type": "string",
                    "description": "活动地点ID（可选）。如果指定，系统会检查你是否在那里"
                },
                "mental_update": {"type": "object", "description": "本次选择对应的持续心智更新", "additionalProperties": True},
            },
            "required": ["description"],
        },
        handler=_make_do_activity(agent_id, engine_ref, registry),
    )

    # ── 10. move_to ────────────────────────────────────────────
    registry.register(
        name="move_to",
        description="移动到一个地点。这是你实际移动的工具——调用它来走到目的地。到达后再用 do_activity 开始要做的事。",
        parameters={
            "type": "object",
            "properties": {
                "location_id": {
                    "type": "string",
                    "description": "WorldPack中声明的目标地点ID"
                },
                "reason": {"type": "string", "description": "此时前往该地点的具体原因"},
                "mental_update": {"type": "object", "description": "本次选择对应的持续心智更新", "additionalProperties": True},
            },
            "required": ["location_id"],
        },
        handler=_make_move_to(agent_id, engine_ref, registry),
    )

    # ── 10. decide_action ──────────────────────────────────────
    registry.register(
        name="decide_action",
        description="提交非活动类最终决策，主要用于 talk 或 reflect。普通行为请使用 do_activity；如果仍提交 act，系统会把它转换成耗时活动。",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["act", "talk", "reflect"],
                    "description": "行动类型：talk=和人说话，reflect=静思，act=兼容旧输出但会被系统转成耗时活动"
                },
                "location": {
                    "type": "string",
                    "description": "当action为move时必填，目标地点ID"
                },
                "content": {
                    "type": "string",
                    "description": "行动描述（1-2句话，中文）"
                },
                "emoji": {
                    "type": "string",
                    "description": "相关emoji表情"
                },
                "target_name": {
                    "type": "string",
                    "description": "当action为talk时必填，对话对象的名字"
                },
                "mental_update": {"type": "object", "description": "本次选择对应的持续心智更新", "additionalProperties": True},
            },
            "required": ["action", "content", "emoji"],
        },
        handler=_make_decide_action(agent_id, engine_ref, registry),
    )

    # ── 10. dialogue_reply ────────────────────────────────────
    registry.register(
        name="dialogue_reply",
        description="在对话中提交回复。调用此工具来发送你说的话。",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "你说的话（1-3句，中文，口语化）"
                },
                "action": {
                    "type": "string",
                    "enum": ["continue", "end"],
                    "description": "continue=继续对话, end=自然结束对话"
                },
                "emoji": {
                    "type": "string",
                    "description": "相关emoji"
                },
            },
            "required": ["content", "action", "emoji"],
        },
        handler=_make_dialogue_reply(),
    )

    return registry


# ── Handler factories ─────────────────────────────────────────

def _make_get_weather(engine_ref: "SimulationEngine"):
    async def handler() -> str:
        weather = getattr(engine_ref, "weather", {})
        town_agent = getattr(engine_ref, "town_agent", None)
        if town_agent:
            ws = town_agent.get_state()
            w = ws.get("weather", {})
            cond = w.get("condition", weather)
            temp = w.get("temperature", "适中")
            wind = w.get("wind", "无风")
            desc = w.get("description", "")
            return f"天气: {cond}，温度: {temp}，风力: {wind}。{desc}"
        return f"天气: {weather}"
    return handler


def _make_get_nearby_agents(agent_id: str, engine_ref: "SimulationEngine"):
    async def handler() -> str:
        agent = _find_agent(agent_id, engine_ref)
        if not agent:
            return "无法获取状态"
        nearby = agents_at_location(engine_ref.agents, agent.state.current_location)
        others = [a for a in nearby if a.id != agent_id]
        if not others:
            return "附近没有其他人。"
        lines = [f"- {a.name}（{a.state.occupation}），{a.state.mood}，正在{a.state.current_action}" for a in others]
        return "附近的人：\n" + "\n".join(lines)
    return handler


def _make_get_schedule(agent_id: str, engine_ref: "SimulationEngine"):
    async def handler() -> str:
        agent = _find_agent(agent_id, engine_ref)
        if not agent:
            return "无法获取日程"
        hour = engine_ref.hour
        minute = engine_ref.minute
        return agent.get_schedule_for_time(hour, minute)
    return handler


def _make_get_location_info():
    async def handler(location_id: str) -> str:
        loc = LOCATION_MAP.get(location_id)
        if not loc:
            return f"未知地点: {location_id}"
        return f"{loc.name}（{loc.id}）：类型={loc.type}，{loc.emoji}"
    return handler


def _make_get_relationship(agent_id: str, engine_ref: "SimulationEngine"):
    async def handler(target_name: str) -> str:
        target = _find_agent_by_name(target_name, engine_ref)
        if not target:
            return f"没有找到叫{target_name}的人"
        rs = getattr(engine_ref, "relationship_store", None)
        if not rs:
            return "人际关系系统不可用"
        rel = await rs.get(agent_id, target.id)
        return rs.format_for_llm(rel, target.name)
    return handler


def _make_list_known_agents(agent_id: str, engine_ref: "SimulationEngine"):
    async def handler() -> str:
        agents = engine_ref.agents
        lines = []
        for a in agents:
            loc = location_name(a.state.current_location)
            lines.append(f"- {a.name}（{a.state.occupation}，{a.state.age}岁），在{loc}")
        return "已知角色：\n" + "\n".join(lines)
    return handler


def _make_recall_memories(agent_id: str, engine_ref: "SimulationEngine"):
    async def handler(keyword: str = "") -> str:
        agent = _find_agent(agent_id, engine_ref)
        if not agent:
            return "无法获取记忆"
        memories = await engine_ref.memory.retrieve(
            agent_id, agent.state.current_location,
            nearby_agents=[],
            current_sim_timestamp=engine_ref.get_sim_timestamp(),
        )
        if not memories:
            return "没有相关记忆。"
        text = memories_to_text(memories)
        if keyword:
            # Simple keyword filter
            lines = text.split("\n")
            filtered = [l for l in lines if keyword in l]
            if not filtered:
                return f"没有找到与'{keyword}'相关的记忆。"
            return "相关记忆：\n" + "\n".join(filtered)
        return "近期记忆：\n" + text
    return handler


def _make_remember(agent_id: str, engine_ref: "SimulationEngine"):
    async def handler(content: str, importance: int = 5) -> str:
        agent = _find_agent(agent_id, engine_ref)
        loc = agent.state.current_location if agent else "unknown"
        imp = max(1, min(10, importance))
        await engine_ref.memory.add(agent_id, loc, content, "observation", imp, sim_time=engine_ref.get_sim_time_str())
        return f"已记住: {content}"
    return handler


def _make_decide_action(agent_id: str, engine_ref: "SimulationEngine", registry: ToolRegistry):
    """Handler returns a serialised action dict as a string.

    The LLM function_call_loop will return the tool result content, which
    we parse back in decide_action().
    """
    async def handler(action: str, content: str, emoji: str,
                      location: str = "", target_name: str = "",
                      mental_update: dict | None = None) -> str:
        import json

        # Resolve target_name to target_id
        target_id = ""
        if action == "talk" and target_name:
            target = _find_agent_by_name(target_name, engine_ref)
            if target:
                target_id = target.id

        result = {
            "action": action,
            "content": content,
            "emoji": emoji,
            "mental_update": mental_update or {},
        }
        if action == "move" and location:
            result["location"] = location
        if action == "talk" and target_id:
            result["target"] = target_id

        registry.submit_action(result)
        return json.dumps(result, ensure_ascii=False)
    return handler


def _make_dialogue_reply():
    async def handler(content: str, action: str, emoji: str) -> str:
        import json
        return json.dumps({
            "content": content,
            "action": action,
            "emoji": emoji,
        }, ensure_ascii=False)
    return handler




def _make_do_activity(agent_id: str, engine_ref: "SimulationEngine", registry: ToolRegistry):
    from .world import location_name, LOCATION_MAP
    async def handler(description: str, duration_minutes: int | None = None,
                      location_id: str = "", mental_update: dict | None = None) -> str:
        agent = _find_agent(agent_id, engine_ref)
        if not agent:
            return "无法获取你的状态"
        if location_id and location_id in LOCATION_MAP and location_id != agent.state.current_location:
            return f"你现在在{location_name(agent.state.current_location)}，不在{location_name(location_id)}。请先调用 move_to 走到那里，再开始活动。"
        if agent.state.status == "MOVING":
            return "你正在移动中，到达后再开始活动。"
        registry.submit_action({
            "action": "act",
            "content": description,
            "location": agent.state.current_location,
            "duration_minutes": duration_minutes,
            "source": "llm",
            "mental_update": mental_update or {},
        })
        return f"已提交活动：{description}。系统将在本 tick 统一开始该活动。"
    return handler
def _make_move_to(agent_id: str, engine_ref: "SimulationEngine", registry: ToolRegistry):
    from .world import location_center, location_name, LOCATION_MAP, can_enter, bfs_path

    async def handler(location_id: str, reason: str = "",
                      mental_update: dict | None = None) -> str:
        agent = _find_agent(agent_id, engine_ref)
        if not agent:
            return "无法获取你的状态"

        if agent.state.status == "MOVING":
            return "你正在移动中，请等待到达目的地后再操作。"

        loc = LOCATION_MAP.get(location_id)
        if not loc:
            return f"找不到地点 {location_id}，可用地点：{', '.join(LOCATION_MAP.keys())}"

        if agent.state.current_location == location_id:
            return f"你已经在{loc.name}了"

        # Access control
        if not can_enter(agent_id, location_id):
            return f"{loc.name}是私宅，你没有钥匙不能进去。你可以选择在门口等待或者去别的地方。"

        # BFS pathfinding
        cx, cy = location_center(location_id)
        path = bfs_path(
            agent.state.x, agent.state.y, cx, cy,
            agent_id=agent_id,
            target_location_id=location_id,
        )
        if not path:
            return f"无法找到从({agent.state.x},{agent.state.y})到{loc.name}的路径。"

        registry.submit_action({
            "action": "move",
            "location": location_id,
            "content": reason or "自主前往",
            "source": "llm",
            "mental_update": mental_update or {},
        })
        return f"已提交前往{loc.name}的意图，系统将在本 tick 校验并开始移动。"

    return handler


# ── Helpers ───────────────────────────────────────────────────

def _find_agent(agent_id: str, engine_ref: "SimulationEngine"):
    for a in engine_ref.agents:
        if a.id == agent_id:
            return a
    return None


def _find_agent_by_name(name: str, engine_ref: "SimulationEngine"):
    for a in engine_ref.agents:
        if a.name == name:
            return a
    return None
