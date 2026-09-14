"""LLM provider with function-calling support."""

import asyncio
import json
import random
from collections import deque
import httpx
from ..config import config
from .world_pack import DEFAULT_WORLD

FALLBACK_TALK_REPLIES = [str(item) for item in DEFAULT_WORLD.data.get("fallback_dialogue", ["嗯。"])]

class LLMProvider:
    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.base_url = config.DEEPSEEK_BASE_URL.rstrip("/")
        self.model = config.DEEPSEEK_MODEL
        self.fallback = not self.api_key or self.api_key.startswith("sk-your-") or "your-key" in self.api_key
        self._client: httpx.AsyncClient | None = None
        self._request_slots = asyncio.Semaphore(max(1, config.LLM_MAX_CONCURRENCY))
        self._request_history: deque[dict] = deque(maxlen=200)
        if self.fallback:
            import warnings
            warnings.warn("LLM disabled or no API key — using schedule-based fallback mode")

    async def _chat_raw(self, messages: list[dict], tools: list[dict] | None = None,
                         temperature: float = 0.7, max_tokens: int = 1024,
                         response_format: dict | None = None,
                         mode: str | None = None, task: str = "generic",
                         reasoning_effort: str | None = None) -> dict:
        """Raw chat completion with task-level thinking mode selection.

        ``chat`` disables DeepSeek thinking and keeps temperature active.
        ``thinking`` enables reasoning and omits temperature because DeepSeek
        ignores sampling parameters in that mode.
        """
        if self.fallback:
            return {"choices": [{"message": {"content": "{}"}}]}

        selected_mode = (mode or config.LLM_DEFAULT_MODE).lower()
        if selected_mode not in {"chat", "thinking"}:
            raise ValueError(f"unsupported LLM mode: {selected_mode}")
        selected_effort = reasoning_effort or config.LLM_REASONING_EFFORT
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled" if selected_mode == "thinking" else "disabled"},
        }
        if selected_mode == "thinking":
            body["reasoning_effort"] = selected_effort
        else:
            body["temperature"] = temperature
        self._request_history.append({
            "task": task,
            "mode": selected_mode,
            "model": self.model,
            "reasoningEffort": selected_effort if selected_mode == "thinking" else None,
            "messageCount": len(messages),
            "toolCount": len(tools or []),
            "responseFormat": (response_format or {}).get("type") if response_format else None,
            "inputChars": sum(len(str(message.get("content") or "")) for message in messages),
        })
        if response_format:
            body["response_format"] = response_format
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        # Reusing the client preserves TCP/TLS connections between completions.
        # Planning, dialogue, and memory share one bounded request pool.
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(config.LLM_TIMEOUT_SECONDS),
                limits=httpx.Limits(
                    max_connections=config.LLM_MAX_CONCURRENCY * 2,
                    max_keepalive_connections=config.LLM_MAX_CONCURRENCY,
                ),
            )
        async with self._request_slots:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body,
            )
        resp.raise_for_status()
        return resp.json()

    def recent_request_metadata(self, limit: int = 50) -> list[dict]:
        """Return bounded, prompt-free request metadata for diagnostics."""
        return list(self._request_history)[-max(1, min(limit, 200)):]

    async def close(self) -> None:
        """Release the shared HTTP connection pool during application shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def function_call_loop(self, system: str, user: str, tools: list[dict],
                                   tool_registry, max_calls: int = 5,
                                   mode: str = "chat", task: str = "tool_loop") -> dict:
        """Run a function-calling loop: LLM calls tools, we execute, repeat."""
        if self.fallback:
            return {"action": "act", "content": "按日程活动", "emoji": "😊"}

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        call_count = 0

        while call_count < max_calls:
            call_count += 1
            try:
                data = await self._chat_raw(messages, tools, mode=mode, task=task)
            except Exception as e:
                import warnings
                raise RuntimeError(f"LLM API error: {e}")
            choice = data["choices"][0]
            msg = choice["message"]

            # LLM wants to call tools
            if msg.get("tool_calls"):
                messages.append(msg)
                for tc in msg["tool_calls"]:
                    func = tc["function"]
                    result = await tool_registry.execute(
                        type("ToolCall", (), {"id": tc["id"], "name": func["name"],
                                               "arguments": json.loads(func.get("arguments", "{}"))})()
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result.content,
                    })
                    # Action tools submit an intent instead of mutating the
                    # simulation during prompting. Return it for tick-level
                    # validation and event generation.
                    submitted = tool_registry.consume_submitted_action()
                    if submitted:
                        return submitted
                continue

            # LLM returned final content — parse as JSON
            content = msg.get("content", "{}")
            try:
                return json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
            except json.JSONDecodeError:
                return {"action": "act", "content": content[:80], "emoji": "😐"}

        return {"action": "act", "content": "", "emoji": "😶"}

    async def reflect(self, persona: str, memories: str) -> list[str]:
        if self.fallback:
            return await self.reflect_fallback()
        system = (
            "你负责为当前世界中的角色提炼长期认知。只输出JSON数组，包含1-3条简短、具体、"
            "会影响未来行为的第一人称反思。不要复述流水账。"
        )
        data = await self._chat_raw([
            {"role": "system", "content": system},
            {"role": "user", "content": f"角色：\n{persona}\n\n近期经历：\n{memories}"},
        ], temperature=0.5, mode="thinking", task="reflection")
        content = data["choices"][0]["message"].get("content", "[]")
        try:
            parsed = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
            return [str(item)[:300] for item in parsed[:3]] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    async def reason_about_memory(self, persona: str, fact_summary: str,
                                  current_goal: str, emotion: str,
                                  source_fact_ids: list[str]) -> dict:
        """Derive bounded subjective meaning from already-known facts."""
        if self.fallback:
            return {}
        data = await self._chat_raw([
            {"role": "system", "content": (
                "你负责为角色解释一段已经发生的经历。只输出JSON对象："
                "interpretation(string), emotion(string), future_intention(string), "
                "unresolved(boolean), confidence(0到1)。不得添加人物、事件、约定或客观事实；"
                "future_intention只能是角色未来想做的事，不是已发生的承诺。"
            )},
            {"role": "user", "content": json.dumps({
                "persona": persona, "fact_summary": fact_summary,
                "current_goal": current_goal, "current_emotion": emotion,
                "source_fact_ids": source_fact_ids,
            }, ensure_ascii=False)},
        ], mode="thinking", task="memory_reasoning", max_tokens=600)
        try:
            parsed = json.loads(data["choices"][0]["message"].get("content", "{}"))
            return parsed if isinstance(parsed, dict) else {}
        except (KeyError, json.JSONDecodeError):
            return {}

    async def encode_experience_memory(self, persona: str, fact_summary: str,
                                       interpretation_hint: str, emotion: str,
                                       future_intention: str = "",
                                       sim_time: str = "", time_period: str = "") -> str:
        if self.fallback:
            return fact_summary
        system = (
            "把结构化经历编码成一条自然的第一人称记忆。时间、事件事实和已有心理评价都是"
            "不可修改的事实。只使用给定内容，不虚构人物、物品、动作、感官、开关店状态或"
            "共同历史。不得添加未给出的上午、下午、晚上、开店前、关店后等时间关系。"
            "写清发生了什么；仅在已有情绪或理解显著时表达感受，不要为中性经历补写平静、从容或不着急。"
            "不超过90字。"
        )
        data = await self._chat_raw([
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"角色：\n{persona}\n发生时间（不可修改）：{sim_time or '未提供'}\n"
                f"准确时段（不可修改）：{time_period or '未提供'}\n事实：{fact_summary}\n"
                f"已有理解：{interpretation_hint}\n已有情绪：{emotion}\n"
                f"已有后续意图：{future_intention or '无'}"
            )},
        ], temperature=0.45, mode="chat", task="experience_memory")
        content = data["choices"][0]["message"].get("content", "").strip()
        return content[:300] or fact_summary

    async def encode_dialogue_memory(self, persona: str, conversation: str,
                                     other_names: list[str], objective_summary: str) -> str:
        if self.fallback:
            return f"在交谈中遇见了{'、'.join(other_names)}，谈到：{objective_summary}"
        system = (
            "你在为角色形成一条主观记忆。只输出一段第一人称记忆，包含发生的事实、"
            "角色自己的关注或判断；不写分析标签，不超过100字，不要写'我和某某交谈：'。"
        )
        data = await self._chat_raw([
            {"role": "system", "content": system},
            {"role": "user", "content": f"角色：\n{persona}\n\n对话：\n{conversation}\n\n客观摘要：{objective_summary}"},
        ], temperature=0.5, mode="chat", task="dialogue_memory")
        return data["choices"][0]["message"].get("content", objective_summary).strip()[:300]

    async def validate_claim_support(self, claim: str, facts: list[dict]) -> dict:
        if self.fallback:
            return {"supported": bool(facts), "rewrite": claim if facts else f"我不太确定，不过{claim}"}
        system = (
            "判断台词中的确定性事实是否被给定事实支持。只输出JSON："
            "supported(boolean), rewrite(string)。不得引入新事实。若证据不足，rewrite必须"
            "用可能、好像、我不确定等方式保守表达；若台词只是问候、意见或提问，可视为支持。"
        )
        try:
            data = await self._chat_raw([
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"claim": claim, "facts": facts}, ensure_ascii=False)},
            ], temperature=0.1, mode="chat", task="claim_validation")
            content = data["choices"][0]["message"].get("content", "{}")
            result = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
            return {
                "supported": bool(result.get("supported", False)),
                "rewrite": str(result.get("rewrite", ""))[:300],
            }
        except Exception:
            return {"supported": False, "rewrite": f"我不太确定，不过{claim}"[:300]}

    async def evaluate_dialogue(self, conversation: str,
                                participant_names: list[str]) -> dict:
        if self.fallback:
            return {
                "valence": 0.0,
                "affinity_delta": 0.0,
                "trust_delta": 0.0,
                "summary": conversation[-240:],
            }
        system = (
            "评估一场角色之间的对话。只输出JSON对象，字段为valence(-1到1)、"
            "affinity_delta(-1到1)、trust_delta(-0.5到0.5)、summary(具体摘要)。"
        )
        data = await self._chat_raw([
            {"role": "system", "content": system},
            {"role": "user", "content": f"参与者：{', '.join(participant_names)}\n{conversation}"},
        ], temperature=0.2, mode="thinking", task="dialogue_evaluation")
        content = data["choices"][0]["message"].get("content", "{}")
        try:
            result = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            result = {}
        return {
            "valence": max(-1.0, min(1.0, float(result.get("valence", 0.0)))),
            "affinity_delta": max(-1.0, min(1.0, float(result.get("affinity_delta", 0.0)))),
            "trust_delta": max(-0.5, min(0.5, float(result.get("trust_delta", 0.0)))),
            "summary": str(result.get("summary", conversation[-240:]))[:400],
        }

    # ── Fallback methods ──
    async def decide_action_fallback(self, schedule_location: str | None,
                                      current_location: str) -> dict:
        if schedule_location and schedule_location != current_location:
            return {
                "action": "move", "location": schedule_location,
                "content": "执行当前日程中的移动安排", "emoji": "🚶",
                "source": "schedule_fallback",
            }
        return {
            "action": "act", "content": "执行当前日程安排", "emoji": "😊",
            "source": "schedule_fallback",
        }

    async def dialogue_response_fallback(self) -> dict:
        reply = random.choice(FALLBACK_TALK_REPLIES)
        action = random.choice(["continue", "end"])
        return {"content": reply, "action": action, "emoji": "😊"}

    async def reflect_fallback(self) -> list[str]:
        return []
