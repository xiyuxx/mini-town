"""听到的不能等于知道：没有支撑的话，进听者的是"听说"。"""

import asyncio

from backend.town.engine import SimulationEngine


def run(coro):
    return asyncio.run(coro)


async def make_engine(tmp_path):
    engine = SimulationEngine()
    engine.embedding_provider.fallback = True
    engine.llm.fallback = False          # 下面是打桩，不会真的出网
    db_path = str(tmp_path / "town.db")
    engine.memory.db_path = db_path
    engine.relationship_store.db_path = db_path
    engine.trace.db_path = db_path
    await engine.init()
    return engine


def test_a_claim_the_speaker_cannot_support_is_only_heard(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        speaker, listener = (next(a for a in engine.agents if a.id == i) for i in ("mei", "wang"))
        for agent in (speaker, listener):
            agent.state.current_location = "cafe"
            agent.state.x, agent.state.y = 8, 6
            agent.state.status = "IDLE"
        engine.llm.fallback = True           # 开局用回退台词，避免多一次调用

        async def fake_turn(system, user, tools, registry, **kwargs):
            return {"content": "外头下着小雨呢", "action": "continue", "emoji": "🌧",
                    "mental_update": {}, "referenced_fact_ids": []}

        engine.llm.fallback = False
        engine.llm.function_call_loop = fake_turn

        async def unsupported(claim, facts):
            return {"supported": False, "rewrite": "我不太确定，外头好像下着小雨"}

        engine.llm.validate_claim_support = unsupported
        sim_time = engine.get_sim_time_str()
        await engine.dialogue.start_dialogue([speaker, listener], "早啊", "cafe",
                                             engine.trace, sim_time)
        await engine.dialogue.advance_all(engine.trace, sim_time)

        heard = [
            (agent.id, belief)
            for agent in (speaker, listener)
            for belief in agent.mental_state.beliefs
            if belief.status == "reported"
        ]
        assert heard, "听到一句没有支撑的话，必须留下'听说'的记录"
        assert "小雨" in heard[0][1].proposition
        assert len(heard) == 1, "one turn means one person heard it"
        assert not any(
            belief.status == "observed"
            for agent in (speaker, listener) for belief in agent.mental_state.beliefs
        )

    run(scenario())
