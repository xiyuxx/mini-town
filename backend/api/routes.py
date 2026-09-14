"""FastAPI routes: REST + WebSocket — Phase 1."""

import asyncio
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

router = APIRouter()
_engine = None


def set_engine(engine):
    global _engine
    _engine = engine


def _sim_time_str() -> str:
    return f"第{_engine.day}天 {_engine.hour:02d}:{_engine.minute:02d}"


@router.get("/api/state")
async def get_state():
    return _engine.get_state()


@router.get("/api/agents")
async def get_agents():
    return [a.state.to_dict() for a in _engine.agents]


@router.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = next((a for a in _engine.agents if a.id == agent_id), None)
    if not agent:
        raise HTTPException(status_code=404, detail="agent not found")
    memories = await _engine.memory.get_recent(agent_id, hours=24, limit=50)
    relationships = await _engine.relationship_store.get_all_for(agent_id)
    return {
        "agent": agent.state.to_dict(),
        "mentalState": agent.mental_state.context_dict(_engine.get_sim_timestamp()),
        "memories": [
            {"id": m.id, "time": m.time, "location": m.location,
             "content": m.content, "importance": m.importance, "type": m.type,
             "tier": m.tier, "emotion": m.emotion, "eventType": m.event_type,
             "unresolved": m.unresolved, "futureIntention": m.future_intention,
             "formationScore": m.formation_score, "sourceIds": m.source_ids}
            for m in memories
        ],
        "relationships": [r.to_dict() for r in relationships],
    }


@router.get("/api/agents/{agent_id}/memories")
async def get_agent_memories(agent_id: str):
    if not any(a.id == agent_id for a in _engine.agents):
        raise HTTPException(status_code=404, detail="agent not found")
    memories = await _engine.memory.get_all_for_agent(agent_id, limit=200)
    return [
        {"id": m.id, "time": m.time, "location": m.location,
         "content": m.content, "importance": m.importance, "type": m.type,
         "tier": m.tier, "emotion": m.emotion, "eventType": m.event_type,
         "unresolved": m.unresolved, "futureIntention": m.future_intention,
         "formationScore": m.formation_score, "sourceIds": m.source_ids}
        for m in memories
    ]


@router.get("/api/agents/{agent_id}/relationships")
async def get_agent_relationships(agent_id: str):
    if not any(a.id == agent_id for a in _engine.agents):
        raise HTTPException(status_code=404, detail="agent not found")
    relationships = await _engine.relationship_store.get_all_for(agent_id)
    return [r.to_dict() for r in relationships]


@router.get("/api/locations")
async def get_locations():
    from backend.town.world import LOCATIONS
    return [
        {"id": loc.id, "name": loc.name, "type": loc.type,
         "x": loc.x, "y": loc.y, "width": loc.width, "height": loc.height,
         "color": loc.color, "emoji": loc.emoji}
        for loc in LOCATIONS
    ]


def _decorate_dialogue(dialogue: dict) -> dict:
    names = {agent.id: agent.name for agent in _engine.agents}
    result = dict(dialogue)
    result["participantNames"] = [
        names.get(agent_id, agent_id) for agent_id in dialogue.get("participants", [])
    ]
    return result


@router.get("/api/dialogues")
async def get_dialogues(limit: int = 30):
    dialogues = await _engine.dialogue_store.list_recent(limit=max(1, min(limit, 100)))
    return [_decorate_dialogue(dialogue) for dialogue in dialogues]


@router.get("/api/dialogues/{dialogue_id}")
async def get_dialogue(dialogue_id: str):
    dialogue = await _engine.dialogue_store.get(dialogue_id)
    if not dialogue:
        raise HTTPException(status_code=404, detail="dialogue not found")
    return _decorate_dialogue(dialogue)


@router.get("/api/agents/{agent_id}/dialogues")
async def get_agent_dialogues(agent_id: str, limit: int = 30):
    if not any(agent.id == agent_id for agent in _engine.agents):
        raise HTTPException(status_code=404, detail="agent not found")
    dialogues = await _engine.dialogue_store.list_recent(
        limit=max(1, min(limit, 100)), agent_id=agent_id,
    )
    return [_decorate_dialogue(dialogue) for dialogue in dialogues]


@router.get("/api/context-metrics")
async def get_context_metrics(limit: int = 50):
    return _engine.context.recent_metrics(limit)


@router.get("/api/interactions")
async def get_interactions(limit: int = 50):
    interactions = await _engine.interaction_store.list_recent(
        limit=max(1, min(limit, 200))
    )
    return [item.to_dict() for item in interactions]


@router.get("/api/interactions/{interaction_id}")
async def get_interaction(interaction_id: str):
    interaction = await _engine.interaction_store.get(interaction_id)
    if not interaction:
        raise HTTPException(status_code=404, detail="interaction not found")
    return interaction.to_dict()


@router.get("/api/traces")
async def get_traces():
    traces = await _engine.trace.get_recent(200)
    return traces


@router.post("/api/simulation/start")
async def start_simulation():
    if not _engine.running:
        await _engine.start()
    return {"ok": True}


@router.post("/api/simulation/pause")
async def pause_simulation():
    _engine.pause()
    return {"ok": True}


@router.post("/api/simulation/reset")
async def reset_simulation():
    await _engine.reset()
    return {"ok": True}


@router.post("/api/simulation/speed")
async def set_speed(data: dict):
    speed = data.get("speed", 1)
    _engine.set_speed(speed)
    return {"ok": True, "speed": _engine.speed}


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    queue = await _engine.subscribe()
    try:
        while True:
            state = await queue.get()
            await ws.send_json(state)
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _engine.unsubscribe(queue)
