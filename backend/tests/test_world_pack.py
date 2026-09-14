import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from backend.town.schedule import compile_routine_blocks
from backend.town.world_pack import DEFAULT_WORLD_PATH, load_world_pack


def _default_data():
    return json.loads(DEFAULT_WORLD_PATH.read_text(encoding="utf-8"))


def test_world_pack_rejects_unknown_resource_location(tmp_path):
    data = _default_data()
    data["resources"][0]["location"] = "missing-place"
    path = tmp_path / "invalid-world.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown location"):
        load_world_pack(path)


def test_world_pack_path_replaces_world_content_across_modules(tmp_path):
    data = copy.deepcopy(_default_data())
    data["id"] = "alternate-world"
    data["name"] = "替代世界"
    data["setting"] = "用于验证声明式世界加载"
    data["agents"][0]["name"] = "测试角色"
    data["locations"][0]["name"] = "测试地点"
    data["resources"][0]["quantity"] = 9
    initial = data["weather"]["initial"]
    data["weather"]["states"][initial]["description"] = "测试天气"
    data["infrastructure"]["services"] = {
        "signal": {"normal": "online", "failed": "offline"}
    }
    path = tmp_path / "alternate-world.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    script = """
import json
from backend.town.engine import SimulationEngine
from backend.town.world import LOCATIONS
engine = SimulationEngine()
first_resource = engine.world_pack.resources[0]["id"]
print(json.dumps({
    "world_id": engine.world_id,
    "world_name": engine.world_name,
    "agent_name": engine.world_pack.agents[0]["name"],
    "location_name": LOCATIONS[0].name,
    "resource_quantity": engine.resources.get(first_resource).quantity,
    "weather_description": engine.weather["description"],
    "infrastructure": engine.infrastructure,
}, ensure_ascii=False))
"""
    env = dict(os.environ, WORLD_PACK_PATH=str(path), PYTHONPATH="/workspace")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd="/workspace", env=env,
        check=True, capture_output=True, text=True,
    )
    loaded = json.loads(result.stdout.strip())
    assert loaded == {
        "world_id": "alternate-world",
        "world_name": "替代世界",
        "agent_name": "测试角色",
        "location_name": "测试地点",
        "resource_quantity": 9.0,
        "weather_description": "测试天气",
        "infrastructure": {"signal": "online"},
    }


def test_world_pack_rejects_anchor_outside_declared_location(tmp_path):
    data = _default_data()
    data["anchors"][0]["x"] = data["map"]["width"]
    path = tmp_path / "invalid-anchor.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must be inside"):
        load_world_pack(path)


def test_world_pack_rejects_cross_location_container_anchor(tmp_path):
    data = _default_data()
    data["containers"][0]["anchor"] = "meal_prep_anchor"
    path = tmp_path / "invalid-container.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="belongs to another location"):
        load_world_pack(path)


def test_default_world_includes_daily_and_public_service_buildings():
    data = _default_data()
    locations = {item["id"]: item for item in data["locations"]}

    assert {"fresh_market", "pharmacy", "community_center", "bus_stop"} <= locations.keys()
    assert locations["fresh_market"]["pass_through"] is False
    assert locations["bus_stop"]["pass_through"] is True
    assert any(item["location"] == "fresh_market" and item["kind"] == "food"
               for item in data["resources"])
    assert any(item["location"] == "pharmacy" and item["kind"] == "medical_supply"
               for item in data["resources"])


def test_routine_blocks_compile_time_windows_and_candidates():
    blocks = compile_routine_blocks([{
        "id": "after_work",
        "window": ["17:00", "20:00"],
        "intent": "下班后的个人时间",
        "candidate_locations": ["bookstore", "park"],
        "candidate_activities": ["逛书店", "散步"],
        "priority": 0.35,
        "flexibility": 0.85,
    }])
    assert blocks[0].contains(17, 0)
    assert blocks[0].contains(19, 55)
    assert not blocks[0].contains(20, 0)
    assert blocks[0].candidate_locations == ("bookstore", "park")
    assert blocks[0].flexibility == 0.85


def test_world_pack_rejects_routine_block_unknown_location(tmp_path):
    data = _default_data()
    data["routine_blocks"] = {data["agents"][0]["id"]: [{
        "id": "bad_block",
        "window": ["17:00", "20:00"],
        "intent": "自由时间",
        "candidate_locations": ["missing-place"],
    }]}
    path = tmp_path / "invalid-routine-block.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="routine block.*unknown location"):
        load_world_pack(path)


def test_world_pack_rejects_overlapping_destination_buildings(tmp_path):
    data = _default_data()
    market = next(item for item in data["locations"] if item["id"] == "fresh_market")
    market["x"], market["y"] = 0, 0
    path = tmp_path / "overlapping-world.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="locations overlap"):
        load_world_pack(path)
