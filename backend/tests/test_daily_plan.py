import asyncio

import pytest

from backend.town.daily_plan import parse_daily_plan


def test_daily_plan_parses_non_overlapping_time_blocks():
    plan = parse_daily_plan({
        "focus": "完成上午职责，晚上留出个人时间",
        "review_after_minutes": 180,
        "blocks": [
            {"id": "morning", "window": [360, 720], "intent": "上午工作", "priority": 0.9},
            {"id": "evening", "window": [1020, 1320], "intent": "晚上休息或社交", "flexibility": 0.9},
        ],
    }, "mei", 1, 360)
    assert plan.current_block(400).id == "morning"
    assert plan.current_block(1000) is None
    assert plan.current_block(1100).status == "active"
    assert plan.mark_completed("evening")
    assert plan.current_block(1100) is None
    assert plan.to_dict()["day"] == 1


def test_daily_plan_rejects_overlapping_blocks():
    with pytest.raises(ValueError, match="overlap"):
        parse_daily_plan({
            "blocks": [
                {"id": "a", "window": [360, 600], "intent": "工作"},
                {"id": "b", "window": [590, 720], "intent": "午餐"},
            ]
        }, "mei", 1, 360)


def test_daily_plan_skips_invalid_blocks_but_requires_one_valid_block():
    plan = parse_daily_plan({
        "blocks": [
            {"window": ["bad", 600], "intent": "无效"},
            {"id": "valid", "window": [600, 720], "intent": "午餐"},
        ]
    }, "mei", 1, 600)
    assert [block.id for block in plan.blocks] == ["valid"]

    with pytest.raises(ValueError, match="no valid blocks"):
        parse_daily_plan({"blocks": [{"window": [360, 600]}]}, "mei", 1, 360)
