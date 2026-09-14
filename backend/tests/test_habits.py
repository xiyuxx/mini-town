from backend.town.habits import HabitStore


def routine_action(location="bookstore", activity="逛书店"):
    return {
        "source": "routine_block", "routine_block_id": "mei_evening",
        "location": location, "content": activity,
    }


def test_only_voluntary_routine_block_actions_form_habits():
    store = HabitStore()
    assert store.record("mei", {"source": "routine", "content": "上班"}, 100, True) is None

    habit = store.record("mei", routine_action(), 100, True)
    assert habit is not None
    assert habit.strength == 0.14
    assert store.strength_for("mei", "mei_evening", "bookstore", "逛书店") == 0.14


def test_habit_success_strengthens_failure_weakens_and_decay_applies():
    store = HabitStore()
    first = store.record("mei", routine_action(), 100, True)
    first_strength = first.strength
    second = store.record("mei", routine_action(), 200, True)
    assert second.strength > first_strength

    second_strength = second.strength
    weakened = store.record("mei", routine_action(), 300, False)
    assert weakened.strength < second_strength
    before_decay = weakened.strength
    store.decay("mei", 300 + 10 * 1440)
    assert store.for_agent("mei")[0].strength < before_decay


def test_habit_store_is_bounded_per_agent():
    store = HabitStore(max_per_agent=2)
    for index in range(3):
        store.record("mei", routine_action(f"location_{index}", f"活动{index}"), index + 1, True)
    assert len(store.for_agent("mei")) == 2
