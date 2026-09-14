"""Repeatable cognitive scenario evaluator.

Run offline constraints:
    python -m backend.town.cognitive_eval
Run live two-stage planner samples:
    python -m backend.town.cognitive_eval --live --repeats 3 --live-limit 3
"""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

from .cognition import ActionExpectation, ActionOutcome, FactEvent, Intention
from .engine import SimulationEngine
from .experience import MemoryFormation
from .world import location_center

SCENARIO_PATH = Path(__file__).resolve().parents[1] / "cognitive_scenarios" / "scenarios.json"


class CognitiveEvaluator:
    def __init__(self, live: bool = False, repeats: int = 1, live_limit: int = 0):
        self.live = live
        self.repeats = max(1, repeats)
        self.live_limit = max(0, live_limit)

    async def run(self) -> dict:
        scenarios = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
        results = []
        with tempfile.TemporaryDirectory(prefix="town-cognitive-eval-") as temp_dir:
            engine = SimulationEngine()
            engine.llm.fallback = not self.live
            engine.embedding_provider.fallback = True
            db_path = str(Path(temp_dir) / "eval.db")
            engine.memory.db_path = db_path
            engine.relationship_store.db_path = db_path
            engine.trace.db_path = db_path
            await engine.init()
            live_used = 0
            for scenario in scenarios:
                if scenario["check"] in {"memory_dialogue", "memory_dialogue_failure"}:
                    result = await self._run_memory_dialogue(engine, scenario)
                else:
                    result = self._run_offline(engine, scenario)
                if self.live and scenario["check"] == "interaction" and live_used < self.live_limit:
                    result["live"] = await self._run_live(engine, scenario)
                    live_used += 1
                results.append(result)
        passed = sum(1 for result in results if result["passed"] and result.get("live", {}).get("passed", True))
        return {
            "scenario_count": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 4) if results else 0.0,
            "results": results,
        }

    def _prepare(self, engine, scenario):
        engine.resources.reset_default_town()
        engine.dialogue._active_dialogues.clear()
        engine._dialogue_intents.clear()
        agent = next(item for item in engine.agents if item.id == scenario.get("agent", "hua"))
        agent.state.status = "IDLE"
        location = scenario.get("location", agent.state.current_location)
        agent.state.current_location = location
        agent.state.x, agent.state.y = location_center(location)
        if scenario.get("target"):
            target = next(item for item in engine.agents if item.id == scenario["target"])
            target.state.status = "IDLE"
            target.state.current_location = scenario.get("target_location", location)
            target.state.x, target.state.y = location_center(target.state.current_location)
        if scenario.get("remove_resource"):
            engine.resources.resources.pop(scenario["remove_resource"], None)
        for resource_id, quantity in scenario.get("resource_quantity", {}).items():
            resource = engine.resources.get(resource_id)
            if resource:
                resource.quantity = quantity
                resource.availability = "available" if quantity > 0 else "depleted"
        for resource_id, resource_location in scenario.get("resource_location", {}).items():
            resource = engine.resources.get(resource_id)
            if resource:
                resource.location = str(resource_location).replace("{agent}", agent.id)
        return agent

    def _run_offline(self, engine, scenario) -> dict:
        try:
            agent = self._prepare(engine, scenario)
            check = scenario["check"]
            detail = {}
            if check == "interaction":
                validation = engine.interactions.validate(agent, scenario["action"], engine, scenario["id"])
                passed = validation.feasible is scenario["feasible"]
                if passed and scenario.get("effect"):
                    passed = scenario["effect"] in {effect.get("type") for effect in validation.expected_effects}
                detail = validation.to_dict()
            elif check == "fact_visibility":
                fact = engine.fact_ledger.add(FactEvent(
                    sim_time="eval", sim_timestamp=scenario["fact_time"], type="eval",
                    agent_id="system", location="cafe", details={"scenario": scenario["id"]},
                    known_by=scenario["known_by"],
                ))
                visible = engine.fact_ledger.is_known_by(fact.id, agent.id, scenario["now"])
                passed = visible is scenario["visible"]
                detail = {"fact_id": fact.id, "visible": visible}
            elif check == "intention_lifecycle":
                intention = agent.mental_state.add_intention(Intention(
                    description=scenario["id"], source="eval", created_at=0,
                    interaction_type="work_on_goal",
                ))
                for status in scenario["sequence"]:
                    agent.mental_state.transition_intention(intention.id, status, 10)
                passed = intention.status == scenario["expected"]
                detail = asdict(intention)
            elif check == "goal_lifecycle":
                goal = agent.mental_state.goals[0]
                for status in scenario["sequence"]:
                    agent.mental_state.transition_goal(goal.id, status)
                passed = goal.status == scenario["expected"]
                detail = asdict(goal)
            elif check == "goal_progress":
                goal = agent.mental_state.goals[0]
                before = goal.progress
                applied = engine.interactions.apply_effects(agent, {}, [
                    {"type": "goal_progress", "goal_id": goal.id, "delta": 0.25}
                ])
                passed = goal.progress == min(1.0, before + 0.25)
                detail = {"before": before, "after": goal.progress, "effects": applied}
            elif check == "prediction_error":
                action_id = f"eval_{scenario['id']}"
                agent.mental_state.register_expectation(ActionExpectation(
                    action_id=action_id, option_id="eval",
                    expected_effects=scenario["expected_effects"], expected_duration=5,
                    assumptions=[], created_at=0,
                ))
                outcome = ActionOutcome(
                    action_id=action_id, actual_effects=scenario["actual_effects"],
                    duration=5, success=True, completed_at=5,
                )
                agent.mental_state.record_outcome(outcome)
                passed = outcome.prediction_error >= scenario["minimum_error"]
                detail = asdict(outcome)
            elif check == "memory_time":
                consistent = MemoryFormation._time_consistent(
                    scenario["content"], scenario["facts"], scenario["hour"]
                )
                passed = consistent is scenario["consistent"]
                detail = {"consistent": consistent}
            else:
                passed, detail = False, {"error": f"unknown check {check}"}
            return {"id": scenario["id"], "passed": passed, "detail": detail}
        except Exception as exc:
            return {"id": scenario["id"], "passed": False, "detail": {"error": str(exc)}}

    async def _run_memory_dialogue(self, engine, scenario) -> dict:
        try:
            agent = self._prepare(engine, scenario)
            target = next(item for item in engine.agents if item.id == scenario["target"])
            if scenario["check"] == "memory_dialogue":
                intention = agent.mental_state.add_intention(Intention(
                    description=scenario["intention"], source="memory",
                    created_at=engine.get_sim_timestamp(),
                    target_location=agent.state.current_location,
                ))
            else:
                intention = None
            engine._queue_encounter_dialogue(agent, target)
            events = await engine._run_group_dialogues(engine.get_sim_time_str())
            line = next((event for event in events if event["type"] == "dialogue_line"), None)
            if intention:
                passed = bool(line and line.get("trigger", {}).get("type") == "memory_intention"
                              and line["trigger"].get("intentionId") == intention.id)
            else:
                passed = not events and not engine.dialogue.participant_ids()
            return {"id": scenario["id"], "passed": passed,
                    "detail": {"event_types": [event["type"] for event in events]}}
        except Exception as exc:
            return {"id": scenario["id"], "passed": False, "detail": {"error": str(exc)}}

    async def _run_live(self, engine, scenario) -> dict:
        attempts = []
        for _ in range(self.repeats):
            agent = self._prepare(engine, scenario)
            decision = await agent.decide_action(engine.get_sim_time_str(), engine)
            validation = engine.interactions.validate(agent, decision, engine, "live")
            planning = decision.get("planning", {})
            proposal = planning.get("proposal", {})
            confirmed = (proposal.get("assessment") or {}).get("confirmed_fact_ids", [])
            valid_refs, invalid_refs = engine.fact_ledger.validate_references(
                confirmed, agent.id, engine.get_sim_timestamp()
            )
            passed = validation.feasible and not invalid_refs and bool(planning)
            attempts.append({
                "passed": passed, "decision": decision,
                "validation": validation.to_dict(), "valid_fact_ids": valid_refs,
                "invalid_fact_ids": invalid_refs,
            })
        passed_count = sum(1 for attempt in attempts if attempt["passed"])
        return {
            "passed": passed_count == len(attempts),
            "pass_rate": round(passed_count / len(attempts), 4),
            "attempts": attempts,
        }


async def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--live-limit", type=int, default=3)
    args = parser.parse_args()
    report = await CognitiveEvaluator(args.live, args.repeats, args.live_limit).run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(_main())
