import asyncio

from backend.town.cognitive_eval import CognitiveEvaluator


def test_all_cognitive_scenarios_pass_offline():
    report = asyncio.run(CognitiveEvaluator(live=False).run())
    failures = [item for item in report["results"] if not item["passed"]]
    assert report["scenario_count"] == 22
    assert not failures, failures
    assert report["pass_rate"] == 1.0
