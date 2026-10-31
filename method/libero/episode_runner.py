"""Episode loop for paired LIBERO arms with shared visual history."""

import json
from pathlib import Path

import numpy as np

from decision_runner import DecisionRunner


class _FixedSelector:
    def __init__(self, selection):
        self.selection = selection

    def select(self, **kwargs):
        return self.selection


def _compact(value):
    if isinstance(value, dict):
        return {
            key: _compact(item)
            for key, item in value.items()
            if not key.endswith(("_image", "_depth"))
        }
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class LiberoEpisodeRunner:
    def __init__(self, model, executor_factory, outcome_builder, max_decisions=30):
        if max_decisions < 1:
            raise ValueError("max_decisions must be positive")
        self.model = model
        self.executor_factory = executor_factory
        self.outcome_builder = outcome_builder
        self.max_decisions = max_decisions

    def run(self, env, arm, goal, initial_state, seed=0):
        if arm not in DecisionRunner.ARMS:
            raise ValueError("unknown arm: {}".format(arm))
        env.seed(seed)
        env.reset()
        observation = env.set_init_state(initial_state)
        history = []
        decisions = []

        for decision_index in range(self.max_decisions):
            usage_before = self.model.usage()
            proposal = self.model.propose(goal, observation, history)
            executor = self.executor_factory(proposal["detections"])
            decision_observation = dict(observation)
            decision_observation["visual_detections"] = proposal["detections"]
            decision_runner = DecisionRunner(
                executor=executor,
                selector=(
                    _FixedSelector(proposal["baseline_selection"])
                    if arm == "baseline"
                    else self.model
                ),
                outcome_builder=self.outcome_builder,
                world_model=self.model,
            )
            decision = decision_runner.run_step(
                env=env,
                arm=arm,
                goal=goal,
                observation=decision_observation,
                history=history,
                candidates=proposal["candidates"],
            )
            next_observation = decision["execution"].get(
                "final_observation", decision["execution"].get("observation")
            )
            if next_observation is None:
                raise ValueError("skill executor must return its final observation")
            usage_after = self.model.usage()
            entry = {
                "decision_index": decision_index,
                "detections": proposal["detections"],
                "decision": _compact(decision),
                "usage_delta": {
                    key: usage_after[key] - usage_before[key] for key in usage_after
                },
            }
            decisions.append(entry)
            history.append(
                {
                    "decision_index": decision_index,
                    "selected_skill": decision["execution"].get("skill"),
                    "selected_args": decision["execution"].get("entity_ids"),
                    "selection_rationale": decision["selection"].get("rationale"),
                    "execution_status": (
                        "low_level_sequence_completed_interaction_unverified"
                        if decision["execution"].get("status") == "succeeded"
                        else decision["execution"].get("status")
                    ),
                    "interaction_verification": "not_verified",
                    "task_success": decision["task_success"],
                }
            )
            observation = next_observation
            if decision["task_success"]:
                break
            if decision["execution"].get("done"):
                break

        return {
            "arm": arm,
            "goal": goal,
            "success": bool(env.check_success()),
            "decision_count": len(decisions),
            "model_usage": self.model.usage(),
            "decisions": decisions,
        }


def save_episode(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(_compact(result), output, indent=2, ensure_ascii=False)
        output.write("\n")
