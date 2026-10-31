"""One-step runner shared by baseline, language-WM, and simulator arms."""

from state_branch import capture_snapshot, restore_snapshot


_OUTCOME_FIELDS = {
    "candidate_id",
    "summary",
    "effects",
    "task_progress",
    "task_success",
    "failure_reason",
}


def _validate_candidates(candidates):
    if not candidates:
        raise ValueError("Agent returned no candidate skills")
    ids = [candidate.get("candidate_id") for candidate in candidates]
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in ids):
        raise ValueError("every candidate needs a non-empty candidate_id")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")
    return {candidate_id for candidate_id in ids}


def _validate_outcome(candidate_id, outcome):
    if not isinstance(outcome, dict) or set(outcome) != _OUTCOME_FIELDS:
        raise ValueError("candidate outcomes must match the shared outcome schema")
    if outcome["candidate_id"] != candidate_id:
        raise ValueError("outcome candidate_id does not match the candidate")
    if not isinstance(outcome["summary"], str) or not isinstance(outcome["effects"], list):
        raise ValueError("outcome summary/effects have invalid types")
    if not isinstance(outcome["task_progress"], str):
        raise ValueError("task_progress must be a string")
    if not isinstance(outcome["task_success"], bool):
        raise ValueError("task_success must be boolean")
    if outcome["failure_reason"] is not None and not isinstance(
        outcome["failure_reason"], str
    ):
        raise ValueError("failure_reason must be a string or null")
    return outcome


class DecisionRunner:
    """Evaluate candidates and execute one selected skill in the live branch.

    `selector.select` and `world_model.predict` are model adapters. Their
    rationale fields should contain concise decision explanations, not hidden
    chain-of-thought. `outcome_builder` maps actual skill feedback to the same
    schema returned by the language world model.
    """

    ARMS = {"baseline", "language_wm", "simulator_assisted"}

    def __init__(self, executor, selector, outcome_builder, world_model=None):
        self.executor = executor
        self.selector = selector
        self.outcome_builder = outcome_builder
        self.world_model = world_model

    def run_step(self, env, arm, goal, observation, history, candidates):
        if arm not in self.ARMS:
            raise ValueError("unknown arm: {}".format(arm))
        candidate_ids = _validate_candidates(candidates)
        outcomes = []
        snapshot = None

        if arm == "language_wm":
            if self.world_model is None:
                raise ValueError("language_wm arm requires a world_model adapter")
            for candidate in candidates:
                outcome = self.world_model.predict(
                    goal=goal,
                    observation=observation,
                    history=history,
                    candidate=candidate,
                )
                outcomes.append(
                    _validate_outcome(candidate["candidate_id"], outcome)
                )
        elif arm == "simulator_assisted":
            snapshot = capture_snapshot(env, observation)
            try:
                for candidate in candidates:
                    branch_observation = restore_snapshot(env, snapshot)
                    result = self.executor(env, candidate, branch_observation)
                    outcome = self.outcome_builder(
                        candidate=candidate,
                        result=result,
                        observation=result.get(
                            "final_observation", result.get("observation", branch_observation)
                        ),
                    )
                    outcomes.append(
                        _validate_outcome(candidate["candidate_id"], outcome)
                    )
            finally:
                restore_snapshot(env, snapshot)

        selection = self.selector.select(
            arm=arm,
            goal=goal,
            observation=observation,
            history=history,
            candidates=candidates,
            outcomes=outcomes if arm != "baseline" else None,
        )
        selected_id = selection.get("candidate_id") if isinstance(selection, dict) else None
        if selected_id not in candidate_ids:
            raise ValueError("selector chose a candidate outside the proposed set")
        selected = next(
            candidate for candidate in candidates if candidate["candidate_id"] == selected_id
        )

        if snapshot is not None:
            observation = restore_snapshot(env, snapshot)
        execution = self.executor(env, selected, observation)
        return {
            "arm": arm,
            "goal": goal,
            "candidates": candidates,
            "candidate_outcomes": outcomes,
            "selection": selection,
            "executed_candidate_id": selected_id,
            "execution": execution,
            "task_success": bool(env.check_success()),
        }
