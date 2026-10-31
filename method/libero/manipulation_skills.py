"""Agent-facing pick/place macros backed by closed-loop primitives."""

import numpy as np

from camera_calibration import localize_observation_bbox
from skill_primitives import command_gripper, move_to


SKILL_SCHEMAS = {
    "pick": {
        "type": "object",
        "properties": {"object_id": {"type": "string"}},
        "required": ["object_id"],
        "additionalProperties": False,
    },
    "place": {
        "type": "object",
        "properties": {
            "object_id": {"type": "string"},
            "target_id": {"type": "string"},
        },
        "required": ["object_id", "target_id"],
        "additionalProperties": False,
    },
}


def make_rgbd_target_provider(env, calibration, detection_provider, offset_provider):
    """Build macro targets from visual boxes plus calibrated grasp offsets.

    `detection_provider(entity_id, observation)` returns a pixel-space box.
    `offset_provider(mode, entity_id, estimate)` supplies a calibrated offset
    from the visible-surface point to the desired end-effector target.
    """

    def target_provider(mode, entity_id, observation):
        bbox = detection_provider(entity_id, observation)
        try:
            estimate, _ = localize_observation_bbox(
                env, observation, bbox, calibration
            )
        except ValueError as error:
            if "valid depth pixels" not in str(error):
                raise
            x1, y1, x2, y2 = [float(value) for value in bbox]
            pad_x = max(2.0, (x2 - x1) * 0.25)
            pad_y = max(2.0, (y2 - y1) * 0.25)
            height, width = observation["agentview_image"].shape[:2]
            expanded_bbox = [
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_y),
                min(float(width), x2 + pad_x),
                min(float(height), y2 + pad_y),
            ]
            estimate, _ = localize_observation_bbox(
                env, observation, expanded_bbox, calibration
            )
        offset = np.asarray(
            offset_provider(mode, entity_id, estimate), dtype=np.float64
        )
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("offset provider must return a finite 3-vector")
        return estimate.position_base + offset

    return target_provider


def make_candidate_executor_factory(
    env,
    calibration,
    offset_provider,
    approach_height_m=0.10,
    lift_height_m=0.12,
    max_move_steps=200,
):
    """Create per-decision executors bound to that step's visual detections."""

    def factory(detections):
        def detection_provider(entity_id, observation):
            if entity_id not in detections:
                raise ValueError("unknown visible entity: {}".format(entity_id))
            return detections[entity_id]["bbox_xyxy"]

        target_provider = make_rgbd_target_provider(
            env, calibration, detection_provider, offset_provider
        )
        macros = PickPlaceExecutor(
            target_provider,
            world_from_base=calibration.world_from_base,
            approach_height_m=approach_height_m,
            lift_height_m=lift_height_m,
            max_move_steps=max_move_steps,
        )

        def execute(current_env, candidate, observation):
            if candidate["skill"] == "pick":
                return macros.pick(
                    current_env, candidate["args"]["object_id"], observation
                )
            if candidate["skill"] == "place":
                return macros.place(
                    current_env,
                    candidate["args"]["object_id"],
                    candidate["args"]["target_id"],
                    observation,
                )
            raise ValueError("unsupported skill: {}".format(candidate["skill"]))

        return execute

    return factory


def skill_result_outcome(candidate, result, observation):
    """Map actual macro feedback into the shared WM outcome structure."""
    primitive_trace = result.get("primitive_trace", [])
    effects = [
        "{}:{}".format(step.get("command", "move_to"), step.get("status", "unknown"))
        for step in primitive_trace
    ]
    status = result.get("status", "failed")
    task_success = bool(result.get("task_success", False))
    skill = candidate["skill"]
    interaction_note = (
        "grasp is unverified"
        if skill == "pick"
        else "object placement is unverified"
    )
    failure_reason = None if status == "succeeded" else result.get(
        "termination_reason", "primitive_failed"
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "summary": "skill {} low-level sequence {}; {}; task_success={}".format(
            skill, status, interaction_note, task_success
        ),
        "effects": effects,
        "task_progress": "goal complete" if task_success else "goal remains incomplete",
        "task_success": task_success,
        "failure_reason": failure_reason or (
            "interaction_not_verified" if not task_success else None
        ),
    }


class PickPlaceExecutor:
    """Execute semantic skills using an injected visual grasp-target provider.

    The provider returns an end-effector target in robot-base coordinates from
    the current RGB-D observation. Agent calls contain entity IDs only.
    """

    def __init__(
        self,
        target_provider,
        world_from_base=None,
        approach_height_m=0.10,
        lift_height_m=0.12,
        position_tolerance_m=0.01,
        max_move_steps=60,
    ):
        if approach_height_m <= 0 or lift_height_m <= 0:
            raise ValueError("approach and lift heights must be positive")
        self.target_provider = target_provider
        self.world_from_base = (
            np.eye(4) if world_from_base is None else np.asarray(world_from_base, dtype=np.float64)
        )
        if self.world_from_base.shape != (4, 4) or not np.allclose(
            self.world_from_base[3], [0.0, 0.0, 0.0, 1.0]
        ):
            raise ValueError("world_from_base must be a 4x4 homogeneous transform")
        self.approach_height_m = float(approach_height_m)
        self.lift_height_m = float(lift_height_m)
        self.position_tolerance_m = float(position_tolerance_m)
        self.max_move_steps = int(max_move_steps)

    def _move(self, env, target, observation):
        target_world = (
            self.world_from_base @ np.append(np.asarray(target, dtype=np.float64), 1.0)
        )[:3]
        return move_to(
            env,
            target_world,
            observation=observation,
            tolerance_m=self.position_tolerance_m,
            max_steps=self.max_move_steps,
        )

    @staticmethod
    def _finish(env, name, entity_ids, steps, observation):
        successful = env.check_success()
        statuses = [step["status"] for step in steps]
        last_trace = steps[-1].get("trace", []) if steps else []
        environment_done = any(
            step.get("environment_terminated", False) for step in steps
        ) or (bool(last_trace[-1].get("done", False)) if last_trace else False)
        status = "succeeded" if all(value == "succeeded" for value in statuses) else "failed"
        return {
            "skill": name,
            "entity_ids": entity_ids,
            "status": status,
            "termination_reason": "primitive_sequence_complete" if status == "succeeded" else "primitive_failed",
            "task_success": bool(successful),
            "done": environment_done,
            "primitive_trace": steps,
            "final_observation": observation,
        }

    def pick(self, env, object_id, observation):
        target = np.asarray(
            self.target_provider("pick", object_id, observation), dtype=np.float64
        )
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target provider must return a finite base-frame 3-vector")
        approach = target + np.array([0.0, 0.0, self.approach_height_m])
        steps = []

        result = command_gripper(env, close=False, observation=observation)
        steps.append(result)
        observation = result["observation"]
        for waypoint in (approach, target):
            result = self._move(env, waypoint, observation)
            steps.append(result)
            observation = result["observation"]
            if result["status"] != "succeeded":
                break
        if all(step["status"] == "succeeded" for step in steps):
            result = command_gripper(env, close=True, observation=observation)
            steps.append(result)
            observation = result["observation"]
        if all(step["status"] == "succeeded" for step in steps):
            result = self._move(env, approach, observation)
            steps.append(result)
            observation = result["observation"]
        return self._finish(env, "pick", {"object_id": object_id}, steps, observation)

    def place(self, env, object_id, target_id, observation):
        target = np.asarray(
            self.target_provider("place", target_id, observation), dtype=np.float64
        )
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target provider must return a finite base-frame 3-vector")
        approach = target + np.array([0.0, 0.0, self.approach_height_m])
        retreat = target + np.array([0.0, 0.0, self.lift_height_m])
        steps = []

        for waypoint in (approach, target):
            result = self._move(env, waypoint, observation)
            steps.append(result)
            observation = result["observation"]
            if result["status"] != "succeeded":
                break
        if all(step["status"] == "succeeded" for step in steps):
            result = command_gripper(env, close=False, observation=observation)
            steps.append(result)
            observation = result["observation"]
        if all(step["status"] == "succeeded" for step in steps):
            result = self._move(env, retreat, observation)
            steps.append(result)
            observation = result["observation"]
        return self._finish(
            env,
            "place",
            {"object_id": object_id, "target_id": target_id},
            steps,
            observation,
        )
