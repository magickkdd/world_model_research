# LIBERO Experiment Interface

## Design

The Agent selects calls from a pre-implemented skill registry. It cannot invent Python code or send raw MuJoCo controls. The Agent-facing manipulation skills are `pick(object_id)` and `place(object_id, target_id)`. These macros compose lower-level skills such as `move_to`, `open_gripper`, and `close_gripper`, and record every substep.

LIBERO provides the Panda simulation, OSC controller, observations, and task-success checks. It does not provide semantic `pick`, `place`, or `push` skills. Those are project code. This experiment does not train a policy.

## Observation and Perception

```text
Observation:
  task_text: str
  rgb: {agentview: image, wrist: image}
  robot:
    joint_position: float[]
    joint_velocity: float[]
    gripper_position: float[]
    end_effector_position: float[3]
    end_effector_orientation: float[4]
  detections: ObjectEstimate[]

ObjectEstimate:
  id: str
  category: str
  pose_in_robot_base: Pose
  pose_covariance: float[][]
  visible: bool
  source_camera: str
  latency_ms: float
```

The observation pipeline calls a fast machine-vision estimator each turn and includes its estimates for the Agent and language WM. The Agent may also request `estimate_pose(entity_id, image)` when it needs a fresh estimate, for example after occlusion. A semantic image detector supplies a 2D box; `visual_pose.py` refines it into a mask with GrabCut, then combines that mask with metric depth, camera intrinsics, and an explicitly supplied camera-to-base calibration to estimate a visible-surface point, sensor covariance, and latency. `localize_bbox()` is the combined entry point. It does not read simulator object state or segmentation buffers. This is a non-mutating perception tool, not a manipulation candidate. A macro skill refreshes the estimate before executing motion so it does not rely on stale coordinates.

For real hardware, pose precision depends on calibration, depth quality, and object/grasp-pose estimation. Record pose uncertainty and latency so perception can be evaluated separately from planning. The current LIBERO pilot enables 128x128 RGB-D. The Agent supplies an object bounding box in image pixels; `calibration_from_robosuite()` provides fixed-camera calibration for simulation, while hardware must use measured calibration. `localize_observation_bbox()` converts the box and RGB-D into a base-frame visible-surface estimate without simulator object state. Simulator-only segmentation labels are not used as perception input. The estimated point is the visible surface location, not the hidden object-frame origin; the grasp planner must account for this distinction.

## Skill Registry

Each registered skill has a name, description, JSON Schema for arguments, executor, and environment-side termination logic.

```text
SkillSpec:
  name: str
  description: str
  args_schema: JSONSchema
  execute(env, perception, args) -> SkillResult
```

The Agent may only reference registered skill names and perceived entity IDs. The runtime validates each call against the schema. The executor decides skill completion/failure from observations, robot state, task success, and a fixed time limit; the Agent does not provide stop conditions.

Example proposal:

```json
{"candidates": [
  {"id": "c1", "skill": "pick", "args": {"object_id": "mug_1"}},
  {"id": "c2", "skill": "place", "args": {"object_id": "mug_1", "target_id": "tray_1"}}
]}
```

The Agent uses only the `pick` and `place` schemas in `manipulation_skills.py`; it sends entity IDs, never coordinates or raw controls. Visual detections follow `DETECTION_SCHEMA` in `visual_detections.py` and are checked for unique IDs, valid confidence, and image-bounded boxes. The runtime's target provider maps the ID to the current image box, calls RGB-D localization, and applies a calibrated tool/grasp offset before invoking the macro. It transforms base-frame targets into world coordinates for LIBERO's OSC controller. The box can be returned as structured visual detections by the same vision-capable Agent. Offset calibration is part of the manipulation setup and must be measured before task evaluation.

## Macro Skills and Trace

`pick(object_id)` obtains an estimated pose from perception, computes a pre-grasp target, and invokes motion/gripper primitives. `place(object_id, target_id)` estimates the object and target poses, moves to a placement pose, and opens the gripper. Neither macro reads simulator ground-truth object coordinates.

```text
SkillResult:
  status: succeeded | failed | timed_out
  termination_reason: str
  primitive_trace: PrimitiveStep[]
  final_observation: Observation
  task_success: bool
```

Each primitive trace records the primitive skill and arguments, start/end observation references, duration, and environment feedback. The Agent chooses a macro while logs retain every `move_to` and gripper substep.

## World-Model Arms

All arms use the same task, observations, skill registry, candidate generator, and executor.

- **Baseline:** Agent selects a candidate using the current observation.
- **Language WM:** the language model predicts each candidate's effects from the observation and skill call. It does not execute candidates; Agent selects using the predictions.
- **Simulator-assisted:** save state, execute each candidate skill in the simulator, extract its actual result, restore state, then let Agent select using those results. This arm is explicitly privileged by simulator access.

Use the same outcome shape for predictions and simulator results:

```text
CandidateOutcome:
  candidate_id: str
  summary: str
  effects: str[]
  task_progress: str
  task_success: bool
  failure_reason: str | null
```

Log the source (`language_prediction` or `simulator_rollout`). Simulator state may derive rollout outcomes internally but is never added to the language-WM input.

## State Branching and Logs

```text
snapshot() -> Snapshot
restore(snapshot) -> Observation
execute(skill_call) -> SkillResult
```

Snapshots restore task, robot, controller, environment-counter, and relevant random-generator state. Every candidate starts from the same snapshot. After all rollouts, restore once more before executing the selected skill in the real trajectory.

For every decision, log the observation reference, candidates, WM predictions or simulator outcomes, selected candidate, macro and primitive traces, final observation, task success, and environment step count. This supports success/efficiency comparisons and per-candidate prediction accuracy analysis.

`decision_runner.py` executes the three arms through one shared interface. Baseline selects directly from candidates; language WM supplies predicted `CandidateOutcome` values; simulator-assisted executes each candidate from the same snapshot and converts real feedback into that outcome shape before selection. The selected skill is then executed once in the restored live branch. The runner checks that the selector chooses a proposed candidate.
