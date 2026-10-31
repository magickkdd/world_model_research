# Method

The Agent proposes calls from a pre-implemented skill registry. Initial manipulation candidates are `pick(object_id)` and `place(object_id, target_id)` macros; their Python implementations compose lower-level motion and gripper skills and retain a full primitive trace.

The observation combines task language, camera images, robot proprioception, and camera-based object-pose estimates. A perception module estimates object poses and uncertainty. Skills refresh estimates before motion. No simulator ground-truth object coordinates are supplied to the Agent or language WM.

The language WM predicts candidate effects from the current observation and skill call. The simulator-assisted arm branches the environment, executes each candidate skill, observes the actual simulator result, restores the state, and gives those results to the selector. See [LIBERO interface](libero/INTERFACE.md).

The reusable implementation includes `libero/state_branch.py` for deterministic state branching, `libero/visual_pose.py` and `libero/camera_calibration.py` for RGB-D localization, `libero/skill_primitives.py` for bounded Panda motion and gripper commands, and `libero/manipulation_skills.py` for Agent-facing `pick` / `place` macros. The macros accept semantic entity IDs; a visual target provider converts the current RGB-D observation into end-effector targets. That provider still needs task-level grasp-pose calibration and validation.
