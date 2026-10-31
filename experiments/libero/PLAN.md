# LIBERO Pilot Plan

## Sequence

1. **Done:** implement `capture_snapshot()` / `restore_snapshot()` and verify an A-B-A replay from one fixed state. LIBERO Spatial task 0, initial state 0, produced matching observation, reward, done flag, MuJoCo integration state, controller state, counters, observable cache, and RNG state after restoring and replaying A.
2. **Simulation pipeline wired:** RGB-D input is enabled with object observations disabled. `visual_pose.py` refines a detector-provided box with GrabCut, linearizes depth, and estimates a visible-surface point. `camera_calibration.py` derives fixed-camera and robot-base calibration from LIBERO's static setup; the helper has been exercised on a real RGB-D observation. `visual_detections.py` validates entity boxes. Hardware must use measured camera calibration.
3. **Macro interface implemented:** `move_to`, gripper commands, and Agent-facing `pick` / `place` macros record primitive steps. A stub target provider verified macro orchestration and snapshot restoration; this did not verify object grasp or placement. The target provider requires calibrated offsets from visible surfaces to end-effector grasp/release points.
4. **Experiment runner implemented:** `model_adapter.py` implements multimodal proposals, per-candidate language predictions, and short auditable selection rationales through the OpenAI-compatible API. `decision_runner.py` and `episode_runner.py` provide baseline, language-WM, and simulator-assisted branches with matched `CandidateOutcome` records. Mock adapter and three-arm routing checks pass. The configured API has been exercised with real LIBERO images.
5. **Task 0 pilot run and diagnosis:** baseline task 0, init state 0, did not complete within the 8-decision run. The recorded episode is `results/task0_baseline_verified.json`. Replaying its actions with simulator-only evaluation showed the bowl was not grasped during the first five pick sequences. The macro's low-level completion had been reported as success without verifying object interaction; history and prompts now label grasp/place as unverified and require image-based confirmation. The run also exposed unstable 128x128 model bounding boxes, with one offline localization estimate 3.6 cm from the simulator reference, and the episode reached the 1000-step environment horizon. The evaluator is `evaluate_recorded_episode.py`. These are pilot diagnostics, not a three-arm comparison or a successful task result.

Snapshot replay check:

```bash
/home/czx/miniconda3/envs/libero/bin/python experiments/libero/verify_snapshot.py --suite libero_spatial --task-id 0 --init-state-id 0
/home/czx/miniconda3/envs/libero/bin/python experiments/libero/verify_visual_pose.py
/home/czx/miniconda3/envs/libero/bin/python experiments/libero/verify_camera_calibration.py --suite libero_spatial --task-id 0 --init-state-id 0
/home/czx/miniconda3/envs/libero/bin/python experiments/libero/verify_skill_primitives.py --suite libero_spatial --task-id 0 --init-state-id 0
/home/czx/miniconda3/envs/libero/bin/python experiments/libero/verify_decision_runner.py --suite libero_spatial --task-id 0 --init-state-id 0
/home/czx/miniconda3/envs/libero/bin/python experiments/libero/verify_model_adapter.py
```

Single-arm entry point (requires `OPENAI_API_KEY` in the WSL process environment and calibrated base-frame offsets):

```bash
/home/czx/miniconda3/envs/libero/bin/python experiments/libero/run_single_episode.py --arm baseline --suite libero_spatial --task-id 0 --init-state-id 0 --pick-offset X Y Z --place-offset X Y Z --output experiments/libero/results/task0_baseline.json
```

## Measures

- Task success and executed control steps.
- Per-candidate WM prediction accuracy against simulator outcomes.
- Simulator-assisted action choice versus language-WM and baseline choice.
- Perception pose error, uncertainty calibration, and latency.
- Model calls, tokens, and wall-clock time.

All arms share candidate skills and perception estimates. Only the information used to select an action differs. Simulator rollout outcomes are labeled as privileged simulator feedback.
