"""Exercise bounded Panda primitives and restore their initial state."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "libero"))

from manipulation_skills import PickPlaceExecutor
from camera_calibration import calibration_from_robosuite
from skill_primitives import command_gripper, move_to
from state_branch import capture_snapshot, restore_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", default="/home/czx/LWM/LIBERO")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    args = parser.parse_args()

    libero_root = Path(args.libero_root).resolve()
    sys.path.insert(0, str(libero_root))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(
        bddl_file_name=str(bddl_path),
        use_camera_obs=False,
        use_object_obs=False,
        has_offscreen_renderer=False,
    )
    try:
        env.seed(0)
        env.reset()
        initial_state = suite.get_task_init_states(args.task_id)[args.init_state_id]
        observation = env.set_init_state(initial_state)
        snapshot = capture_snapshot(env, observation)

        start = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        calibration = calibration_from_robosuite(env)
        eef_base = (
            np.linalg.inv(calibration.world_from_base)
            @ np.append(start, 1.0)
        )[:3]
        target = start + np.array([0.04, 0.0, 0.0])
        outbound = move_to(env, target, observation=observation)
        assert outbound["status"] == "succeeded", outbound
        inbound = move_to(env, start, observation=outbound["observation"])
        assert inbound["status"] == "succeeded", inbound
        closed = command_gripper(env, close=True, observation=inbound["observation"])
        opened = command_gripper(env, close=False, observation=closed["observation"])
        assert closed["status"] == opened["status"] == "succeeded"

        observation = restore_snapshot(env, snapshot)
        target = eef_base + np.array([0.0, 0.0, 0.02])
        macros = PickPlaceExecutor(
            target_provider=lambda mode, entity_id, current: target,
            world_from_base=calibration.world_from_base,
            approach_height_m=0.10,
            lift_height_m=0.12,
        )
        pick_result = macros.pick(env, "probe_object", observation)
        assert pick_result["status"] == "succeeded", pick_result
        place_result = macros.place(
            env, "probe_object", "probe_target", pick_result["final_observation"]
        )
        assert place_result["status"] == "succeeded", place_result

        restore_snapshot(env, snapshot)
        print(
            "primitive and macro orchestration smoke check passed:",
            "outbound_steps={}, inbound_steps={}, gripper_steps={}, macro_primitives={}".format(
                len(outbound["trace"]),
                len(inbound["trace"]),
                len(closed["trace"]) + len(opened["trace"]),
                len(pick_result["primitive_trace"]) + len(place_result["primitive_trace"]),
            ),
            task.language,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
