"""Inspect RGB-D target points for a supplied pixel box; no object-state access."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "libero"))
sys.path.insert(0, "/home/czx/LWM/LIBERO")

from camera_calibration import calibration_from_robosuite, localize_observation_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=float, nargs=4, required=True)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_id = 0
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(
        bddl_file_name=str(bddl),
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=128,
        camera_widths=128,
        camera_depths=True,
        use_object_obs=False,
        has_offscreen_renderer=True,
    )
    try:
        env.seed(0)
        env.reset()
        observation = env.set_init_state(suite.get_task_init_states(task_id)[0])
        calibration = calibration_from_robosuite(env)
        estimate, _ = localize_observation_bbox(env, observation, args.bbox, calibration)
        eef_world = np.append(observation["robot0_eef_pos"], 1.0)
        eef_base = np.linalg.inv(calibration.world_from_base) @ eef_world
        print("goal:", task.language)
        print("eef_pos_world:", np.round(observation["robot0_eef_pos"], 4).tolist())
        print("eef_pos_base:", np.round(eef_base[:3], 4).tolist())
        print("world_from_base:", np.round(calibration.world_from_base, 4).tolist())
        print("visible_surface_base:", np.round(estimate.position_base, 4).tolist())
        print("covariance_base:", np.round(estimate.covariance_base, 6).tolist())
        print("visible_surface_spread:", np.round(estimate.visible_surface_spread, 4).tolist())
        print("median_depth_m:", round(estimate.median_depth_m, 4))
    finally:
        env.close()


if __name__ == "__main__":
    main()
