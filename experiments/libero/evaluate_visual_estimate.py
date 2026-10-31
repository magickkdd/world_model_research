"""Offline-only geometry diagnostic; object truth is never sent to the Agent."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "libero"))
sys.path.insert(0, "/home/czx/LWM/LIBERO")

from camera_calibration import calibration_from_robosuite, localize_observation_bbox
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=float, nargs=4, required=True)
    parser.add_argument("--contains", default="bowl")
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
        estimate, mask = localize_observation_bbox(env, observation, args.bbox, calibration)
        print("Agent input estimate base:", np.round(estimate.position_base, 4).tolist())
        print("mask pixels:", estimate.valid_depth_pixels)
        core = env.env
        matches = [
            (name, obj)
            for name, obj in core.objects_dict.items()
            if args.contains.lower() in name.lower()
        ]
        if not matches:
            print("No matching simulator object found for offline comparison")
            return
        world_from_base = calibration.world_from_base
        world_from_camera = get_camera_extrinsic_matrix(core.sim, "agentview")
        camera_from_world = np.linalg.inv(world_from_camera)
        K = calibration.intrinsics
        for name, obj in matches:
            body_id = core.sim.model.body_name2id(obj.root_body)
            position_world = np.asarray(core.sim.data.body_xpos[body_id])
            position_base = (
                np.linalg.inv(world_from_base) @ np.append(position_world, 1.0)
            )[:3]
            position_camera = (camera_from_world @ np.append(position_world, 1.0))[:3]
            projected = K @ position_camera
            projected = projected[:2] / projected[2]
            print(
                "offline simulator reference {} base: {} projected_pixel: {} delta_estimate_minus_ref: {}".format(
                    name,
                    np.round(position_base, 4).tolist(),
                    np.round(projected, 2).tolist(),
                    np.round(estimate.position_base - position_base, 4).tolist(),
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
