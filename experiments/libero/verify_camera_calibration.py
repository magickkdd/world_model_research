"""Check fixed-camera calibration and RGB-D localization wiring."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "libero"))

from camera_calibration import calibration_from_robosuite, localize_observation_bbox


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
        state = suite.get_task_init_states(args.task_id)[args.init_state_id]
        observation = env.set_init_state(state)
        calibration = calibration_from_robosuite(env)
        transform = calibration.base_from_camera
        np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6
        )
        estimate, mask = localize_observation_bbox(
            env, observation, [48, 48, 80, 80], calibration
        )
        assert mask.any()
        assert np.all(np.isfinite(estimate.position_base))
        assert np.all(np.isfinite(estimate.covariance_base))
        print(
            "camera calibration and RGB-D wiring passed:",
            "position_base={}, valid_pixels={}".format(
                np.round(estimate.position_base, 4).tolist(),
                estimate.valid_depth_pixels,
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
