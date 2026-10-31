"""Check that a LIBERO branch can be restored and replayed consistently."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "method" / "libero"))

from state_branch import LiberoSnapshot, capture_snapshot, restore_snapshot


def assert_values_equal(left, right, path="state"):
    if isinstance(left, np.ndarray):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-7, err_msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_values_equal(left[key], right[key], "{}.{}".format(path, key))
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_values_equal(left_item, right_item, "{}[{}]".format(path, index))
    else:
        assert left == right, path


def assert_snapshots_equal(left, right):
    for name in LiberoSnapshot.__dataclass_fields__:
        assert_values_equal(getattr(left, name), getattr(right, name), name)


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
    bddl_path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
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
        initial_state = suite.get_task_init_states(args.task_id)[args.init_state_id]
        observation = env.set_init_state(initial_state)
        allowed_keys = {
            "agentview_image",
            "agentview_depth",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_proprio-state",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        }
        unexpected = set(observation) - allowed_keys
        assert not unexpected, "unexpected observation keys: {}".format(sorted(unexpected))
        assert {"agentview_image", "agentview_depth"} <= set(observation)
        assert observation["agentview_image"].shape == (128, 128, 3)
        assert observation["agentview_depth"].shape == (128, 128, 1)
        print("observation keys:", sorted(observation))
        snapshot = capture_snapshot(env, observation)

        action_a = np.zeros(env.env.action_dim)
        action_a[-1] = -1
        action_b = np.zeros(env.env.action_dim)
        action_b[0] = 0.2
        action_b[-1] = 1

        observation_a1, reward_a1, done_a1, _ = env.step(action_a)
        result_a1 = capture_snapshot(env, observation_a1)
        restore_snapshot(env, snapshot)
        env.step(action_b)
        restore_snapshot(env, snapshot)
        observation_a2, reward_a2, done_a2, _ = env.step(action_a)
        result_a2 = capture_snapshot(env, observation_a2)

        assert_snapshots_equal(result_a1, result_a2)
        assert reward_a1 == reward_a2
        assert done_a1 == done_a2
        print(
            "snapshot replay matched:",
            args.suite,
            args.task_id,
            args.init_state_id,
            task.language,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
