"""Exercise all three decision arms with deterministic mock model adapters."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHOD = ROOT / "method" / "libero"
sys.path.insert(0, str(METHOD))

from decision_runner import DecisionRunner
from state_branch import capture_snapshot, restore_snapshot


class MockSelector:
    def select(self, **kwargs):
        if kwargs["arm"] != "baseline":
            assert len(kwargs["outcomes"]) == len(kwargs["candidates"])
        return {"candidate_id": "candidate_a", "rationale": "mock choice"}


class MockWorldModel:
    def predict(self, goal, observation, history, candidate):
        return {
            "candidate_id": candidate["candidate_id"],
            "summary": "mock predicted feedback",
            "effects": [],
            "task_progress": "unchanged",
            "task_success": False,
            "failure_reason": None,
        }


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
        state = suite.get_task_init_states(args.task_id)[args.init_state_id]
        observation = env.set_init_state(state)
        initial = capture_snapshot(env, observation)

        def execute(current_env, candidate, current_observation):
            action = np.zeros(current_env.env.action_dim)
            action[0] = 0.2 if candidate["candidate_id"] == "candidate_a" else -0.2
            next_observation, reward, done, info = current_env.step(action)
            return {
                "final_observation": next_observation,
                "reward": float(reward),
                "done": bool(done),
                "info": info,
            }

        def outcome_builder(candidate, result, observation):
            return {
                "candidate_id": candidate["candidate_id"],
                "summary": "reward={:.3f}, done={}".format(result["reward"], result["done"]),
                "effects": [],
                "task_progress": "unchanged",
                "task_success": False,
                "failure_reason": None,
            }

        runner = DecisionRunner(
            executor=execute,
            selector=MockSelector(),
            outcome_builder=outcome_builder,
            world_model=MockWorldModel(),
        )
        candidates = [
            {"candidate_id": "candidate_a", "skill": "pick", "args": {"object_id": "x"}},
            {"candidate_id": "candidate_b", "skill": "place", "args": {"object_id": "x", "target_id": "y"}},
        ]
        for arm in sorted(runner.ARMS):
            observation = restore_snapshot(env, initial)
            trace = runner.run_step(
                env, arm, task.language, observation, history=[], candidates=candidates
            )
            assert trace["executed_candidate_id"] == "candidate_a"
            assert trace["execution"]["done"] is False
        restore_snapshot(env, initial)
        print("three-arm decision runner smoke check passed:", ", ".join(sorted(runner.ARMS)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
