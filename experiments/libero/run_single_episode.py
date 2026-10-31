"""Run one LIBERO task episode for a selected experimental arm."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHOD = ROOT / "method" / "libero"
sys.path.insert(0, str(METHOD))

from camera_calibration import calibration_from_robosuite
from episode_runner import LiberoEpisodeRunner, save_episode
from manipulation_skills import (
    make_candidate_executor_factory,
    skill_result_outcome,
)
from model_adapter import OpenAICompatibleLiberoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", default="/home/czx/LWM/LIBERO")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument(
        "--arm",
        choices=["baseline", "language_wm", "simulator_assisted"],
        required=True,
    )
    parser.add_argument("--max-decisions", type=int, default=30)
    parser.add_argument("--max-move-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pick-offset", type=float, nargs=3, required=True)
    parser.add_argument("--place-offset", type=float, nargs=3, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON output path for this run",
    )
    args = parser.parse_args()

    libero_root = Path(args.libero_root).resolve()
    sys.path.insert(0, str(libero_root))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.init_state_id < len(initial_states):
        raise ValueError("init-state-id is outside this task's available states")
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
        env.seed(args.seed)
        env.reset()
        env.set_init_state(initial_states[args.init_state_id])
        calibration = calibration_from_robosuite(env)
        offsets = {
            "pick": np.asarray(args.pick_offset, dtype=np.float64),
            "place": np.asarray(args.place_offset, dtype=np.float64),
        }

        def offset_provider(mode, entity_id, estimate):
            return offsets[mode]

        model = OpenAICompatibleLiberoModel()
        executor_factory = make_candidate_executor_factory(
            env,
            calibration,
            offset_provider,
            max_move_steps=args.max_move_steps,
        )
        runner = LiberoEpisodeRunner(
            model=model,
            executor_factory=executor_factory,
            outcome_builder=skill_result_outcome,
            max_decisions=args.max_decisions,
        )
        result = runner.run(
            env,
            arm=args.arm,
            goal=task.language,
            initial_state=initial_states[args.init_state_id],
            seed=args.seed,
        )
        result.update(
            {
                "suite": args.suite,
                "task_id": args.task_id,
                "init_state_id": args.init_state_id,
                "seed": args.seed,
                "model": model.model,
                "base_url": model.base_url,
                "offsets_base_frame_m": {
                    key: value.tolist() for key, value in offsets.items()
                },
                "max_move_steps": args.max_move_steps,
            }
        )
        save_episode(args.output, result)
        print(
            "episode complete: arm={}, success={}, decisions={}, model_usage={}".format(
                result["arm"],
                result["success"],
                result["decision_count"],
                result["model_usage"],
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
