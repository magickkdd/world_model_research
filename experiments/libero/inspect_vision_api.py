"""Check that the configured API accepts a real LIBERO image and schema prompt."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "libero"))
sys.path.insert(0, "/home/czx/LWM/LIBERO")

from model_adapter import OpenAICompatibleLiberoModel


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from pathlib import Path as FilePath

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_id = 0
    task = suite.get_task(task_id)
    bddl = FilePath(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
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
        model = OpenAICompatibleLiberoModel(timeout_s=120)
        proposal = model.propose(task.language, observation, history=[])
        print("goal:", task.language)
        print("detections:", proposal["detections"])
        print("candidates:", proposal["candidates"])
        print("baseline selection:", proposal["baseline_selection"])
        print("model usage:", model.usage())
    finally:
        env.close()


if __name__ == "__main__":
    main()
