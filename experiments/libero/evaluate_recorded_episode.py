"""Replay a saved episode and report simulator-only object state for diagnosis."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _object_position(core, name_fragment):
    matches = [
        (name, obj)
        for name, obj in core.objects_dict.items()
        if name_fragment in name.lower()
    ]
    result = {}
    for name, obj in matches:
        body_id = core.sim.model.body_name2id(obj.root_body)
        result[name] = np.asarray(core.sim.data.body_xpos[body_id]).copy()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_json", type=Path)
    parser.add_argument("--libero-root", default="/home/czx/LWM/LIBERO")
    args = parser.parse_args()
    episode = json.loads(args.episode_json.read_text(encoding="utf-8"))

    sys.path.insert(0, str(Path(args.libero_root).resolve()))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()[episode["suite"]]()
    task = suite.get_task(episode["task_id"])
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(
        bddl_file_name=str(bddl),
        use_camera_obs=False,
        use_object_obs=False,
        has_offscreen_renderer=False,
    )
    try:
        env.seed(episode["seed"])
        env.reset()
        env.set_init_state(
            suite.get_task_init_states(episode["task_id"])[episode["init_state_id"]]
        )
        core = env.env
        initial_bowl = _object_position(core, "black_bowl")
        initial_plate = _object_position(core, "plate")
        print("initial bowl:", {k: v.round(4).tolist() for k, v in initial_bowl.items()})
        print("initial plate:", {k: v.round(4).tolist() for k, v in initial_plate.items()})

        for decision in episode["decisions"]:
            for primitive in decision["decision"]["execution"].get(
                "primitive_trace", []
            ):
                for step in primitive.get("trace", []):
                    if core.done or core.timestep >= core.horizon:
                        break
                    env.step(np.asarray(step["action"], dtype=np.float64))
            bowl = _object_position(core, "black_bowl")
            plate = _object_position(core, "plate")
            print(
                "after decision {}: success={}, bowl={}, plate={}".format(
                    decision["decision_index"],
                    bool(env.check_success()),
                    {k: v.round(4).tolist() for k, v in bowl.items()},
                    {k: v.round(4).tolist() for k, v in plate.items()},
                )
            )
            if core.done or core.timestep >= core.horizon:
                print("replay stopped at simulator termination, timestep", core.timestep)
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
