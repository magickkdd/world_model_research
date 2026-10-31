"""Small, closed-loop motion and gripper primitives for LIBERO's Panda arm."""

import time

import numpy as np


def _eef_position(observation):
    return np.asarray(observation["robot0_eef_pos"], dtype=np.float64)


def _environment_terminated(env):
    core = getattr(env, "env", env)
    return bool(getattr(core, "done", False)) or int(
        getattr(core, "timestep", 0)
    ) >= int(getattr(core, "horizon", 2**63 - 1))


def _step_record(index, action, observation, reward, done, info):
    return {
        "index": index,
        "action": np.asarray(action, dtype=np.float64).copy(),
        "eef_position": _eef_position(observation).copy(),
        "reward": float(reward),
        "done": bool(done),
        "info": dict(info),
    }


def move_to(
    env,
    target_position,
    observation=None,
    tolerance_m=0.01,
    max_steps=60,
    max_delta_m=0.01,
):
    """Move the end effector toward a world-frame point with bounded increments."""
    started = time.perf_counter()
    target = np.asarray(target_position, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_position must be a finite 3-vector")
    if tolerance_m <= 0 or max_steps < 1 or max_delta_m <= 0:
        raise ValueError("tolerance, max_steps, and max_delta_m must be positive")

    robot = env.robots[0]
    controller = robot.controller
    translation_scale = np.asarray(controller.output_max[:3], dtype=np.float64)
    if np.any(translation_scale <= 0):
        raise ValueError("controller translation output limits must be positive")
    observation = observation if observation is not None else env._get_observations()
    start = _eef_position(observation).copy()
    trace = []
    status = "timed_out"
    termination_reason = "timed_out"

    for index in range(max_steps):
        if _environment_terminated(env):
            status = "failed"
            termination_reason = "environment_terminated"
            break
        current = _eef_position(observation)
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if distance <= tolerance_m:
            status = "succeeded"
            break

        command_delta = delta * min(max_delta_m / distance, 1.0)
        action = np.zeros(env.env.action_dim, dtype=np.float64)
        action[:3] = command_delta / translation_scale
        observation, reward, done, info = env.step(action)
        trace.append(_step_record(index, action, observation, reward, done, info))
        if done:
            status = "succeeded" if env.check_success() else "failed"
            termination_reason = "task_success" if status == "succeeded" else "environment_done"
            break

    final_position = _eef_position(observation).copy()
    if np.linalg.norm(target - final_position) <= tolerance_m:
        status = "succeeded"
    return {
        "status": status,
        "termination_reason": (
            "target_tolerance" if status == "succeeded" else termination_reason
        ),
        "environment_terminated": _environment_terminated(env),
        "start_position": start,
        "target_position": target,
        "final_position": final_position,
        "final_error_m": float(np.linalg.norm(target - final_position)),
        "duration_ms": (time.perf_counter() - started) * 1000.0,
        "trace": trace,
        "observation": observation,
    }


def command_gripper(env, close, observation=None, steps=8):
    """Issue a bounded open or close command and retain every observation."""
    if steps < 1:
        raise ValueError("steps must be positive")
    started = time.perf_counter()
    observation = observation if observation is not None else env._get_observations()
    command = 1.0 if close else -1.0
    action = np.zeros(env.env.action_dim, dtype=np.float64)
    action[-1] = command
    trace = []
    for index in range(steps):
        if _environment_terminated(env):
            return {
                "status": "failed",
                "termination_reason": "environment_terminated",
                "environment_terminated": True,
                "duration_ms": (time.perf_counter() - started) * 1000.0,
                "trace": trace,
                "observation": observation,
            }
        observation, reward, done, info = env.step(action)
        trace.append(_step_record(index, action, observation, reward, done, info))
        if done:
            successful = env.check_success()
            return {
                "status": "succeeded" if successful else "failed",
                "termination_reason": "task_success" if successful else "environment_done",
                "environment_terminated": _environment_terminated(env),
                "duration_ms": (time.perf_counter() - started) * 1000.0,
                "trace": trace,
                "observation": observation,
            }
    return {
        "status": "succeeded",
        "termination_reason": "command_duration_elapsed",
        "environment_terminated": _environment_terminated(env),
        "command": "close" if close else "open",
        "duration_ms": (time.perf_counter() - started) * 1000.0,
        "trace": trace,
        "observation": observation,
    }
