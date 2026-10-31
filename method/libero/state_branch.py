"""Capture and restore a LIBERO environment for same-state action rollouts."""

import copy
import random
from dataclasses import dataclass

import mujoco
import numpy as np


_MODEL_FIELDS = (
    "body_pos",
    "body_quat",
    "geom_pos",
    "geom_quat",
    "geom_rgba",
    "mat_rgba",
    "site_pos",
    "site_quat",
    "site_rgba",
)
_OBJECT_CONTAINERS = ("objects_dict", "fixtures_dict", "object_sites_dict")
_ENV_FIELDS = ("cur_time", "timestep", "done")
_OBSERVABLE_FIELDS = (
    "_time_since_last_sample",
    "_current_delay",
    "_current_observed_value",
    "_sampled",
)
_OBJECT_REFERENCES = {"sim", "env", "robot", "model", "_model", "_data", "viewer"}
_STATEFUL_MODULE_PREFIXES = (
    "robosuite.controllers.interpolators",
    "robosuite.utils.buffers",
)


@dataclass
class LiberoSnapshot:
    integration_state: np.ndarray
    model_state: dict
    environment_state: dict
    robot_state: list
    observable_state: dict
    observation_cache: dict
    object_properties: dict
    numpy_rng_state: tuple
    python_rng_state: object
    observation: object


def _safe_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.copy()
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, tuple):
        return tuple(_safe_value(item) for item in value)
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {copy.deepcopy(key): _safe_value(item) for key, item in value.items()}
    return None


def _capture_object_fields(obj, depth=0):
    captured = {}
    for name, value in vars(obj).items():
        if name in _OBJECT_REFERENCES:
            continue
        saved = _safe_value(value)
        if saved is not None or value is None:
            captured[name] = saved
        elif depth < 2 and any(
            value.__class__.__module__.startswith(prefix)
            for prefix in _STATEFUL_MODULE_PREFIXES
        ):
            captured[name] = _capture_object_fields(value, depth + 1)
    return captured


def _restore_object_fields(obj, state):
    for name, value in state.items():
        current = getattr(obj, name, None)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            _restore_object_fields(current, value)
        else:
            setattr(obj, name, copy.deepcopy(value))


def _core_env(env):
    return getattr(env, "env", env)


def _capture_integration_state(core):
    sim = core.sim
    model = sim.model._model
    data = sim.data._data
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_spec)
    return state


def capture_snapshot(env, observation=None):
    core = _core_env(env)
    sim = core.sim
    model_state = {
        name: np.array(getattr(sim.model, name), copy=True)
        for name in _MODEL_FIELDS
        if hasattr(sim.model, name)
    }
    robot_state = []
    for robot in core.robots:
        state = {"controller": _capture_object_fields(robot.controller)}
        if hasattr(robot, "gripper"):
            state["gripper"] = _capture_object_fields(robot.gripper)
        state["buffers"] = {
            name: _capture_object_fields(value)
            for name, value in vars(robot).items()
            if name.startswith("recent_") and hasattr(value, "__dict__")
        }
        robot_state.append(state)

    object_properties = {}
    for container_name in _OBJECT_CONTAINERS:
        container = getattr(core, container_name, {})
        object_properties[container_name] = {
            name: copy.deepcopy(obj.object_properties)
            for name, obj in container.items()
            if hasattr(obj, "object_properties")
        }

    observable_state = {
        name: {
            field: copy.deepcopy(getattr(observable, field))
            for field in _OBSERVABLE_FIELDS
        }
        for name, observable in core._observables.items()
    }
    if observation is None:
        observation = core._get_observations()

    return LiberoSnapshot(
        integration_state=_capture_integration_state(core),
        model_state=model_state,
        environment_state={name: copy.deepcopy(getattr(core, name)) for name in _ENV_FIELDS},
        robot_state=robot_state,
        observable_state=observable_state,
        observation_cache=copy.deepcopy(core._obs_cache),
        object_properties=object_properties,
        numpy_rng_state=copy.deepcopy(np.random.get_state()),
        python_rng_state=random.getstate(),
        observation=copy.deepcopy(observation),
    )


def restore_snapshot(env, snapshot):
    core = _core_env(env)
    sim = core.sim
    model = sim.model._model
    data = sim.data._data

    for name, value in snapshot.model_state.items():
        getattr(sim.model, name)[:] = value
    mujoco.mj_setState(
        model,
        data,
        snapshot.integration_state,
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    mujoco.mj_forward(model, data)

    for name, value in snapshot.environment_state.items():
        setattr(core, name, copy.deepcopy(value))
    for robot, state in zip(core.robots, snapshot.robot_state):
        _restore_object_fields(robot.controller, state["controller"])
        if "gripper" in state:
            _restore_object_fields(robot.gripper, state["gripper"])
        for name, buffer_state in state["buffers"].items():
            _restore_object_fields(getattr(robot, name), buffer_state)
    for container_name, values in snapshot.object_properties.items():
        container = getattr(core, container_name, {})
        for name, properties in values.items():
            container[name].object_properties = copy.deepcopy(properties)
    for name, fields in snapshot.observable_state.items():
        observable = core._observables[name]
        for field, value in fields.items():
            setattr(observable, field, copy.deepcopy(value))
    core._obs_cache = copy.deepcopy(snapshot.observation_cache)
    np.random.set_state(copy.deepcopy(snapshot.numpy_rng_state))
    random.setstate(snapshot.python_rng_state)
    return copy.deepcopy(snapshot.observation)
