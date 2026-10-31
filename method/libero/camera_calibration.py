"""Camera and robot-base calibration helpers for the fixed LIBERO setup."""

from dataclasses import dataclass

import numpy as np
import robosuite.macros as robosuite_macros
import robosuite.utils.transform_utils as transforms
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)

from visual_pose import linearize_mujoco_depth, localize_bbox


# Keep image pixel coordinates consistent with OpenCV intrinsics and model boxes.
robosuite_macros.IMAGE_CONVENTION = "opencv"


@dataclass
class CameraCalibration:
    camera_name: str
    height: int
    width: int
    intrinsics: np.ndarray
    base_from_camera: np.ndarray
    world_from_base: np.ndarray


def calibration_from_robosuite(env, camera_name="agentview", height=128, width=128):
    """Read fixed camera and robot-mount transforms as calibration metadata.

    The returned transform uses only camera extrinsics and the robot base
    mounting pose. It never reads task-object positions or segmentation.
    For hardware, replace this function with measured calibration values.
    """
    core = getattr(env, "env", env)
    world_from_camera = get_camera_extrinsic_matrix(core.sim, camera_name)
    robot = core.robots[0]
    world_from_base = transforms.pose2mat((robot.base_pos, robot.base_ori))
    base_from_camera = transforms.pose_inv(world_from_base) @ world_from_camera
    intrinsics = get_camera_intrinsic_matrix(
        core.sim, camera_name, height, width
    )
    return CameraCalibration(
        camera_name=camera_name,
        height=height,
        width=width,
        intrinsics=intrinsics,
        base_from_camera=base_from_camera,
        world_from_base=world_from_base,
    )


def localize_observation_bbox(env, observation, bbox_xyxy, calibration):
    """Localize a semantic box from an RGB-D observation and calibration."""
    core = getattr(env, "env", env)
    rgb = observation[calibration.camera_name + "_image"]
    depth_buffer = np.asarray(
        observation[calibration.camera_name + "_depth"], dtype=np.float64
    ).squeeze()
    if rgb.shape[:2] != (calibration.height, calibration.width):
        raise ValueError("RGB image shape does not match camera calibration")
    if depth_buffer.shape != (calibration.height, calibration.width):
        raise ValueError("depth image shape does not match camera calibration")
    metric_depth = linearize_mujoco_depth(
        depth_buffer,
        core.sim.model.stat.extent,
        core.sim.model.vis.map.znear,
        core.sim.model.vis.map.zfar,
    )
    return localize_bbox(
        rgb,
        bbox_xyxy,
        metric_depth,
        calibration.intrinsics,
        calibration.base_from_camera,
        source_camera=calibration.camera_name,
    )
