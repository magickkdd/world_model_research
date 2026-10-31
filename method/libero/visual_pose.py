"""RGB-D mask localization using an explicit camera-to-robot calibration."""

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PositionEstimate:
    position_base: np.ndarray
    covariance_base: np.ndarray
    visible_surface_spread: np.ndarray
    pixel_center: np.ndarray
    median_depth_m: float
    valid_depth_pixels: int
    source_camera: str
    latency_ms: float


def segment_bbox(rgb_image, bbox_xyxy, iterations=5):
    """Refine a semantic detector's 2D box into a foreground mask with GrabCut."""
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb_image must be an HxWx3 uint8 image")
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    height, width = image.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        raise ValueError("bbox must cover at least 4x4 pixels")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    labels = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    labels[y1:y2, x1:x2] = cv2.GC_PR_BGD
    inset_x = max(1, int(round((x2 - x1) * 0.18)))
    inset_y = max(1, int(round((y2 - y1) * 0.18)))
    inner_x1, inner_x2 = x1 + inset_x, x2 - inset_x
    inner_y1, inner_y2 = y1 + inset_y, y2 - inset_y
    labels[inner_y1:inner_y2, inner_x1:inner_x2] = cv2.GC_PR_FGD
    seed_x = max(1, int(round((x2 - x1) * 0.08)))
    seed_y = max(1, int(round((y2 - y1) * 0.08)))
    labels[
        y1 + (y2 - y1) // 2 - seed_y : y1 + (y2 - y1) // 2 + seed_y,
        x1 + (x2 - x1) // 2 - seed_x : x1 + (x2 - x1) // 2 + seed_x,
    ] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image,
        labels,
        (x1, y1, x2 - x1, y2 - y1),
        background_model,
        foreground_model,
        iterations,
        cv2.GC_INIT_WITH_MASK,
    )
    mask = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    if not np.any(mask):
        raise ValueError("segmentation returned an empty mask")
    return mask


def linearize_mujoco_depth(depth_buffer, extent, znear, zfar):
    """Convert MuJoCo's normalized depth buffer to metric camera-axis depth."""
    depth = np.asarray(depth_buffer, dtype=np.float64)
    if np.any(~np.isfinite(depth)) or np.any((depth < 0.0) | (depth > 1.0)):
        raise ValueError("MuJoCo depth buffer must be finite and in [0, 1]")
    near = float(extent) * float(znear)
    far = float(extent) * float(zfar)
    if near <= 0.0 or far <= near:
        raise ValueError("invalid near/far clipping planes")
    return near / (1.0 - depth * (1.0 - near / far))


def estimate_mask_position(
    mask,
    depth_m,
    intrinsics,
    base_from_camera,
    source_camera="agentview",
    pixel_sigma=1.5,
    depth_sigma_m=0.01,
    min_valid_pixels=12,
    max_depth_m=3.0,
):
    """Estimate a masked visible-surface point and its sensor-error covariance.

    The mask must come from an image detector or segmenter. This function does
    not access simulator object state or segmentation buffers.
    """
    started = time.perf_counter()
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    T_base_camera = np.asarray(base_from_camera, dtype=np.float64)
    if mask.ndim != 2 or depth.shape != mask.shape:
        raise ValueError("mask and depth_m must be same-size 2D arrays")
    if K.shape != (3, 3) or T_base_camera.shape != (4, 4):
        raise ValueError("expected 3x3 intrinsics and 4x4 base_from_camera")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")
    if not np.allclose(T_base_camera[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("base_from_camera must be a homogeneous transform")

    rows, cols = np.nonzero(mask)
    values = depth[rows, cols]
    valid = np.isfinite(values) & (values > 0.0) & (values <= max_depth_m)
    rows, cols, values = rows[valid], cols[valid], values[valid]
    if len(values) < min_valid_pixels:
        raise ValueError(
            "mask has {} valid depth pixels; need at least {}".format(
                len(values), min_valid_pixels
            )
        )

    u = float(np.median(cols))
    v = float(np.median(rows))
    z = float(np.median(values))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    point_camera = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
    point_h = np.append(point_camera, 1.0)
    point_base = (T_base_camera @ point_h)[:3]

    # Propagate detector and depth sensor uncertainty at the mask center.
    jacobian = np.array(
        [
            [z / fx, 0.0, (u - cx) / fx],
            [0.0, z / fy, (v - cy) / fy],
            [0.0, 0.0, 1.0],
        ]
    )
    sensor_covariance = np.diag(
        [float(pixel_sigma) ** 2, float(pixel_sigma) ** 2, float(depth_sigma_m) ** 2]
    )
    rotation = T_base_camera[:3, :3]
    covariance_base = (
        rotation @ jacobian @ sensor_covariance @ jacobian.T @ rotation.T
    )

    cloud_camera = np.column_stack(
        ((cols - cx) * values / fx, (rows - cy) * values / fy, values)
    )
    cloud_base = cloud_camera @ rotation.T + T_base_camera[:3, 3]
    spread = 1.4826 * np.median(np.abs(cloud_base - point_base), axis=0)

    return PositionEstimate(
        position_base=point_base,
        covariance_base=covariance_base,
        visible_surface_spread=spread,
        pixel_center=np.array([u, v]),
        median_depth_m=z,
        valid_depth_pixels=int(len(values)),
        source_camera=source_camera,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def localize_bbox(
    rgb_image,
    bbox_xyxy,
    depth_m,
    intrinsics,
    base_from_camera,
    source_camera="agentview",
    **estimate_options
):
    """Run box refinement and RGB-D backprojection for one semantic detection."""
    mask = segment_bbox(rgb_image, bbox_xyxy)
    estimate = estimate_mask_position(
        mask,
        depth_m,
        intrinsics,
        base_from_camera,
        source_camera=source_camera,
        **estimate_options
    )
    return estimate, mask
