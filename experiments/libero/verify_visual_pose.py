"""Check RGB-D backprojection with a known synthetic calibration."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "libero"))

from visual_pose import (
    estimate_mask_position,
    linearize_mujoco_depth,
    localize_bbox,
    segment_bbox,
)
from visual_detections import validate_detections


def main():
    mask = np.ones((5, 5), dtype=bool)
    depth_m = np.ones((5, 5), dtype=np.float64)
    intrinsics = np.array(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
    )
    base_from_camera = np.eye(4)
    base_from_camera[:3, 3] = [1.0, 2.0, 3.0]
    estimate = estimate_mask_position(
        mask, depth_m, intrinsics, base_from_camera, min_valid_pixels=12
    )
    np.testing.assert_allclose(estimate.position_base, [1.0, 2.0, 4.0], atol=0.01)
    assert np.all(np.linalg.eigvalsh(estimate.covariance_base) >= -1e-12)

    metric = linearize_mujoco_depth(np.array([0.0]), 1.0, 0.1, 10.0)
    np.testing.assert_allclose(metric, [0.1])

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[10:22, 10:22] = [220, 40, 40]
    image_depth = np.ones((32, 32), dtype=np.float64)
    image_intrinsics = np.array(
        [[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]
    )
    mask = segment_bbox(image, [7, 7, 25, 25], iterations=2)
    assert mask[15, 15] and not mask[4, 4]
    estimate, localized_mask = localize_bbox(
        image,
        [7, 7, 25, 25],
        image_depth,
        image_intrinsics,
        base_from_camera,
        min_valid_pixels=12,
    )
    assert localized_mask.any()
    np.testing.assert_allclose(estimate.position_base, [1.0, 2.0, 4.0], atol=0.01)
    detections = validate_detections(
        {
            "detections": [
                {
                    "entity_id": "red_block_1",
                    "label": "red block",
                    "bbox_xyxy": [7, 7, 25, 25],
                    "confidence": 0.9,
                }
            ]
        },
        image_width=32,
        image_height=32,
    )
    assert detections["red_block_1"]["bbox_xyxy"] == [7.0, 7.0, 25.0, 25.0]
    print("RGB-D geometry synthetic check passed")


if __name__ == "__main__":
    main()
