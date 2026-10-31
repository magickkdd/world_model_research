"""Schema and validation for Agent-provided semantic image detections."""

import math


DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "minLength": 1},
                    "label": {"type": "string", "minLength": 1},
                    "bbox_xyxy": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["entity_id", "label", "bbox_xyxy", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["detections"],
    "additionalProperties": False,
}


def validate_detections(payload, image_width, image_height):
    """Validate model detections and return an ID-indexed mapping."""
    if (
        not isinstance(payload, dict)
        or set(payload) != {"detections"}
        or not isinstance(payload["detections"], list)
    ):
        raise ValueError("detection response must contain only a detections array")
    result = {}
    for detection in payload["detections"]:
        required = {"entity_id", "label", "bbox_xyxy", "confidence"}
        if not isinstance(detection, dict) or set(detection) != required:
            raise ValueError("each detection must match DETECTION_SCHEMA")
        entity_id = detection["entity_id"]
        label = detection["label"]
        bbox = detection["bbox_xyxy"]
        confidence = detection["confidence"]
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError("entity_id must be a non-empty string")
        if entity_id in result:
            raise ValueError("duplicate entity_id: {}".format(entity_id))
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a non-empty string")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox_xyxy must contain four pixel coordinates")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in bbox
        ):
            raise ValueError("bbox_xyxy values must be finite numbers")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
            raise ValueError("bbox_xyxy is outside the image or has zero area")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(confidence)
        ):
            raise ValueError("confidence must be finite")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        result[entity_id] = {
            "label": label,
            "bbox_xyxy": [float(value) for value in bbox],
            "confidence": float(confidence),
        }
    return result
