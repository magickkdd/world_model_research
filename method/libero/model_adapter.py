"""OpenAI-compatible multimodal adapter for structured LIBERO decisions."""

import base64
import json
import os

import cv2
import numpy as np
import requests

from decision_runner import _validate_outcome
from manipulation_skills import SKILL_SCHEMAS
from visual_detections import validate_detections


_OUTCOME_SCHEMA_TEXT = {
    "candidate_id": "string",
    "summary": "string",
    "effects": ["short factual effect"],
    "task_progress": "string",
    "task_success": "boolean",
    "failure_reason": "string or null",
}


def _json_content(content):
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    if not isinstance(content, str):
        raise ValueError("model response content is not text")
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


def _state_summary(observation):
    summary = {}
    for key, value in observation.items():
        if key.endswith(("_image", "_depth")):
            continue
        if isinstance(value, np.ndarray):
            summary[key] = value.tolist()
        elif isinstance(value, np.generic):
            summary[key] = value.item()
        else:
            summary[key] = value
    return summary


class OpenAICompatibleLiberoModel:
    """Implements Agent, language-WM, and selector calls over chat completions."""

    def __init__(self, api_key=None, base_url=None, model=None, timeout_s=120):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is unavailable in this process")
        self.base_url = (base_url or os.getenv(
            "OPENAI_BASE_URL", "https://spacetimeai.cc/v1"
        )).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _image_content(self, observation):
        image = np.asarray(observation["agentview_image"])
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("agentview_image must be HxWx3")
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(
            ".jpg", encoded_image, [int(cv2.IMWRITE_JPEG_QUALITY), 65]
        )
        if not success:
            raise RuntimeError("could not encode agentview image")
        data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": data_url, "detail": "low"},
        }

    def _request_json(self, system_prompt, user_prompt, observation, attach_image=True):
        user_content = [{"type": "text", "text": user_prompt}]
        if attach_image and observation is not None and "agentview_image" in observation:
            user_content.append(self._image_content(observation))
        response = self.session.post(
            self.base_url + "/chat/completions",
            headers={"Authorization": "Bearer " + self.api_key},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=self.timeout_s,
        )
        self.call_count += 1
        if not response.ok:
            raise RuntimeError(
                "chat completion failed with HTTP {}".format(response.status_code)
            )
        payload = response.json()
        usage = payload.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("chat completion returned no message content")
        return _json_content(content)

    def usage(self):
        return {
            "calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def propose(self, goal, observation, history):
        system = (
            "You control a Panda robot through the registered pick and place skills. "
            "A completed pick sequence means only that the arm reached its waypoints and closed the gripper; "
            "it does not prove the object was grasped. A completed place sequence does not prove the object "
            "was released onto the target. After either skill, inspect the new image and history before "
            "choosing the next skill. Propose place only when there is visual evidence the object is held; "
            "if the object remains on its support, re-localize and retry pick. Verify placement from the image "
            "and task feedback. "
            "Use the image to identify only task-relevant entities and pixel bounding boxes. "
            "Propose every distinct useful registered skill candidate; do not cap the "
            "list at an arbitrary count. Never output coordinates outside image boxes "
            "or raw robot controls. Return JSON only."
        )
        user = json.dumps(
            {
                "task_goal": goal,
                "robot_state": _state_summary(observation),
                "recent_history": history,
                "skill_schemas": SKILL_SCHEMAS,
                "required_output": {
                    "detections": [
                        {
                            "entity_id": "stable identifier for this visible entity",
                            "label": "short semantic label",
                            "bbox_xyxy": [0, 0, 10, 10],
                            "confidence": 0.0,
                        }
                    ],
                    "candidates": [
                        {
                            "candidate_id": "unique id",
                            "skill": "pick or place",
                            "args": {},
                        }
                    ],
                    "baseline_selection": {
                        "candidate_id": "one candidate ID",
                        "rationale": "brief decision explanation",
                    },
                },
            },
            ensure_ascii=True,
        )
        payload = self._request_json(system, user, observation)
        if not isinstance(payload, dict) or set(payload) != {
            "detections",
            "candidates",
            "baseline_selection",
        }:
            raise ValueError("proposal response must contain detections, candidates, and baseline_selection")
        image = observation["agentview_image"]
        detections = validate_detections(
            {"detections": payload["detections"]},
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
        candidates = payload["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("proposal response contains no candidates")
        candidate_ids = set()
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != {
                "candidate_id",
                "skill",
                "args",
            }:
                raise ValueError("candidate does not match the skill-call schema")
            candidate_id = candidate["candidate_id"]
            skill = candidate["skill"]
            args = candidate["args"]
            if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_ids:
                raise ValueError("candidate IDs must be non-empty and unique")
            candidate_ids.add(candidate_id)
            if skill not in SKILL_SCHEMAS or not isinstance(args, dict):
                raise ValueError("candidate references an unknown skill or invalid args")
            required = set(SKILL_SCHEMAS[skill]["required"])
            if set(args) != required or not all(
                isinstance(value, str) and value in detections for value in args.values()
            ):
                raise ValueError("skill arguments must reference detected entity IDs")
        baseline_selection = payload["baseline_selection"]
        if (
            not isinstance(baseline_selection, dict)
            or set(baseline_selection) != {"candidate_id", "rationale"}
            or baseline_selection["candidate_id"] not in candidate_ids
            or not isinstance(baseline_selection["rationale"], str)
        ):
            raise ValueError("baseline_selection must reference one proposed candidate")
        return {
            "detections": detections,
            "candidates": candidates,
            "baseline_selection": baseline_selection,
        }

    def predict(self, goal, observation, history, candidate):
        system = (
            "Predict the likely immediate result of one registered robot skill from "
            "the current structured visual detections, robot state, task goal, and recent history. Report a "
            "concise observable prediction, not hidden chain-of-thought. Return JSON only."
        )
        user = json.dumps(
            {
                "task_goal": goal,
                "robot_state": _state_summary(observation),
                "recent_history": history,
                "candidate": candidate,
                "required_output_schema": _OUTCOME_SCHEMA_TEXT,
            },
            ensure_ascii=True,
        )
        outcome = self._request_json(system, user, observation, attach_image=False)
        return _validate_outcome(candidate["candidate_id"], outcome)

    def select(self, arm, goal, observation, history, candidates, outcomes):
        system = (
            "Select one candidate skill for the robot. Consider the task goal, current "
            "structured visual detections and robot state, recent history, and supplied candidate outcomes. "
            "Return a concise decision rationale suitable for an experiment log; do not "
            "provide hidden chain-of-thought. Return JSON only."
        )
        user = json.dumps(
            {
                "arm": arm,
                "task_goal": goal,
                "robot_state": _state_summary(observation),
                "recent_history": history,
                "candidates": candidates,
                "candidate_outcomes": outcomes,
                "required_output": {
                    "candidate_id": "one ID from candidates",
                    "rationale": "brief decision explanation",
                },
            },
            ensure_ascii=True,
        )
        selection = self._request_json(system, user, observation, attach_image=False)
        if (
            not isinstance(selection, dict)
            or set(selection) != {"candidate_id", "rationale"}
            or not isinstance(selection["rationale"], str)
        ):
            raise ValueError("selection response must contain candidate_id and rationale")
        return selection
