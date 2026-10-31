"""Validate model adapter schemas and image request shape without API access."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "libero"))

from model_adapter import OpenAICompatibleLiberoModel


class MockResponse:
    ok = True
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class MockSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, headers, json, timeout):
        self.requests.append({"url": url, "headers": headers, "body": json})
        return MockResponse(
            {
                "choices": [{"message": {"content": self.responses.pop(0)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )


def main():
    adapter = OpenAICompatibleLiberoModel(api_key="mock-only")
    session = MockSession(
        [
            '{"detections":[{"entity_id":"mug_1","label":"mug",'
            '"bbox_xyxy":[4,5,20,24],"confidence":0.9}],"candidates":['
            '{"candidate_id":"pick_mug","skill":"pick",'
            '"args":{"object_id":"mug_1"}}],"baseline_selection":'
            '{"candidate_id":"pick_mug","rationale":"the mug is the task target"}}',
            '{"candidate_id":"pick_mug","summary":"gripper reaches mug",'
            '"effects":["mug contact expected"],"task_progress":"picking",'
            '"task_success":false,"failure_reason":null}',
            '{"candidate_id":"pick_mug","rationale":"the mug is the target"}',
        ]
    )
    adapter.session = session
    observation = {
        "agentview_image": np.zeros((32, 32, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
    }
    proposal = adapter.propose("pick up the mug", observation, history=[])
    candidate = proposal["candidates"][0]
    prediction = adapter.predict("pick up the mug", observation, [], candidate)
    selection = adapter.select(
        "language_wm", "pick up the mug", observation, [], [candidate], [prediction]
    )
    assert selection["candidate_id"] == candidate["candidate_id"]
    assert len(session.requests) == 3
    assert "image_url" in session.requests[0]["body"]["messages"][1]["content"][1]
    assert all(
        len(request["body"]["messages"][1]["content"]) == 1
        for request in session.requests[1:]
    )
    assert adapter.usage() == {
        "calls": 3,
        "prompt_tokens": 30,
        "completion_tokens": 15,
    }
    print("mock model adapter schema and multimodal request check passed")


if __name__ == "__main__":
    main()
