"""Print a compact decision and primitive summary from an episode JSON file."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json")
    args = parser.parse_args()
    with open(args.result_json, "r", encoding="utf-8") as source:
        result = json.load(source)

    print(
        "arm={}, success={}, decisions={}, usage={}".format(
            result["arm"],
            result["success"],
            result["decision_count"],
            result["model_usage"],
        )
    )
    for decision in result["decisions"]:
        trace = decision["decision"]
        execution = trace["execution"]
        print("  detections={}".format(decision.get("detections")))
        print(
            "decision {}: candidates={}, selected={}, rationale={}".format(
                decision["decision_index"],
                len(trace["candidates"]),
                trace["selection"].get("candidate_id"),
                trace["selection"].get("rationale"),
            )
        )
        print(
            "  skill={} args={} status={} task_success={}".format(
                execution.get("skill"),
                execution.get("entity_ids"),
                execution.get("status"),
                execution.get("task_success"),
            )
        )
        for primitive in execution.get("primitive_trace", []):
            print(
                "  {}: status={}, reason={}, steps={}, final_error_m={}, target={}, final={}".format(
                    primitive.get("command", "move_to"),
                    primitive.get("status"),
                    primitive.get("termination_reason"),
                    len(primitive.get("trace", [])),
                    primitive.get("final_error_m"),
                    primitive.get("target_position"),
                    primitive.get("final_position"),
                )
            )


if __name__ == "__main__":
    main()
