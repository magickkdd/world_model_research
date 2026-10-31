import json
import os
from pathlib import Path

from openai import OpenAI

from api_fewshot_pilot import REPO_ROOT, WM_PROMPT, environment_facts, prompt_input


def build_messages(demonstrations, facts, state, action, history=None):
    messages = [{"role": "system", "content": WM_PROMPT}]
    for example in demonstrations:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": prompt_input(
                        example["facts"], example["state"], example["action"]
                    ),
                },
                {"role": "assistant", "content": example["next_state"]},
            ]
        )
    current_input = prompt_input(facts, state, action)
    if history:
        current_input += "\n\nRecent real action history:\n" + "\n".join(history)
    messages.append({"role": "user", "content": current_input})
    return messages


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    result_path = REPO_ROOT / "outputs/api_wm_assisted/alfworld_task_2422_paired.json"
    result = json.loads(result_path.read_text())["results"][0]["wm_assisted"]
    step = result["steps"][4]
    previous_steps = result["steps"][2:4]
    history = [
        f"Action: {previous['action']}\nReal environment feedback: {previous['real_observation']}"
        for previous in previous_steps
    ]
    demonstrations = json.loads(
        (REPO_ROOT / "outputs/api_fewshot_pilot/alfworld_gpt-6-luna_20.json").read_text()
    )["demonstrations"][:3]
    facts = environment_facts(2422, "valid_train")

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://spacetimeai.cc/v1"),
        timeout=90,
        max_retries=0,
    )
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    conditions = {
        "local_state_only": build_messages(
            demonstrations, facts, step["state"], step["action"]
        ),
        "with_recent_real_history": build_messages(
            demonstrations,
            facts,
            step["state"]
            + "\n\nExplicit current state derived from real feedback:\n"
            + "Location: sinkbasin 1\n"
            + "Inventory: soapbar 3\n"
            + "The soapbar has not been cleaned yet.",
            step["action"],
            history,
        ),
    }
    outputs = {name: [] for name in conditions}
    output_path = REPO_ROOT / "outputs/api_wm_assisted/state_context_repeats_2422.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for name, messages in conditions.items():
        for repeat in range(args.repeats):
            response = client.chat.completions.create(model=model, messages=messages)
            outputs[name].append(
                {
                    "repeat": repeat + 1,
                    "prediction": response.choices[0].message.content.strip(),
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                        "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    },
                }
            )
            output_path.write_text(
                json.dumps(
                    {
                        "model": model,
                        "task_id": 2422,
                        "action": step["action"],
                        "ground_truth": step["real_observation"],
                        "original_prediction": step["predictions"][0],
                        "repeats_per_condition": args.repeats,
                        "conditions": {
                            key: {"messages": conditions[key], "runs": value}
                            for key, value in outputs.items()
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            print(f"condition={name} repeat={repeat + 1} prediction={outputs[name][-1]['prediction']}", flush=True)
    print(f"results={output_path}")


if __name__ == "__main__":
    main()
