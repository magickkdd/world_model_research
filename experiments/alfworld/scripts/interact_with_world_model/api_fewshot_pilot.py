import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "collect_init_context"))
from collect_wm_instruct_alfworld import get_alfworld_description, get_traj_file_from_id


AGENT_PROMPT = (
    "Interact with a household to solve a task. Imagine you are an intelligent agent "
    "in a household environment and your target is to perform actions to complete the "
    "task goal. At the beginning of your interactions, you will be given the detailed "
    "description of the current environment and your goal to accomplish. For each of "
    "your turns, choose one action from the available actions and output it in this "
    "format:\nThought:\nyour thoughts.\n\nAction:\nyour next action\nAfter each turn, "
    "the environment will give you immediate feedback. If it says Nothing happened, "
    "the action was invalid. Do not use actions outside the available list."
)
AGENT_ACK = "OK. I'll follow your instructions and try my best to solve the task."
WM_PROMPT = (
    "You are an ALFWorld transition model. Each user message gives environment facts, "
    "an observed state, and a valid action. Return only the next textual observation "
    "that the environment would produce after that action. Do not add thoughts, "
    "explanations, or a proposed action. If the task is completed, append [SUCCESS]."
)
ACTION_PRIORITY = (
    "go to", "take", "open", "close", "put", "toggle", "clean", "heat", "cool", "use", "examine"
)


class AlfWorldClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        response = requests.post(f"{self.base_url}/create", timeout=30)
        response.raise_for_status()
        self.env_id = response.json()["id"]

    def reset(self, game):
        response = requests.post(
            f"{self.base_url}/reset",
            json={"id": self.env_id, "game": game, "world_type": "Text"},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def step(self, action):
        response = requests.post(
            f"{self.base_url}/step",
            json={"id": self.env_id, "action": action},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        requests.post(f"{self.base_url}/close", json={"id": self.env_id}, timeout=30)


def action_group(action):
    for group in ACTION_PRIORITY:
        if action == group or action.startswith(group + " "):
            return group
    return action.split(" ", 1)[0]


def choose_action(actions, seen_groups):
    for group in ACTION_PRIORITY:
        if group in seen_groups:
            continue
        action = next((item for item in actions if action_group(item) == group), None)
        if action:
            return action
    action = next((item for item in actions if action_group(item) == "go to"), None)
    if action:
        return action
    return actions[0]


def choose_tasks(rows, count):
    selected = []
    seen = set()
    for row in rows:
        family = row["task_type"].split("-", 1)[0]
        if family in seen:
            continue
        selected.append(row)
        seen.add(family)
        if len(selected) == count:
            break
    return selected


def choose_balanced_tasks(rows, count):
    groups = {}
    for row in rows:
        family = row["task_type"].split("-", 1)[0]
        groups.setdefault(family, []).append(row)
    selected = []
    for position in range(max(map(len, groups.values()), default=0)):
        for group in groups.values():
            if position < len(group):
                selected.append(group[position])
                if len(selected) == count:
                    return selected
    return selected


def observed_text(result):
    actions = ",".join(result["available_actions"])
    return f"{result['observation']}\nAVAILABLE ACTIONS: {actions}"


def environment_facts(item_id, split):
    traj_file = get_traj_file_from_id(item_id, split)
    return get_alfworld_description(traj_file)


def collect_demonstrations(env, train_rows, count):
    examples = []
    seen_groups = set()
    for row in choose_tasks(train_rows, min(len(train_rows), 12)):
        item_id = row["item_id"]
        facts = environment_facts(item_id, "train")
        state = env.reset(item_id)
        for _ in range(12):
            action = choose_action(state["available_actions"], seen_groups)
            group = action_group(action)
            result = env.step(action)
            if group in ACTION_PRIORITY and group not in seen_groups:
                examples.append(
                    {
                        "item_id": item_id,
                        "task_type": row["task_type"],
                        "facts": facts,
                        "state": observed_text(state),
                        "action": action,
                        "next_state": result["observation"],
                    }
                )
                seen_groups.add(group)
                if len(examples) >= count:
                    return examples
            if result.get("done"):
                break
            state = {
                "observation": result["observation"],
                "available_actions": result["available_actions"],
            }
    return examples


def prompt_input(facts, state, action):
    return f"Environment facts:\n{facts}\n\nObserved state:\n{state}\n\nAction: {action}"


def predict(client, model, demos, facts, state, action):
    messages = [{"role": "system", "content": WM_PROMPT}]
    for example in demos:
        messages.append(
            {
                "role": "user",
                "content": prompt_input(example["facts"], example["state"], example["action"]),
            }
        )
        messages.append({"role": "assistant", "content": example["next_state"]})
    messages.append({"role": "user", "content": prompt_input(facts, state, action)})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://127.0.0.1:37601")
    parser.add_argument("--test-tasks", type=int, default=5)
    parser.add_argument("--fewshot", type=int, default=3)
    parser.add_argument(
        "--output",
        default="outputs/api_fewshot_pilot/alfworld_gpt-6-luna.json",
    )
    args = parser.parse_args()

    api_client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://spacetimeai.cc/v1"),
    )
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    train_rows = json.loads(
        (REPO_ROOT / "AgentGym/agentenv-alfworld/configs/mappings_train.json").read_text()
    )
    test_rows = json.loads(
        (REPO_ROOT / "AgentGym/agentenv-alfworld/configs/mappings_test.json").read_text()
    )
    test_rows = choose_balanced_tasks(test_rows, args.test_tasks)
    env = AlfWorldClient(args.env_url)
    try:
        demonstrations = collect_demonstrations(env, train_rows, args.fewshot)
        predictions = []
        for row in test_rows:
            item_id = row["item_id"]
            state_data = env.reset(item_id)
            state = observed_text(state_data)
            facts = environment_facts(item_id, "valid_train")
            action = choose_action(state_data["available_actions"], set())
            gold = env.step(action)["observation"]
            zero_shot = predict(api_client, model, [], facts, state, action)
            few_shot = predict(api_client, model, demonstrations, facts, state, action)
            predictions.append(
                {
                    "item_id": item_id,
                    "task_type": row["task_type"],
                    "action": action,
                    "ground_truth": gold,
                    "zero_shot": zero_shot,
                    "few_shot": few_shot,
                    "zero_shot_exact": zero_shot == gold,
                    "few_shot_exact": few_shot == gold,
                    "zero_shot_normalized_exact": normalized(zero_shot) == normalized(gold),
                    "few_shot_normalized_exact": normalized(few_shot) == normalized(gold),
                }
            )
    finally:
        env.close()

    output = {
        "model": model,
        "base_url": os.getenv("OPENAI_BASE_URL", "https://spacetimeai.cc/v1"),
        "fewshot": len(demonstrations),
        "demonstrations": demonstrations,
        "metrics": {
            "n": len(predictions),
            "zero_shot_exact_accuracy": sum(x["zero_shot_exact"] for x in predictions)
            / len(predictions),
            "few_shot_exact_accuracy": sum(x["few_shot_exact"] for x in predictions)
            / len(predictions),
            "zero_shot_normalized_exact_accuracy": sum(
                x["zero_shot_normalized_exact"] for x in predictions
            )
            / len(predictions),
            "few_shot_normalized_exact_accuracy": sum(
                x["few_shot_normalized_exact"] for x in predictions
            )
            / len(predictions),
        },
        "predictions": predictions,
    }
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))
    print(f"results={output_path}")


if __name__ == "__main__":
    main()
