import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from api_fewshot_pilot import (
    AGENT_ACK,
    AGENT_PROMPT,
    AlfWorldClient,
    REPO_ROOT,
    WM_PROMPT,
    choose_balanced_tasks,
    environment_facts,
    observed_text,
    prompt_input,
)


def generate(client, model, messages, usage):
    response = client.chat.completions.create(model=model, messages=messages)
    usage["calls"] += 1
    if response.usage:
        usage["prompt_tokens"] += response.usage.prompt_tokens
        usage["completion_tokens"] += response.usage.completion_tokens
    return response.choices[0].message.content.strip()


def extract_action(text):
    match = re.search(r"Action:\s*(.+)", text, re.DOTALL)
    if match is None:
        return ""
    return match.group(1).strip().splitlines()[0].strip()


def initial_agent_history(state):
    return [
        {"role": "user", "content": AGENT_PROMPT},
        {"role": "assistant", "content": AGENT_ACK},
        {"role": "user", "content": observed_text(state)},
    ]


def run_real(env, client, model, item_id, max_steps, usage):
    initial_state = env.reset(item_id)
    history = initial_agent_history(initial_state)
    actions = []
    success = False
    for _ in range(max_steps):
        react = generate(client, model, history, usage)
        history.append({"role": "assistant", "content": react})
        action = extract_action(react)
        actions.append(action)
        result = env.step(action)
        observation = result["observation"]
        history.append({"role": "user", "content": observation})
        if result.get("done"):
            success = float(result.get("reward", 0)) in (1.0, 100.0)
            break
    return {"success": int(success), "actions": actions, "conversations": history}


def run_world_model(env, client, model, item_id, max_steps, demonstrations, usage):
    initial_state = env.reset(item_id)
    facts = environment_facts(item_id, "valid_train")
    agent_history = initial_agent_history(initial_state)
    wm_history = [{"role": "system", "content": WM_PROMPT}]
    for example in demonstrations:
        wm_history.append(
            {
                "role": "user",
                "content": prompt_input(
                    example["facts"], example["state"], example["action"]
                ),
            }
        )
        wm_history.append({"role": "assistant", "content": example["next_state"]})

    current_observation = observed_text(initial_state)
    actions = []
    success = False
    for _ in range(max_steps):
        react = generate(client, model, agent_history, usage)
        agent_history.append({"role": "assistant", "content": react})
        action = extract_action(react)
        actions.append(action)

        wm_history.append(
            {
                "role": "user",
                "content": prompt_input(facts, current_observation, action),
            }
        )
        prediction = generate(client, model, wm_history, usage)
        if "[SUCCESS]" in prediction:
            success = True
            prediction = prediction.replace("[SUCCESS]", "").strip()
        wm_history.append({"role": "assistant", "content": prediction})
        agent_history.append({"role": "user", "content": prediction})
        current_observation = prediction
        if success:
            break
    return {"success": int(success), "actions": actions, "conversations": agent_history}


def replay_actions(env, item_id, actions):
    env.reset(item_id)
    success = False
    observations = []
    for action in actions:
        result = env.step(action)
        observations.append(result["observation"])
        if result.get("done"):
            success = float(result.get("reward", 0)) in (1.0, 100.0)
            break
    return {"success": int(success), "observations": observations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://127.0.0.1:37601")
    parser.add_argument("--test-tasks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--demonstrations",
        default="outputs/api_fewshot_pilot/alfworld_gpt-6-luna_20.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/api_rollout_pilot/alfworld_gpt-6-luna_3x10.json",
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://spacetimeai.cc/v1"),
    )
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    demo_file = REPO_ROOT / args.demonstrations
    demonstrations = json.loads(demo_file.read_text())["demonstrations"]
    mappings = json.loads(
        (REPO_ROOT / "AgentGym/agentenv-alfworld/configs/mappings_test.json").read_text()
    )
    test_rows = choose_balanced_tasks(mappings, args.test_tasks)
    env = AlfWorldClient(args.env_url)
    total_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    results = []
    try:
        for row in test_rows:
            item_id = row["item_id"]
            real_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            wm_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            real = run_real(env, client, model, item_id, args.max_steps, real_usage)
            world_model = run_world_model(
                env,
                client,
                model,
                item_id,
                args.max_steps,
                demonstrations,
                wm_usage,
            )
            wm2real = replay_actions(env, item_id, world_model["actions"])
            for usage in (real_usage, wm_usage):
                for key in total_usage:
                    total_usage[key] += usage[key]
            results.append(
                {
                    "item_id": item_id,
                    "task_type": row["task_type"],
                    "real": real,
                    "world_model": world_model,
                    "wm2real": wm2real,
                    "usage": {"real": real_usage, "world_model": wm_usage},
                }
            )
    finally:
        env.close()

    count = len(results)
    metrics = {
        "n": count,
        "max_steps": args.max_steps,
        "real_success_rate": sum(row["real"]["success"] for row in results) / count,
        "world_model_success_rate": sum(
            row["world_model"]["success"] for row in results
        )
        / count,
        "wm2real_success_rate": sum(row["wm2real"]["success"] for row in results)
        / count,
        **total_usage,
    }
    output = {
        "model": model,
        "base_url": os.getenv("OPENAI_BASE_URL", "https://spacetimeai.cc/v1"),
        "demonstrations_file": str(demo_file),
        "metrics": metrics,
        "results": results,
    }
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"results={output_path}")


if __name__ == "__main__":
    main()
