import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from api_fewshot_pilot import AlfWorldClient, REPO_ROOT, observed_text
from collect_wm_instruct_alfworld import get_alfworld_description, loc_to_tuple, parse_game_metadata


PLAN_GROUPS = {
    "GotoLocation": "GotoLocation",
    "PickupObject": "PickupObject",
    "PutObject": "PutObject",
    "CleanObject": "CleanObject",
    "HeatObject": "HeatObject",
    "CoolObject": "CoolObject",
    "SliceObject": "SliceObject",
    "ToggleObject": "ToggleObject",
}


def group_for(action):
    return PLAN_GROUPS.get(action)


def surface_label(label):
    parts = label.lower().split()
    if parts and parts[-1].isdigit():
        return f"{''.join(parts[:-1])} {parts[-1]}"
    return "".join(parts)


def matching_action(plan_item, available, facts="", metadata=None):
    step = plan_item.get("discrete_action", {})
    planner = plan_item.get("planner_action", {})
    kind = step.get("action")
    args = step.get("args", [])
    if not args:
        return None
    obj = re.escape(args[0].lower())
    object_id = planner.get("objectId")
    if kind == "CleanObject":
        object_id = planner.get("cleanObjectId", object_id)
    exact_object = surface_label(metadata.humanize_id(object_id)) if metadata and object_id else None
    patterns = {
        "GotoLocation": rf"^go to {obj} \d+$",
        "PickupObject": rf"^take {obj} \d+ from ",
        "PutObject": rf"^(?:move|put) {obj} \d+ (?:to|in|into|on) ",
        "CleanObject": rf"^clean {obj} \d+ with ",
        "HeatObject": rf"^heat {obj} \d+ with ",
        "CoolObject": rf"^cool {obj} \d+ with ",
        "SliceObject": rf"^slice {obj} \d+ with ",
        "ToggleObject": rf"^(?:use|toggle) {obj} \d+$",
    }
    pattern = patterns.get(kind)
    if kind == "GotoLocation" and metadata and planner.get("location"):
        names = metadata.location_to_receptacle.get(loc_to_tuple(planner["location"]), [])
        target = surface_label(args[0])
        exact = next(
            (metadata.humanize_id(name) for name in names if surface_label(metadata.humanize_id(name)).split()[0] == target),
            None,
        )
        location = surface_label(exact or metadata.humanize_location(planner["location"], args[0]))
        action = next((a for a in available if a.lower() == f"go to {location}"), None)
        if action:
            return action
    if kind == "PickupObject" and exact_object:
        pattern = rf"^take {re.escape(exact_object)} from "
    if kind == "PutObject" and len(args) > 1:
        receptacle_id = planner.get("receptacleObjectId")
        receptacle = surface_label(metadata.humanize_id(receptacle_id)) if metadata and receptacle_id else surface_label(args[1])
        exact_object = metadata.humanize_id(object_id).lower() if metadata and object_id else None
        object_pattern = re.escape(exact_object) if exact_object else rf"{obj} \d+"
        pattern = rf"^(?:move|put) {object_pattern} (?:to|in|into|on) {re.escape(receptacle)}$"
    found = next((action for action in available if pattern and re.search(pattern, action.lower())), None)
    if found or kind != "GotoLocation":
        return found
    target = re.escape(args[0].lower())
    for line in facts.splitlines():
        if not re.search(rf"\b{target}\b", line.lower()):
            continue
        location = re.search(r"On the (.+?), you see", line, re.I)
        if location:
            name = re.escape(surface_label(location.group(1)))
            return next((action for action in available if re.fullmatch(rf"go to {name}", action.lower())), None)
    return None


def append_example(examples, counts, family_counts, row, facts, state, action, feedback, group, per_group):
    family = row["task_type"].split("-", 1)[0]
    family_cap = per_group if group in {"CleanObject", "HeatObject", "CoolObject", "SliceObject", "ToggleObject", "OpenObject", "CloseObject", "ExamineObject"} else 2
    if counts[group] >= per_group or family_counts[(group, family)] >= family_cap:
        return
    examples.append({
        "item_id": row["item_id"],
        "task_type": row["task_type"],
        "transition_type": group,
        "state": observed_text(state),
        "action": action,
        "next_state": feedback,
    })
    counts[group] += 1
    family_counts[(group, family)] += 1


def collect_plan_examples(env, rows, max_per_group):
    root = Path.home() / ".cache/alfworld/json_2.1.1/train"
    examples, counts, family_counts = [], Counter(), Counter()
    row_by_family = defaultdict(list)
    for row in rows:
        row_by_family[row["task_type"].split("-", 1)[0]].append(row)
    families = list(row_by_family)
    for family in families:
        sampled_family_tasks = 0
        for row in row_by_family[family]:
            if all(counts[group] >= max_per_group for group in PLAN_GROUPS.values()):
                break
            trajectory = root / row["task_type"] / row["task_id"] / "traj_data.json"
            if not trajectory.exists() or not trajectory.with_name("game.tw-pddl").exists():
                continue
            data = json.loads(trajectory.read_text())
            planned = data.get("plan", {}).get("high_pddl", [])
            needed = {
                group_for(item.get("discrete_action", {}).get("action"))
                for item in planned
                if group_for(item.get("discrete_action", {}).get("action"))
                and counts[group_for(item["discrete_action"]["action"])] < max_per_group
                and family_counts[(group_for(item["discrete_action"]["action"]), family)]
                < (max_per_group if group_for(item["discrete_action"]["action"]) in {"CleanObject", "HeatObject", "CoolObject", "SliceObject", "ToggleObject"} else 2)
            }
            if not needed:
                continue
            if sampled_family_tasks >= max(12, max_per_group):
                break
            sampled_family_tasks += 1
            try:
                state = env.reset(row["item_id"])
            except RuntimeError:
                continue
            facts = get_alfworld_description(trajectory)
            metadata = parse_game_metadata(trajectory.with_name("game.tw-pddl"))
            for item in planned:
                discrete = item.get("discrete_action", {})
                kind = discrete.get("action")
                group = group_for(kind)
                action = matching_action(item, state["available_actions"], facts, metadata)
                if not action:
                    break
                result = env.step(action)
                if group and group in needed:
                    append_example(examples, counts, family_counts, row, facts, state, action, result["observation"], group, max_per_group)
                state = {"observation": result["observation"], "available_actions": result.get("available_actions", [])}
                if result.get("done"):
                    break
        print(f"sampled family {family} ({sampled_family_tasks} episodes); transition counts={dict(counts)}", flush=True)
    return examples, counts


def collect_open_close_examples(env, rows, examples, counts, target):
    family_counts = Counter()
    attempts = 0
    for row in rows:
        if counts["OpenObject"] >= target and counts["CloseObject"] >= target:
            break
        trajectory = Path.home() / ".cache/alfworld/json_2.1.1/train" / row["task_type"] / row["task_id"] / "traj_data.json"
        if not trajectory.exists() or not trajectory.with_name("game.tw-pddl").exists():
            continue
        attempts += 1
        if attempts > 80:
            break
        facts = get_alfworld_description(trajectory)
        try:
            state = env.reset(row["item_id"])
        except RuntimeError:
            continue
        nav = next((a for a in state["available_actions"] if re.match(r"^go to (?:cabinet|drawer|safe) \d+$", a)), None)
        if not nav:
            continue
        state_before = state
        result = env.step(nav)
        state = {"observation": result["observation"], "available_actions": result.get("available_actions", [])}
        examine_action = next((a for a in state["available_actions"] if a.startswith("examine ")), None)
        if examine_action and counts["ExamineObject"] < target:
            before = state
            result = env.step(examine_action)
            append_example(examples, counts, family_counts, row, facts, before, examine_action, result["observation"], "ExamineObject", target)
            state = {"observation": result["observation"], "available_actions": result.get("available_actions", [])}
        open_action = next((a for a in state["available_actions"] if a.startswith("open ")), None)
        if open_action and counts["OpenObject"] < target:
            before = state
            result = env.step(open_action)
            append_example(examples, counts, family_counts, row, facts, before, open_action, result["observation"], "OpenObject", target)
            state = {"observation": result["observation"], "available_actions": result.get("available_actions", [])}
        close_action = next((a for a in state["available_actions"] if a.startswith("close ")), None)
        if close_action and counts["CloseObject"] < target:
            before = state
            result = env.step(close_action)
            append_example(examples, counts, family_counts, row, facts, before, close_action, result["observation"], "CloseObject", target)
            state = {"observation": result["observation"], "available_actions": result.get("available_actions", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://127.0.0.1:36001")
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--output", default="outputs/api_wm_assisted/alfworld_transition_demos_compact.json")
    args = parser.parse_args()
    rows = json.loads((REPO_ROOT / "AgentGym/agentenv-alfworld/configs/mappings_train.json").read_text())
    env = AlfWorldClient(args.env_url)
    try:
        examples, counts = collect_plan_examples(env, rows, args.per_group)
        collect_open_close_examples(env, rows, examples, counts, args.per_group)
    finally:
        env.close()
    output = {
        "source": "ALFWorld training trajectories executed in the text environment",
        "examples_per_transition_type": args.per_group,
        "transition_counts": dict(counts),
        "demonstrations": examples,
    }
    path = REPO_ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"count": len(examples), "transition_counts": dict(counts)}, ensure_ascii=False, indent=2))
    print(f"demonstrations={path}")


if __name__ == "__main__":
    main()
