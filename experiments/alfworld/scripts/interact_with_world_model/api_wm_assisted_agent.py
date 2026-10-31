import argparse
import gzip
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from api_fewshot_pilot import AGENT_ACK, AlfWorldClient, REPO_ROOT, choose_balanced_tasks, observed_text

TASK_AGENT_PROMPT = (
    "Interact with a household to complete the stated task using only available actions. "
    "Keep a concise, explicit progress ledger: completed requirements and remaining requirements. "
    "Update it from real environment feedback after every action. "
    "Never treat an unrelated visible object as a goal or assume objects conceal other objects "
    "unless real feedback confirms it. Do not repeat actions or revisit states without a concrete "
    "expected change. For a task that asks to look at an object under a lamp, keep the target object "
    "in inventory, turn on the target lamp, and go to the lamp; place it under the lamp when the "
    "task explicitly specifies that relationship. Do not move it to unrelated surfaces. "
    "The real environment feedback is authoritative. When the goal requires cleaning, explicitly "
    "clean the target object and wait for successful cleaning feedback before placing it; lack of "
    "a dirty-state description does not mean cleaning is complete. "
    "For each turn, give a short auditable rationale, progress, and action. "
    "Provide only the concise decision basis, not private hidden reasoning."
)

LLM_WM_PROMPT = (
    "You predict one-step feedback from the ALFWorld Text environment. Predict the exact "
    "observation returned after the supplied action is executed once. Ground the prediction "
    "in the current observation, available actions, inventory, and real action history; use "
    "the demonstrations to learn ALFWorld-specific feedback wording and action semantics. "
    "Use any prior prediction checks as corrections: the real feedback is authoritative. "
    "Do not substitute a different action or insert a prerequisite action. Do not infer a "
    "device prerequisite from physical-world convention alone; follow ALFWorld examples and "
    "observed state. "
    "Do not invent hidden objects or claim task success unless the environment feedback says so. "
    "Return only the predicted environment observation, with no rationale or action proposal."
)


TRACE_PATH = None


def generate(client, model, messages, usage, metadata=None):
    usage["calls"] += 1
    try:
        response = client.chat.completions.create(model=model, messages=messages)
    except Exception as exc:
        if TRACE_PATH:
            TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
            failed_trace = {
                "metadata": metadata or {},
                "model": model,
                "request_messages": messages,
                "error": f"{type(exc).__name__}: {exc}",
            }
            with gzip.open(TRACE_PATH, "at", encoding="utf-8") as stream:
                stream.write(json.dumps(failed_trace, ensure_ascii=False) + "\n")
        raise
    if response.usage:
        usage["prompt_tokens"] += response.usage.prompt_tokens
        usage["completion_tokens"] += response.usage.completion_tokens
    message = response.choices[0].message
    response_message = message.model_dump(exclude_none=True)
    trace = {
        "metadata": metadata or {},
        "model": response.model,
        "request_messages": messages,
        "response_message": response_message,
        "finish_reason": response.choices[0].finish_reason,
        "usage": response.usage.model_dump(exclude_none=True) if response.usage else None,
    }
    if TRACE_PATH:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(TRACE_PATH, "at", encoding="utf-8") as stream:
            stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return (message.content or "").strip(), response_message


def parse_candidates(text, available_actions, limit):
    section = text
    header = re.search(r"^Candidates\s*:\s*", text, re.I | re.M)
    if header:
        section = text[header.end():]
    candidates = []
    for line in section.splitlines():
        line = re.sub(r"^\s*(?:[-*]\s*|\d+[.)]\s*)", "", line).strip().strip("`\"'")
        line = re.sub(r"^(?:candidate|action)\s*:\s*", "", line, flags=re.I)
        if line in available_actions and line not in candidates:
            candidates.append(line)
        if len(candidates) >= limit:
            break
    if len(candidates) < limit:
        text_lower = section.lower()
        matches = []
        for action in available_actions:
            pattern = rf"(?<![a-z0-9]){re.escape(action.lower())}(?![a-z0-9])"
            matches.extend((match.start(), action) for match in re.finditer(pattern, text_lower))
        for _position, action in sorted(matches):
            if action not in candidates:
                candidates.append(action)
            if len(candidates) >= limit:
                break
    return candidates


def parse_choice(text, candidates):
    match = re.search(r"(?:choice|select|action)\s*:?\s*(\d+)", text, re.I)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(candidates):
            return candidates[index], False
    rationale = parse_labeled_value(text, "Rationale", "").lower()
    referenced = [candidate for candidate in candidates if candidate.lower() in rationale]
    if len(referenced) == 1:
        return referenced[0], True
    return candidates[0], True


def parse_labeled_value(text, label, default):
    match = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", text, re.I | re.M)
    return match.group(1).strip() if match else default


def transition_group(action):
    lowered = action.lower()
    if lowered.startswith("go to "):
        return "GotoLocation"
    if lowered.startswith("take "):
        return "PickupObject"
    if lowered.startswith(("move ", "put ")):
        return "PutObject"
    for prefix, group in (
        ("clean ", "CleanObject"), ("heat ", "HeatObject"), ("cool ", "CoolObject"),
        ("slice ", "SliceObject"), ("use ", "ToggleObject"), ("toggle ", "ToggleObject"),
    ):
        if lowered.startswith(prefix):
            return group
    if lowered.startswith("open "):
        return "OpenObject"
    if lowered.startswith("close "):
        return "CloseObject"
    if lowered.startswith("examine "):
        return "ExamineObject"
    return None


def demonstrations_for_action(demonstrations, action, limit):
    group = transition_group(action)
    normalized_action = re.sub(r"\s+", " ", action.strip().lower())
    selected, seen_families = [], set()
    for item in demonstrations:
        if item.get("transition_type") != group:
            continue
        if re.sub(r"\s+", " ", item.get("action", "").strip().lower()) == normalized_action:
            continue
        family = item.get("task_type", "").split("-", 1)[0]
        if family in seen_families:
            continue
        selected.append(item)
        seen_families.add(family)
        if len(selected) >= limit:
            break
    return selected


def wm_input(task_goal, state_text, inventory, real_history, calibration_history, action):
    history_text = "\n".join(real_history[-6:]) if real_history else "No actions have been taken yet."
    calibration_text = "\n".join(calibration_history[-4:]) if calibration_history else "No checked predictions yet."
    inventory_text = ", ".join(sorted(inventory)) if inventory else "empty"
    return (
        f"Task goal:\n{task_goal}\n\n"
        f"Current ALFWorld observation and available actions:\n{state_text}\n\n"
        f"Current inventory:\n{inventory_text}\n\n"
        f"Recent real actions and feedback:\n{history_text}\n\n"
        f"Recent prediction checks for executed actions:\n{calibration_text}\n\n"
        f"Execute this available action once:\n{action}\n\n"
        "Predict only the immediate ALFWorld observation after this action."
    )


def normalized_feedback(text):
    value = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[.!?]+$", "", value)


def feedback_token_f1(prediction, ground_truth):
    predicted = re.findall(r"[a-z0-9]+", prediction.lower())
    actual = re.findall(r"[a-z0-9]+", ground_truth.lower())
    if not predicted or not actual:
        return 1.0 if predicted == actual else 0.0
    overlap = sum(min(predicted.count(token), actual.count(token)) for token in set(predicted))
    precision = overlap / len(predicted)
    recall = overlap / len(actual)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def goal_terms(task_goal):
    stop = {
        "a", "an", "the", "to", "in", "into", "on", "under", "at", "with", "from",
        "of", "and", "then", "it", "is", "are", "your", "task", "please", "look",
        "put", "place", "move", "pick", "take", "find", "go", "get", "turn", "make",
        "clean", "heat", "cool", "slice", "examine", "use", "open", "close", "some",
    }
    return [word for word in re.findall(r"[a-z]+", task_goal.lower()) if word not in stop]


def destination_matches_goal(action, task_goal, target_entities):
    destination = re.split(r"\b(?:to|in|into|on)\b", action.lower())[-1]
    compact_destination = re.sub(r"[^a-z0-9]", "", destination)
    names = goal_terms(task_goal) + list(target_entities)
    return any(
        re.sub(r"[^a-z0-9]", "", name.lower()) in compact_destination
        for name in names
        if re.sub(r"[^a-z0-9]", "", name.lower())
    )


def operation_completed(verb, target_entities, real_history):
    marker = {
        "clean": "you clean the",
        "heat": "you heat the",
        "cool": "you cool the",
        "slice": "you slice the",
    }.get(verb, "")
    if not marker:
        return False
    compact = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    entities = [compact(entity) for entity in target_entities]
    return any(
        marker in record.lower() and any(entity and entity in compact(record) for entity in entities)
        for record in real_history
    )


def goal_related_actions(task_goal, available_actions, inventory, real_history=(), target_entities=()):
    terms = goal_terms(task_goal)
    entity_names = list(target_entities)
    if not entity_names:
        entity_names = terms
    compact = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    inventory_compact = {compact(item) for item in inventory}
    goal_lower = task_goal.lower()
    matched = []

    for entity in entity_names:
        entity_key = compact(entity)
        is_device = "lamp" in entity_key or "light" in entity_key
        is_held = any(entity_key in item for item in inventory_compact)
        for action in available_actions:
            action_key = compact(action)
            lowered = action.lower()
            if lowered.startswith("go to ") and entity_key in action_key:
                matched.append(action)
            elif lowered.startswith("take ") and not is_held and entity_key in action_key:
                matched.append(action)
            elif lowered.startswith(("clean ", "heat ", "cool ", "slice ")):
                verb = lowered.split(" ", 1)[0]
                verb_matches_goal = {
                    "clean": "clean" in goal_lower,
                    "heat": "heat" in goal_lower or "hot" in goal_lower,
                    "cool": "cool" in goal_lower or "cold" in goal_lower,
                    "slice": "slice" in goal_lower,
                }
                if verb_matches_goal.get(verb, False) and entity_key in action_key:
                    matched.append(action)
            elif lowered.startswith(("put ", "move ")) and re.search(r"\b(?:put|place|move)\b", goal_lower):
                if entity_key in action_key and not is_device and destination_matches_goal(action, task_goal, target_entities):
                    matched.append(action)
            elif lowered.startswith(("use ", "toggle ")) and is_device and "look at" in goal_lower:
                recent_toggles = re.findall(r"You turn (on|off) the ([a-z0-9 ]+)", "\n".join(real_history), re.I)
                current_on = any(
                    state.lower() == "on" and compact(device) in entity_key
                    for state, device in recent_toggles
                )
                if not current_on:
                    matched.append(action)

    target_in_inventory = any(
        compact(entity) in item for entity in entity_names for item in inventory_compact
    )
    tool_locations = []
    if target_in_inventory and "clean" in goal_lower and not operation_completed("clean", entity_names, real_history):
        tool_locations.append("sinkbasin")
    if target_in_inventory and ("heat" in goal_lower or "hot" in goal_lower) and not operation_completed("heat", entity_names, real_history):
        tool_locations.extend(("microwave", "stoveburner"))
    if target_in_inventory and ("cool" in goal_lower or "cold" in goal_lower) and not operation_completed("cool", entity_names, real_history):
        tool_locations.append("fridge")
    if target_in_inventory and "slice" in goal_lower and not operation_completed("slice", entity_names, real_history):
        tool_locations.append("knife")
    for action in available_actions:
        if action.lower().startswith("go to ") and any(tool in compact(action) for tool in tool_locations):
            if action not in matched:
                matched.append(action)
    return matched


def completed_goal_placements(task_goal, target_entities, real_history):
    compact = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    target_objects = {compact(entity) for entity in target_entities}
    completed = set()
    for record in real_history:
        match = re.search(
            r"Real feedback:\s*You (?:move|put) the (.+?) (?:to|in|into|on) (.+?)(?:\.|$)",
            record,
            re.I,
        )
        if not match:
            continue
        obj, destination = match.groups()
        object_key = compact(obj)
        if any(target and target in object_key for target in target_objects) and destination_matches_goal(
            f"move {obj} to {destination}", task_goal, target_entities
        ):
            completed.add(object_key)
    return completed


def final_destination_entities(task_goal, target_entities):
    goal = task_goal.lower()
    if not re.search(r"\b(?:put|place|move)\b", goal):
        return set()
    destination = re.split(r"\b(?:in|into|on|to)\b", goal)[-1].strip(" .")
    destination = re.sub(r"^(?:the|a|an)\s+", "", destination)
    destination_key = re.sub(r"[^a-z0-9]", "", destination)
    if not destination_key:
        return set()
    compact = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    return {
        compact(entity) for entity in target_entities
        if destination_key in compact(entity) or compact(entity) in destination_key
    }


def required_operations_complete(task_goal, target_entities, real_history):
    goal = task_goal.lower()
    requirements = (
        ("clean", "clean" in goal),
        ("heat", "heat" in goal or "hot" in goal),
        ("cool", "cool" in goal or "cold" in goal),
        ("slice", "slice" in goal),
    )
    return all(not required or operation_completed(verb, target_entities, real_history) for verb, required in requirements)


def action_is_goal_consistent(action, task_goal, target_entities, inventory, real_history):
    lowered = action.lower()
    compact = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    compact_action = compact(lowered)
    compact_entities = [compact(item) for item in target_entities]
    completed_placements = completed_goal_placements(task_goal, target_entities, real_history)
    goal = task_goal.lower()
    if lowered.startswith("go to "):
        destination = compact(lowered[len("go to "):])
        final_destinations = final_destination_entities(task_goal, target_entities)
        object_entities = [entity for entity in compact_entities if entity not in final_destinations]
        compact_inventory = [compact(item) for item in inventory]
        target_in_inventory = any(
            entity and any(entity in item for item in compact_inventory)
            for entity in object_entities
        )
        if any(entity and entity in destination for entity in final_destinations):
            return target_in_inventory and required_operations_complete(task_goal, target_entities, real_history)
        if any(entity and entity in destination for entity in object_entities):
            return True
        needed_tools = (
            (("sinkbasin",), "clean", "clean" in goal),
            (("microwave", "stoveburner"), "heat", "heat" in goal or "hot" in goal),
            (("fridge",), "cool", "cool" in goal or "cold" in goal),
            (("knife",), "slice", "slice" in goal),
        )
        target_in_inventory = any(
            entity and any(entity in item for item in compact_inventory)
            for entity in object_entities
        )
        if any(
            required and target_in_inventory
            and not operation_completed(verb, target_entities, real_history)
            and any(tool in destination for tool in tools)
            for tools, verb, required in needed_tools
        ):
            return True
        if re.search(r"\b(?:under|beneath)\b", goal) and any(
            "lamp" in entity or "light" in entity for entity in object_entities
        ):
            return True
        return not target_in_inventory
    if lowered.startswith(("put ", "move ")):
        match = re.match(r"(?:put|move) (.+?) (?:to|in|into|on) (.+)$", lowered)
        if not match:
            return False
        moved_object = compact(match.group(1))
        destination = compact(match.group(2))
        lamps = [entity for entity in compact_entities if "lamp" in entity or "light" in entity]
        objects = [entity for entity in compact_entities if entity not in lamps]
        moved_target = any(entity and entity in moved_object for entity in objects)
        if re.search(r"\b(?:under|beneath)\b", goal):
            return moved_target
        if re.search(r"\b(?:put|place|move)\b", goal):
            if "clean" in goal and not any(
                "you clean the" in feedback.lower()
                and any(entity in compact(feedback) for entity in objects)
                for feedback in real_history
            ):
                return False
            return (
                moved_target
                and moved_object not in completed_placements
                and moved_object in {compact(item) for item in inventory}
                and destination_matches_goal(action, task_goal, target_entities)
            )
        return False
    if lowered.startswith("take "):
        match = re.match(r"take (.+?) from ", lowered)
        picked_object = compact(match.group(1)) if match else ""
        matching_entities = [
            entity for entity in compact_entities
            if entity and entity in picked_object and picked_object not in completed_placements
        ]
        if matching_entities:
            compact_inventory = [re.sub(r"[^a-z0-9]", "", item.lower()) for item in inventory]
            return not any(entity in item for entity in matching_entities for item in compact_inventory)
        tool_needed = (
            ("knife" in lowered and any(verb in goal for verb in ("slice", "heat", "cool", "clean")))
            or ("sponge" in lowered and "clean" in goal)
        )
        return tool_needed
    if lowered.startswith(("clean ", "heat ", "cool ", "slice ")):
        action_verb = lowered.split(" ", 1)[0]
        action_object = compact(lowered.split(" ", 1)[1].split(" with ", 1)[0])
        verb_required = {
            "clean": "clean" in goal,
            "heat": "heat" in goal or "hot" in goal,
            "cool": "cool" in goal or "cold" in goal,
            "slice": "slice" in goal,
        }
        already_completed = operation_completed(action_verb, target_entities, real_history)
        return (
            verb_required.get(action_verb, False)
            and any(entity and entity in action_object for entity in compact_entities)
            and not already_completed
        )
    if lowered.startswith(("use ", "toggle ")):
        if not any(term in lowered for term in ("lamp", "light")):
            return False
        events = re.findall(r"You turn (on|off) the ([a-z0-9 ]+)", "\n".join(real_history), re.I)
        return not any(state.lower() == "on" for state, _device in events)
    if lowered.startswith(("open ", "close ", "examine ")):
        if any(entity and entity in compact_action for entity in compact_entities):
            return True
        searching = not any(
            entity and any(entity in compact(item) for item in inventory)
            for entity in compact_entities
        )
        return lowered.startswith(("open ", "examine ")) and searching
    return False


def initial_history(state):
    return [
        {"role": "user", "content": TASK_AGENT_PROMPT},
        {"role": "assistant", "content": AGENT_ACK},
        {"role": "user", "content": observed_text(state)},
    ]


def extract_task_goal(observation):
    match = re.search(r"Your task is to:\s*(.+)", observation)
    return match.group(1).strip() if match else "Not explicitly stated"


def update_inventory(inventory, observation):
    pickup = re.search(r"You pick up the (.+?) from\b", observation, re.I)
    if pickup:
        inventory.add(pickup.group(1).strip().lower())
    placement = re.search(r"You (?:move|put) the (.+?) (?:to|in|into|on)\b", observation, re.I)
    if placement:
        inventory.discard(placement.group(1).strip().lower())


def restore_simulation_state(simulator, item_id, real_actions, expected_state):
    state = simulator.reset(item_id)
    for action in real_actions:
        result = simulator.step(action)
        if result.get("done"):
            raise RuntimeError(f"Simulation replay ended before current state after action: {action}")
        state = {"observation": result["observation"], "available_actions": result.get("available_actions", [])}

    normalize = lambda value: re.sub(r"\s+", " ", value).strip()
    if normalize(state["observation"]) != normalize(expected_state["observation"]):
        raise RuntimeError("Simulation restore mismatch: observation differs from real environment.")
    if set(state["available_actions"]) != set(expected_state["available_actions"]):
        raise RuntimeError("Simulation restore mismatch: available actions differ from real environment.")
    return state


def simulate_candidates(simulator, item_id, real_actions, expected_state, candidates):
    simulations = []
    for action in candidates:
        restore_simulation_state(simulator, item_id, real_actions, expected_state)
        result = simulator.step(action)
        simulations.append({
            "action": action,
            "observation": result["observation"],
            "available_actions": result.get("available_actions", []),
            "done": result.get("done", False),
            "reward": result.get("reward", 0),
            "success": result.get("done", False) and float(result.get("reward", 0)) in (1.0, 100.0),
        })
    restore_simulation_state(simulator, item_id, real_actions, expected_state)
    return simulations


def run_episode(env, simulator, client, model, row, max_steps, arm, candidate_count, demonstrations, fewshot):
    item_id = row["item_id"]
    state = env.reset(item_id)
    history = initial_history(state)
    task_goal = extract_task_goal(state["observation"])
    target_entities = goal_terms(task_goal)
    inventory = set()
    real_history = []
    real_actions = []
    calibration_history = []
    progress = "No actions completed; infer requirements from the task and current state."
    state_signatures = []
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    steps = []
    success = False
    assisted = arm != "baseline"

    for step_index in range(max_steps):
        state_text = observed_text(state)
        all_agent_candidates = candidate_count <= 0
        step_candidate_limit = (
            len(state["available_actions"])
            if all_agent_candidates
            else min(candidate_count, len(state["available_actions"]))
        )
        signature = re.sub(r"\s+", " ", state_text).strip().lower()
        repeated_state = state_signatures.count(signature) >= 1
        loop_warning = (
            "LOOP WARNING: this exact environment state has occurred before without task completion. "
            "Choose an action that changes a stated unmet requirement; do not repeat inspection or travel."
            if repeated_state else "No repeated-state warning."
        )
        context = (
            f"Task goal: {task_goal}\nProgress ledger: {progress}\n"
            f"Current real state and available actions:\n{state_text}\n"
            f"Tracked inventory: {', '.join(sorted(inventory)) if inventory else 'empty'}\n"
            f"Recent real feedback:\n{chr(10).join(real_history[-6:]) if real_history else 'No actions yet.'}\n"
            f"{loop_warning}"
        )
        if assisted:
            candidate_request = (
                "Output every available action that could plausibly advance the task or gather "
                "information needed to complete it; do not cap the list or include clearly "
                "irrelevant actions."
                if all_agent_candidates
                else f"Output up to {step_candidate_limit} exact available actions, one per line."
            )
            proposal_prompt = (
                "Update the progress ledger using real feedback and current state. Choose candidates "
                "that satisfy an unmet requirement or move toward a known goal object. Do not select "
                "unrelated objects merely because they are visible. Do not infer hidden/blocked objects "
                "without feedback. Infer the next action directly from the task goal and current state. "
                "Output exactly these sections: Progress: one concise sentence; Rationale: one "
                f"concise sentence; Candidates: followed by the candidate actions. {candidate_request}\n\n{context}"
            )
            proposed, proposal_response = generate(
                client, model, [
                    {"role": "system", "content": TASK_AGENT_PROMPT},
                    {"role": "user", "content": proposal_prompt},
                ], usage,
                {"task_id": item_id, "arm": arm, "step": step_index + 1, "phase": "candidate_generation"},
            )
            progress = parse_labeled_value(proposed, "Progress", progress)
            proposed_candidates = parse_candidates(proposed, state["available_actions"], step_candidate_limit)
            raw_proposed_candidates = list(proposed_candidates)
            proposed_candidates = [
                action for action in proposed_candidates
                if action_is_goal_consistent(action, task_goal, target_entities, inventory, real_history)
            ]
            goal_actions = [
                action for action in goal_related_actions(
                    task_goal, state["available_actions"], inventory, real_history, target_entities
                )
                if action_is_goal_consistent(action, task_goal, target_entities, inventory, real_history)
            ]
            candidates = []
            for action in goal_actions + proposed_candidates:
                if action not in candidates:
                    candidates.append(action)
                if len(candidates) >= step_candidate_limit:
                    break
            direct_goal_actions = [
                action for action in goal_actions
                if action.startswith(("take ", "clean ", "heat ", "cool ", "use ", "open ", "close "))
                or (action.startswith(("put ", "move ")) and re.search(r"\b(?:put|place|move)\b", task_goal, re.I))
            ]
            if repeated_state and direct_goal_actions and not all_agent_candidates:
                candidates = direct_goal_actions[:step_candidate_limit]
            candidate_filter_fallback = False
            if not candidates:
                candidates = raw_proposed_candidates or state["available_actions"][:step_candidate_limit]
                candidate_filter_fallback = True
            if not candidates:
                raise ValueError(f"No available actions to evaluate for task {item_id}: {task_goal}")
            if arm == "llm_wm_assisted":
                predictions = []
                for action in candidates:
                    messages = [{"role": "system", "content": LLM_WM_PROMPT}]
                    for example in demonstrations_for_action(demonstrations, action, fewshot):
                        messages.extend([
                            {
                                "role": "user",
                                "content": f"Observed ALFWorld state:\n{example['state']}\n\nAction to execute:\n{example['action']}",
                            },
                            {"role": "assistant", "content": example["next_state"]},
                        ])
                    prediction, _ = generate(
                        client,
                        model,
                        messages + [{
                            "role": "user",
                            "content": wm_input(
                                task_goal, state_text, inventory, real_history, calibration_history, action
                            ),
                        }],
                        usage,
                        {"task_id": item_id, "arm": arm, "step": step_index + 1, "phase": "world_model", "action": action},
                    )
                    predictions.append(prediction)
                simulation_results = simulate_candidates(
                    simulator, item_id, real_actions, state, candidates
                )
                llm_prediction_audit = [
                    {
                        "action": action,
                        "prediction": prediction,
                        "real_feedback": truth["observation"],
                        "exact_match": normalized_feedback(prediction) == normalized_feedback(truth["observation"]),
                        "token_f1": feedback_token_f1(prediction, truth["observation"]),
                    }
                    for action, prediction, truth in zip(candidates, predictions, simulation_results)
                ]
            else:
                simulation_results = simulate_candidates(
                    simulator, item_id, real_actions, state, candidates
                )
                predictions = [result["observation"] for result in simulation_results]
                llm_prediction_audit = []

            if arm == "simulator_assisted":
                outcome_heading = "Actual ALFWorld simulation feedback"
                outcome_details = {
                    index: f"Task done: {result['done']}; success: {result['success']}; reward: {result['reward']}\n"
                    f"Available actions after simulation: {', '.join(result['available_actions']) or 'none'}"
                    for index, result in enumerate(simulation_results, 1)
                }
            else:
                outcome_heading = "LLM-predicted ALFWorld feedback (uncertain)"
                outcome_details = {}
            choice_prompt = (
                f"Candidate outcomes are labeled as {outcome_heading}. Choose the action that best satisfies the "
                "next unmet requirement. Avoid unrelated objects and repeated states. Reply with "
                f"`Rationale: <one concise sentence naming why this beats the strongest alternative>` "
                f"and `Choice: N`. The only valid candidate IDs are "
                f"1 through {len(candidates)}; do not use indices from AVAILABLE ACTIONS.\n\n"
                f"{context}\nProgress: {progress}\n\n"
                + "\n\n".join(
                    f"Candidate {index}: {action}\n"
                    f"{outcome_heading}: {prediction}"
                    + (f"\n{outcome_details[index]}" if index in outcome_details else "")
                    for index, (action, prediction) in enumerate(zip(candidates, predictions), 1)
                )
            )
            decision, decision_response = generate(
                client, model, [
                    {"role": "system", "content": TASK_AGENT_PROMPT},
                    {"role": "user", "content": choice_prompt},
                ], usage,
                {"task_id": item_id, "arm": arm, "step": step_index + 1, "phase": "action_selection"},
            )
            action, selection_fallback = parse_choice(decision, candidates)
            selected_index = candidates.index(action)
            chosen_prediction = predictions[selected_index]
            chosen_prediction_audit = llm_prediction_audit[selected_index] if llm_prediction_audit else None
        else:
            react_prompt = (
                "Update the progress ledger from real feedback. Choose an available action that advances "
                "the next unmet requirement. Avoid unrelated objects and repeated states. Reply with "
                "`Progress: ...`, `Rationale: ...`, and `Action: <exact available action>`.\n\n"
                f"{context}"
            )
            decision, decision_response = generate(
                client, model, [
                    {"role": "system", "content": TASK_AGENT_PROMPT},
                    {"role": "user", "content": react_prompt},
                ], usage,
                {"task_id": item_id, "arm": "baseline", "step": step_index + 1, "phase": "action_selection"},
            )
            progress = parse_labeled_value(decision, "Progress", progress)
            match = re.search(r"Action\s*:\s*(.+)", decision, re.I)
            action_text = match.group(1).strip().splitlines()[0] if match else decision.strip().splitlines()[0]
            matching_actions = [
                candidate for candidate in state["available_actions"]
                if action_text.lower().startswith(candidate.lower())
            ]
            action = max(matching_actions, key=len) if matching_actions else action_text
            if action not in state["available_actions"]:
                raise ValueError(f"Agent returned an unavailable action: {action!r}")
            candidates, predictions, simulation_results = [action], [], []
            proposed, proposal_response = None, None
            candidate_filter_fallback = False
            selection_fallback = False
            simulation_results, llm_prediction_audit = [], []
            chosen_prediction, chosen_prediction_audit = None, None

        inventory_before = sorted(inventory)
        result = env.step(action)
        real_feedback = result["observation"]
        update_inventory(inventory, real_feedback)
        real_history.append(f"Action: {action}\nReal feedback: {real_feedback}")
        real_actions.append(action)
        if arm == "llm_wm_assisted" and chosen_prediction is not None:
            exact_match = normalized_feedback(chosen_prediction) == normalized_feedback(real_feedback)
            calibration_history.append(
                f"Action: {action}\nPredicted: {chosen_prediction}\n"
                f"Actual: {real_feedback}\nExact match: {exact_match}"
            )
            if chosen_prediction_audit is not None:
                chosen_prediction_audit["actual_real_feedback"] = real_feedback
        steps.append(
            {
                "step": step_index + 1,
                "state": state_text,
                "progress": progress,
                "loop_warning": loop_warning,
                "inventory_before_action": inventory_before,
                "candidates": candidates,
                "candidate_filter_fallback": candidate_filter_fallback,
                "proposal_output": proposed,
                "proposal_response": proposal_response,
                "predictions": predictions,
                "simulation_results": simulation_results if assisted else [],
                "simulation_restore_verified": bool(assisted),
                "llm_prediction_audit": llm_prediction_audit,
                "chosen_prediction": chosen_prediction,
                "chosen_prediction_audit": chosen_prediction_audit,
                "decision": decision,
                "decision_response": decision_response,
                "decision_rationale": parse_labeled_value(decision, "Rationale", "not provided"),
                "selection_fallback": selection_fallback,
                "action": action,
                "real_observation": real_feedback,
                "done": result.get("done", False),
                "reward": result.get("reward", 0),
            }
        )
        history.extend(
            [
                {"role": "assistant", "content": f"Action: {action}"},
                {"role": "user", "content": real_feedback},
            ]
        )
        state = {
            "observation": real_feedback,
            "available_actions": result.get("available_actions", []),
        }
        state_signatures.append(signature)
        if result.get("done"):
            success = float(result.get("reward", 0)) in (1.0, 100.0)
            break

    return {"success": int(success), "steps": steps, "usage": usage}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://127.0.0.1:36001")
    parser.add_argument("--test-tasks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--candidates",
        type=int,
        default=0,
        help="Maximum Agent-proposed candidates per step; 0 means keep all candidates",
    )
    parser.add_argument("--fewshot", type=int, default=3)
    parser.add_argument(
        "--demonstrations",
        default="outputs/api_wm_assisted/alfworld_transition_demos_compact.json",
        help="Training transition examples used only by the LLM world-model arm",
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument(
        "--arm",
        choices=("all", "both", "baseline", "baseline_simulator", "llm_wm_assisted", "simulator_assisted"),
        default="all",
    )
    parser.add_argument("--output", default="outputs/api_wm_assisted/alfworld_three_arm_comparison.json")
    parser.add_argument("--trace", help="Compressed JSONL containing complete API inputs and outputs")
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://spacetimeai.cc/v1"),
        timeout=90,
        max_retries=1,
    )
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    global TRACE_PATH
    TRACE_PATH = REPO_ROOT / args.trace if args.trace else REPO_ROOT / (args.output + ".trace.jsonl.gz")
    demonstrations_path = REPO_ROOT / args.demonstrations
    demonstrations = json.loads(demonstrations_path.read_text())["demonstrations"]
    mappings = json.loads(
        (REPO_ROOT / "AgentGym/agentenv-alfworld/configs/mappings_test.json").read_text()
    )
    tasks = [row for row in mappings if row["item_id"] == args.task_id] if args.task_id else choose_balanced_tasks(mappings, args.test_tasks)
    if not tasks:
        raise ValueError(f"No test task found for item_id={args.task_id}")
    arms = (
        ("baseline", "llm_wm_assisted", "simulator_assisted")
        if args.arm in ("all", "both")
        else ("baseline", "simulator_assisted")
        if args.arm == "baseline_simulator"
        else (args.arm,)
    )
    env = AlfWorldClient(args.env_url)
    simulator = AlfWorldClient(args.env_url) if any(arm != "baseline" for arm in arms) else None
    results = []
    run_error = None
    try:
        for row in tasks:
            result = {"item_id": row["item_id"], "task_type": row["task_type"]}
            results.append(result)
            for arm in arms:
                print(f"task={row['item_id']} {arm}", flush=True)
                try:
                    result[arm] = run_episode(
                        env,
                        simulator,
                        client,
                        model,
                        row,
                        args.max_steps,
                        arm,
                        args.candidates,
                        demonstrations,
                        args.fewshot,
                    )
                except Exception as exc:
                    result[arm] = {"success": 0, "steps": [], "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}, "error": f"{type(exc).__name__}: {exc}"}
                    print(f"task={row['item_id']} {arm}_error={result[arm]['error']}", flush=True)
                save_results(args, model, results, run_error)
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
        if simulator:
            simulator.close()

    output_path = save_results(args, model, results, run_error)
    metrics = json.loads(output_path.read_text())["metrics"]
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if run_error:
        print(f"run_error={run_error}")
    print(f"results={output_path}")


def save_results(args, model, results, run_error):
    arm_names = ("baseline", "llm_wm_assisted", "simulator_assisted")
    runs = {
        arm: [item[arm] for item in results if arm in item]
        for arm in arm_names
    }
    valid = {
        arm: [run for run in runs[arm] if "error" not in run]
        for arm in arm_names
    }
    metrics = {
        "requested_tasks": 1 if args.task_id else args.test_tasks,
        "task_id": args.task_id,
        "max_steps": args.max_steps,
        "candidate_count": args.candidates or None,
        "candidate_policy": "all_agent_candidates" if args.candidates <= 0 else "up_to_n",
        "fewshot_examples_per_action": args.fewshot,
        "trace_file": str(TRACE_PATH),
        "arms": {
            arm: {
                "attempted": len(runs[arm]),
                "errors": sum("error" in run for run in runs[arm]),
                "successes": sum(run["success"] for run in valid[arm]),
                "success_rate": sum(run["success"] for run in valid[arm]) / len(valid[arm]) if valid[arm] else 0,
                "mean_steps": sum(len(run["steps"]) for run in valid[arm]) / len(valid[arm]) if valid[arm] else 0,
                "usage": {
                    key: sum(run.get("usage", {}).get(key, 0) for run in runs[arm])
                    for key in ("calls", "prompt_tokens", "completion_tokens")
                },
            }
            for arm in arm_names
        },
        "paired_vs_baseline": {},
    }
    for arm in ("llm_wm_assisted", "simulator_assisted"):
        pairs = [
            item for item in results
            if "baseline" in item and arm in item
            and "error" not in item["baseline"] and "error" not in item[arm]
        ]
        step_savings = [len(item["baseline"]["steps"]) - len(item[arm]["steps"]) for item in pairs]
        metrics["paired_vs_baseline"][arm] = {
            "valid_pairs": len(pairs),
            "both_successes": sum(item["baseline"]["success"] and item[arm]["success"] for item in pairs),
            "arm_success_wins": sum(item[arm]["success"] > item["baseline"]["success"] for item in pairs),
            "fewer_steps_count": sum(delta > 0 for delta in step_savings),
            "more_steps_count": sum(delta < 0 for delta in step_savings),
            "mean_steps_saved": sum(step_savings) / len(step_savings) if step_savings else 0,
        }

    audits = [
        audit
        for run in valid["llm_wm_assisted"]
        for step in run["steps"]
        for audit in step.get("llm_prediction_audit", [])
    ]
    selected_audits = [
        step["chosen_prediction_audit"]
        for run in valid["llm_wm_assisted"]
        for step in run["steps"]
        if step.get("chosen_prediction_audit")
    ]
    metrics["llm_prediction_accuracy"] = {
        "candidate_count": len(audits),
        "candidate_exact_matches": sum(audit["exact_match"] for audit in audits),
        "candidate_exact_match_rate": sum(audit["exact_match"] for audit in audits) / len(audits) if audits else 0,
        "candidate_mean_token_f1": sum(audit["token_f1"] for audit in audits) / len(audits) if audits else 0,
        "selected_count": len(selected_audits),
        "selected_exact_matches": sum(audit["exact_match"] for audit in selected_audits),
        "selected_exact_match_rate": sum(audit["exact_match"] for audit in selected_audits) / len(selected_audits) if selected_audits else 0,
    }
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model": model,
                "base_url": os.getenv("OPENAI_BASE_URL", "https://spacetimeai.cc/v1"),
                "prediction_methods": {
                    "llm_wm_assisted": "LLM predicts each candidate; simulator results are logged only for offline accuracy evaluation and are not shown to the acting Agent",
                    "simulator_assisted": "ALFWorld branch simulation with reset-and-replay state restoration",
                },
                "metrics": metrics,
                "run_error": run_error,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return output_path


if __name__ == "__main__":
    main()
