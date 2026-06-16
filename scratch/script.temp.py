import json
from typing import Dict, Optional, Tuple

import json_repair
from rapidfuzz import fuzz

from groke.prompts import GRID_NAVIGATOR_PROMPT, TEXTUAL_NAVIGATOR_PROMPT, JSON_NAVIGATOR_PROMPT, GRAPH_VIS_NAVIGATOR_PROMPT
from groke.scorer.graph_context import navigate
from groke.scorer.grid_representation import convert2grid
from groke.scorer.presentation_formats import generate_all_representations
from groke.data_loader import get_data_by_instruction
from groke.visualize import visualize_area
from groke.templates import navigator_batch

previous_visited = []


def _get_previous_nodes(self, path_data, previous_visited):
    new_path_data = []

    for pd in path_data:
        if pd['node_id'] not in previous_visited:
            continue

        new_connectivity = []
        connectivity = pd.get('connectivity', {})
        if connectivity:
            for con in connectivity:
                if con['node_id'] in previous_visited:
                    new_connectivity.append(con)
        pd['connectivity'] = new_connectivity
        new_path_data.append(pd)

    return new_path_data


def _find_available_pois(area_pois, landmarks):
    available_pois = {}

    for node_id, info in area_pois.items():
        tags = json.loads(info["tags"])

        name = (
            f"{tags.get('name', '')} "
            f"{tags.get('amenity', '').replace('_', ' ')} "
            f"{tags.get('cuisine', '').replace('_', ' ')} "
            f"{tags.get('leisure', '').replace('_', ' ')} "
            f"{tags.get('tourism', '').replace('_', ' ')} "
            f"{tags.get('shop', '').replace('_', ' ')}"
        ).strip()

        for landmark in landmarks:
            score = fuzz.partial_ratio(landmark['name'], name)
            if score > 70:
                available_pois.setdefault(landmark['name'], []).append(node_id)

    return available_pois


def _get_unique_identifiers(landmark_in_area):
    used = set()
    mapping = {}

    # Characters that are disallowed
    forbidden = {"S", "P"}

    # Fallback alphabet
    alphabet = [chr(i) for i in range(ord("A"), ord("Z") + 1) if chr(i) not in forbidden]

    for name in landmark_in_area:
        assigned = None

        # Try characters from the name first
        for ch in name:
            upper = ch.upper()
            if ch.isalpha() and upper not in forbidden and upper not in used:
                assigned = upper
                used.add(upper)
                break

        # Fallback if needed
        if assigned is None:
            for c in alphabet:
                if c not in used:
                    assigned = c
                    used.add(c)
                    break

        mapping[name] = assigned

    return mapping


def _get_current_instruction_poi(sub_instruction_text, landmarks_in_area):
    current_sub_instructions_landmarks = {}

    for lia_name, lia_node_id in landmarks_in_area.items():
        if lia_name in sub_instruction_text:
            current_sub_instructions_landmarks[lia_name] = lia_node_id

    return current_sub_instructions_landmarks


navigation_instructions = []
with open('ablation_study/test_seen_200_predictions.jsonl') as f:
    for record in map(json.loads, f):
        for candidate in record['response']['candidates']:
            for part in candidate['content']['parts']:
                if 'thought' not in part:
                    navigation_instructions.append({
                        "key": record['key'],
                        **json_repair.repair_json(part['text'], return_objects=True)
                    })

grid_prompts = []
json_prompts = []
textual_prompts = []
graph_vis_prompts = []
requests = []
for idx, ni in enumerate(navigation_instructions):
    # if ni['key'] != '12181':
    #     continue

    print(f"id {ni['key']} started to run")

    split_file = "test_seen_200.json"

    instruction_id = int(ni['key'])
    sub_goals = ni['sub_goals']
    landmarks = ni['landmarks']

    area_data = get_data_by_instruction(
        instruction_id,
        split_file,
        base_path='./data/map2seq/',
        neighbor_degrees=20
    )

    # visualize_area(area_data)

    instruction_data = area_data.get("instruction_data", {})
    navigation_instruction = instruction_data.get("instructions", "")

    # Extract graph data for calculate headings in the future
    area_nodes = area_data.get("area_nodes", {})
    area_links = area_data.get("area_links", [])
    area_pois = area_data.get("area_pois", {})

    landmarks_in_area = _find_available_pois(area_pois, landmarks)

    # Get route information for initialization
    route = instruction_data.get("route", {})
    osm_path = route.get("osm_path", [])

    # Initialize at start node
    current_node_id = osm_path[0]
    current_heading = float(route.get("initial_heading", 0))
    previous_visited_path = []

    # ==== Stage 3: Navigation Loop ====
    navigation_trajectory = []
    predicted_path = [current_node_id]  # Track the predicted path
    current_sub_idx = 0
    max_steps_per_sub = 15  # Prevent infinite loops
    steps_in_current_sub = 0
    total_steps = 0
    max_total_steps = 100  # Global safety limit

    current_sub = sub_goals[0]
    sub_instruction_text = current_sub.get("description", "")

    # Find the landmark related to instruction and add it to context + add it as legend ...
    unique_identifiers_mapping = None

    sub_instruction_landmarks = _get_current_instruction_poi(sub_instruction_text, landmarks_in_area)
    if sub_instruction_landmarks:
        unique_identifiers_mapping = _get_unique_identifiers(sub_instruction_landmarks)
        landmarks_mapping = "Landmark: " + ", ".join(unique_identifiers_mapping.keys())
        landmark_legend = ", " + ", ".join(
            f"{char} = {name} (landmark)" for name, char in unique_identifiers_mapping.items()
        )
    else:
        landmarks_mapping = ""
        landmark_legend = ""

    path_data = navigate(
        map_json=area_data,
        starting_point=current_node_id,
        heading=int(current_heading),
        pois=sub_instruction_landmarks,
        poi_mapping=unique_identifiers_mapping
    )

    matrix = convert2grid(path_data, previous_visited, area_nodes=area_data['area_nodes'],
                          area_pois=area_data['area_pois'])

    # Get all new formats
    representations = generate_all_representations(
        path_data,
        previous_visited,
        unique_identifiers_mapping,
        current_heading,
        area_nodes=area_data['area_nodes'],
        area_pois=area_data['area_pois'],
        area_links=area_data['area_links'],
        branch_depth=2,
        matrix_representation=matrix.tolist()
    )
    print(path_data)

    # Use in prompts
    # print(representations['textual'])   # Human-readable
    # print(representations['json'])      # Structured data
    # print(representations['graphviz'])  # Graph format
    # print(representations['matrix'])    # Original matrix

    key = ni['key']

    planning_state = "\n".join(
        f"{i + 1}. {sub.get('description', '')} "
        f"({'IN_PROGRESS' if i == current_sub_idx else 'COMPLETED' if i < current_sub_idx else 'TODO'})"
        for i, sub in enumerate(sub_goals)
    )

    grid_next_place = {
        "type": "ARRAY",
        "items": {
            "type": "INTEGER"
        }
    }

    text_next_place = {
        "type": "STRING"
    }

    textual_prompt = TEXTUAL_NAVIGATOR_PROMPT.format(
        navigation_instruction=navigation_instruction,
        navigator_message=sub_instruction_text,
        navigation_context=representations['textual'],
        planning_state=planning_state
    )

    landmarks = landmarks_mapping
    locations: Dict[str, Optional[Tuple[str, str]]] = {"S": None, "P": None}
    matrix_representation = representations['matrix']
    for r, row in enumerate(matrix_representation):
        for c, val in enumerate(row):
            if val in locations:
                locations[val] = (str(r), str(c))

    your_position = f"[{locations['S'][0]}, {locations['S'][1]}] (marked as S)" if current_sub_idx == 0 \
        else f"[{locations['P'][0]}, {locations['P'][1]}] (marked as P) and you start position [{locations['S'][0]}, {locations['S'][1]}] (marked as S)"

    current_heading = f"{current_heading:.2f}"
    location_legend = "S = Current Position" if current_sub_idx == 0 else "S = Start Position, P = Current Position"

    if sub_instruction_landmarks:
        unique_identifiers_mapping = _get_unique_identifiers(sub_instruction_landmarks)
        landmarks_mapping = "Landmark: " + ", ".join(unique_identifiers_mapping.keys())
        landmark_legend = ", " + ", ".join(
            f"{char} = {name} (landmark)" for name, char in unique_identifiers_mapping.items()
        )
    else:
        landmarks_mapping = ""
        landmark_legend = ""

    grid_prompt = GRID_NAVIGATOR_PROMPT.format(
        navigation_instruction=navigation_instruction,
        navigator_message=sub_instruction_text,
        landmarks=landmarks,
        planning_state=planning_state,
        your_position=your_position,
        current_heading=current_heading,
        matrix_representation="\n".join(map(str, matrix_representation)),
        location_legend=location_legend,
        landmark_legend=landmark_legend
    )

    json_prompt = JSON_NAVIGATOR_PROMPT.format(
        navigation_instruction=navigation_instruction,
        navigator_message=sub_instruction_text,
        navigation_context=f'''```json
{representations['json']}
```''',
        planning_state=planning_state
    )

    graph_vis_prompt = GRAPH_VIS_NAVIGATOR_PROMPT.format(
        navigation_instruction=navigation_instruction,
        navigator_message=sub_instruction_text,
        navigation_context=representations['graphviz'],
        planning_state=planning_state,
    )

    json_prompts.append(navigator_batch(key, json_prompt, text_next_place))
    grid_prompts.append(navigator_batch(key, grid_prompt,  grid_next_place))
    textual_prompts.append(navigator_batch(key, textual_prompt, text_next_place))
    graph_vis_prompts.append(navigator_batch(key, graph_vis_prompt, text_next_place))


with open('graph_vis.jsonl', "w") as f:
    for req in graph_vis_prompts:
        f.write(json.dumps(req) + "\n")

with open('grid.jsonl', "w") as f:
    for req in grid_prompts:
        f.write(json.dumps(req) + "\n")

with open('textual.jsonl', "w") as f:
    for req in textual_prompts:
        f.write(json.dumps(req) + "\n")

with open('json.jsonl', "w") as f:
    for req in json_prompts:
        f.write(json.dumps(req) + "\n")