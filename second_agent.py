import json
import math
from collections import deque
from typing import Dict, Optional, Tuple

from rapidfuzz import fuzz

from scorer.graph_context import navigate
from scorer.grid_repsentation import convert2grid
from src.data_loader import get_data_by_instruction
from templates import navigator_batch

OUTPUT_FILE = 'navigator_agent.jsonl'

navigation_instructions = []
with open(f'predications/test_seen_processed.jsonl', 'r') as f:
    for l in f.readlines():
        navigation_instructions.append(json.loads(l))


def _get_heading_to_node(area_links, area_nodes, start_node_id, end_node_id):
    # 1. Build the Graph (Adjacency List) from area_links
    graph = {}
    for link in area_links:
        src = link['source']
        tgt = link['target']
        if src not in graph:
            graph[src] = []
        graph[src].append(tgt)

    # 2. Find the Path (Breadth-First Search)
    # This finds the shortest sequence of nodes connecting Start to End
    queue = deque([[start_node_id]])
    visited = {start_node_id}
    found_path = None

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == end_node_id:
            found_path = path
            break

        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

    if not found_path:
        return "Error: No path found connecting these nodes."

    # 3. Get the Immediate Previous Node
    # The node just before the destination is at index -2
    if len(found_path) < 2:
        return "Error: Start and End nodes are the same."

    prev_node_id = found_path[-2]

    # 4. Calculate Heading (Bearing)
    nodes = area_nodes
    lat1 = math.radians(nodes[prev_node_id]['lat'])
    lon1 = math.radians(nodes[prev_node_id]['lng'])
    lat2 = math.radians(nodes[end_node_id]['lat'])
    lon2 = math.radians(nodes[end_node_id]['lng'])

    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    bearing = math.degrees(math.atan2(y, x))
    compass_bearing = (bearing + 360) % 360

    print(found_path)

    return found_path, compass_bearing


def _build_navigator_message(
        area_data: Dict,
        current_node_id: str,
        current_heading: float,
        pois: dict = None,
        poi_mapping: dict = None,
        target_poi: str = None,
        previous_visited=None
):
    """
    Build the navigation context message for the node navigator using DSL format.
    Uses navigate() to get the full path ahead and encodes it with route DSL format.
    """
    # Use navigate() to get the list of nodes in front
    path_data = navigate(
        map_json=area_data,
        starting_point=current_node_id,
        heading=int(current_heading),
        pois=pois,
        poi_mapping=poi_mapping,
        units=1,
        last_instruction=False
    )

    if previous_visited is None:
        previous_visited = []

    return path_data, convert2grid(
        path_data,
        previous_visited,
        area_nodes=area_data['area_nodes'],  # REQUIRED for positioning
        area_pois=area_data['area_pois'],  # REQUIRED for positioning
    ).tolist()


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


requests = []
for idx, ni in enumerate(navigation_instructions):
    print(f'id {idx} started to run')
    split_file = "test_seen.json"

    instruction_id = int(ni['key'])
    sub_goals = ni['sub_goals']
    landmarks = ni['landmarks']

    area_data = get_data_by_instruction(
        instruction_id,
        split_file,
        base_path='./data/map2seq/',
        neighbor_degrees=20
    )

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

    # while current_sub_idx < len(sub_goals) and total_steps < max_total_steps:
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

    path_data, matrix_representation = _build_navigator_message(
        area_data=area_data,
        current_node_id=current_node_id,
        current_heading=current_heading,
        previous_visited=previous_visited_path,
        pois=sub_instruction_landmarks,
        poi_mapping=unique_identifiers_mapping
    )

    locations: Dict[str, Optional[Tuple[str, str]]] = {"S": None, "P": None}

    for r, row in enumerate(matrix_representation):
        for c, val in enumerate(row):
            if val in locations:
                locations[val] = (str(r), str(c))

    requests.append(navigator_batch(
        key=ni['key'],
        navigation_instruction=navigation_instruction,
        navigator_message=sub_instruction_text,
        current_heading=f"{current_heading:.2f}",
        location_legend="S = Current Position" if current_sub_idx == 0 else "S = Start Position, P = Current Position",
        landmarks=landmarks_mapping,
        landmark_legend=landmark_legend,
        planning_state="\n".join(
            f"{i + 1}. {sub.get('description', '')} "
            f"({'IN_PROGRESS' if i == current_sub_idx else 'COMPLETED' if i < current_sub_idx else 'TODO'})"
            for i, sub in enumerate(sub_goals)
        ),
        your_position=f"[{locations['S'][0]}, {locations['S'][1]}] (marked as S)" if current_sub_idx == 0 \
            else f"[{locations['P'][0]}, {locations['P'][1]}] (marked as P) and you start position [{locations['S'][0]}, {locations['S'][1]}] (marked as S)",
        matrix_representation="\n".join(map(str, matrix_representation)),
    ))

with open(OUTPUT_FILE, "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")
