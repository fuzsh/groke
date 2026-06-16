"""
Script to process navigation results from the first step and generate prompts for subsequent steps.

This script:
1. Reads results from ablation_study/*.jsonl files
2. Determines the current step based on subplan_status
3. Updates heading based on the next_place
4. Includes previous path information
5. Generates new prompt files for the next navigation steps
"""

import json
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque

import json_repair
from rapidfuzz import fuzz

from groke.prompts import (
    GRID_NAVIGATOR_PROMPT,
    TEXTUAL_NAVIGATOR_PROMPT,
    JSON_NAVIGATOR_PROMPT,
    GRAPH_VIS_NAVIGATOR_PROMPT
)
from groke.scorer.graph_context import navigate, calculate_heading_from_coords
from groke.scorer.grid_representation import convert2grid, get_node_id_from_position
from groke.scorer.presentation_formats import generate_all_representations
from groke.data_loader import get_data_by_instruction
from groke.templates import navigator_batch


def load_navigation_instructions(file_path: str) -> Dict[str, Dict]:
    """Load navigation instructions (sub_goals and landmarks) from predictions file."""
    instructions = {}
    with open(file_path) as f:
        for record in map(json.loads, f):
            try:
                for candidate in record['response']['candidates']:
                    for part in candidate['content']['parts']:
                        if 'thought' not in part:
                            parsed = json_repair.repair_json(part['text'], return_objects=True)
                            instructions[str(record['key'])] = {
                                "key": str(record['key']),
                                "sub_goals": parsed.get('sub_goals', []),
                                "landmarks": parsed.get('landmarks', [])
                            }
            except Exception as e:
                print(f"Error parsing instruction for key {record.get('key')}: {e}")
                continue
    return instructions


def load_method_results(file_path: str, method_name: str) -> Dict[str, Dict]:
    """Load results from a method's JSONL file."""
    results = {}
    with open(file_path) as f:
        for line in f:
            record = json.loads(line)
            try:
                for candidate in record['response']['candidates']:
                    for part in candidate['content']['parts']:
                        if 'thought' not in part:
                            parsed = json_repair.repair_json(part['text'], return_objects=True)
                            results[str(record['key'])] = {
                                "key": str(record['key']),
                                "subplan_status": parsed.get('subplan_status', 'IN_PROGRESS'),
                                "next_place": parsed.get('next_place'),
                                "method": method_name
                            }
            except Exception as e:
                continue
    return results


def resolve_grid_next_place(
        next_place,
        area_data: Dict,
        current_node_id: str,
        current_heading: float,
        previous_visited: List[str],
        pois: Dict = None,
        poi_mapping: Dict = None
) -> None | str | tuple[list[dict], Any]:
    """Convert grid coordinates [row, col] to node_id."""
    if next_place is None:
        return None

    # If already a string (node_id), return as-is
    if isinstance(next_place, str):
        return next_place

    # If it's coordinates [row, col], resolve to node_id
    if isinstance(next_place, list) and len(next_place) == 2:
        path_data = navigate(
            map_json=area_data,
            starting_point=current_node_id,
            heading=int(current_heading),
            pois=pois,
            poi_mapping=poi_mapping,
            units=1,
            last_instruction=False
        )
        return path_data, get_node_id_from_position(path_data, previous_visited, next_place)

    return None


def prev_path_fetcher(
        area_data: Dict,
        current_node_id: str,
        current_heading: float,
        pois: Dict = None,
        poi_mapping: Dict = None
) -> list[dict]:
    return navigate(
        map_json=area_data,
        starting_point=current_node_id,
        heading=int(current_heading),
        pois=pois,
        poi_mapping=poi_mapping,
        units=1,
        last_instruction=False
    )


def _find_available_pois(area_pois: Dict, landmarks: List) -> Dict[str, List[str]]:
    """Find POIs in the area that match landmarks from instructions."""
    available_pois = {}

    for node_id, info in area_pois.items():
        tags = json.loads(info["tags"]) if isinstance(info["tags"], str) else info["tags"]

        name = (
            f"{tags.get('name', '')} "
            f"{tags.get('amenity', '').replace('_', ' ')} "
            f"{tags.get('cuisine', '').replace('_', ' ')} "
            f"{tags.get('leisure', '').replace('_', ' ')} "
            f"{tags.get('tourism', '').replace('_', ' ')} "
            f"{tags.get('shop', '').replace('_', ' ')}"
        ).strip()

        for landmark in landmarks:
            # Handle both dict and string landmarks
            if isinstance(landmark, str):
                landmark_name = landmark
            else:
                landmark_name = landmark.get('name', '') if isinstance(landmark, dict) else str(landmark)

            score = fuzz.partial_ratio(landmark_name, name)
            if score > 70:
                available_pois.setdefault(landmark_name, []).append(node_id)

    return available_pois


def _get_unique_identifiers(landmark_in_area: Dict) -> Dict[str, str]:
    """Assign unique letter identifiers to landmarks."""
    used = set()
    mapping = {}
    forbidden = {"S", "P"}
    alphabet = [chr(i) for i in range(ord("A"), ord("Z") + 1) if chr(i) not in forbidden]

    for name in landmark_in_area:
        assigned = None
        for ch in name:
            upper = ch.upper()
            if ch.isalpha() and upper not in forbidden and upper not in used:
                assigned = upper
                used.add(upper)
                break

        if assigned is None:
            for c in alphabet:
                if c not in used:
                    assigned = c
                    used.add(c)
                    break

        mapping[name] = assigned

    return mapping


def _get_previous_nodes(path_data, previous_visited, current_node_id=None):
    """
    Filter path_data to only include previously visited nodes with their connections.

    Args:
        path_data: Full path data from navigate()
        previous_visited: List of node IDs that were visited
        current_node_id: The current position node ID (to include connection from last visited node)
    """
    new_path_data = []

    for pd in path_data:
        if pd['node_id'] not in previous_visited:
            continue

        new_connectivity = []
        connectivity = pd.get('connectivity', {})
        if connectivity:
            for con in connectivity:
                # Include connection if it's to a visited node
                if con['node_id'] in previous_visited:
                    new_connectivity.append(con)
                # Also include connection to current position (from the last visited node)
                elif current_node_id and con['node_id'] == current_node_id:
                    new_connectivity.append(con)
        pd['connectivity'] = new_connectivity
        new_path_data.append(pd)

    return new_path_data


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
        raise ValueError("Error: No path found connecting these nodes.")

    # 3. Get the Immediate Previous Node
    # The node just before the destination is at index -2
    if len(found_path) < 2:
        raise ValueError("Error: Start and End nodes are the same.")

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

    return found_path, compass_bearing


def _get_current_instruction_poi(sub_instruction_text: str, landmarks_in_area: Dict) -> Dict[str, List[str]]:
    """Find landmarks mentioned in the current sub-instruction."""
    current_sub_instructions_landmarks = {}

    for lia_name, lia_node_id in landmarks_in_area.items():
        if lia_name.lower() in sub_instruction_text.lower():
            current_sub_instructions_landmarks[lia_name] = lia_node_id

    return current_sub_instructions_landmarks


def calculate_new_heading(
        area_nodes: Dict,
        from_node_id: str,
        to_node_id: str
) -> float:
    """Calculate heading from one node to another."""
    if from_node_id not in area_nodes or to_node_id not in area_nodes:
        return 0.0

    from_node = area_nodes[from_node_id]
    to_node = area_nodes[to_node_id]

    return calculate_heading_from_coords(
        from_node['lat'], from_node['lng'],
        to_node['lat'], to_node['lng']
    )


def generate_next_step_prompts(
        key: str,
        instruction_data: Dict,
        area_data: Dict,
        current_node_id: str,
        current_heading: float,
        current_sub_idx: int,
        previous_visited: List[str],
        sub_goals: List[Dict],
        landmarks: List[Dict],
        target_method: str = "all"
) -> Dict[str, Dict]:
    """Generate prompts for the next navigation step.

    Args:
        target_method: Which method to generate prompts for.
                      Options: "json", "textual", "grid", "graph_vis", or "all"
    """

    navigation_instruction = instruction_data.get("instructions", "")
    area_nodes = area_data.get("area_nodes", {})
    area_pois = area_data.get("area_pois", {})

    # Get current sub-goal
    if current_sub_idx >= len(sub_goals):
        return {}  # No more sub-goals

    current_sub = sub_goals[current_sub_idx]
    # Handle both dict and string sub-goals
    if isinstance(current_sub, str):
        sub_instruction_text = current_sub
    else:
        sub_instruction_text = current_sub.get("description", "") if isinstance(current_sub, dict) else str(current_sub)

    # Find landmarks in area
    landmarks_in_area = _find_available_pois(area_pois, landmarks)

    # Find landmarks for current instruction
    sub_instruction_landmarks = _get_current_instruction_poi(sub_instruction_text, landmarks_in_area)

    unique_identifiers_mapping = None
    if sub_instruction_landmarks:
        unique_identifiers_mapping = _get_unique_identifiers(sub_instruction_landmarks)
        landmarks_mapping = "Landmark: " + ", ".join(unique_identifiers_mapping.keys())
        landmark_legend = ", " + ", ".join(
            f"{char} = {name} (landmark)" for name, char in unique_identifiers_mapping.items()
        )
    else:
        landmarks_mapping = ""
        landmark_legend = ""

    # Generate path data from current position
    path_data = navigate(
        map_json=area_data,
        starting_point=current_node_id,
        heading=int(current_heading),
        pois=sub_instruction_landmarks,
        poi_mapping=unique_identifiers_mapping
    )

    # Generate grid matrix
    matrix = convert2grid(
        path_data,
        previous_visited,
        area_nodes=area_data['area_nodes'],
        area_pois=area_data['area_pois']
    )

    # Generate all representations
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

    # Build planning state with previous steps marked as COMPLETED
    def get_sub_description(sub):
        if isinstance(sub, str):
            return sub
        return sub.get('description', '') if isinstance(sub, dict) else str(sub)

    planning_state = "\n".join(
        f"{i + 1}. {get_sub_description(sub)} "
        f"({'IN_PROGRESS' if i == current_sub_idx else 'COMPLETED' if i < current_sub_idx else 'TODO'})"
        for i, sub in enumerate(sub_goals)
    )

    # Build previous path representation (extract node_ids from path_data format)
    previous_path_str = ""
    if previous_visited:
        # previous_visited is in path_data format (list of dicts with 'node_id')
        visited_node_ids = [n.get('node_id', str(n)) if isinstance(n, dict) else str(n) for n in previous_visited[-5:]]
        previous_path_str = f"\nPrevious Path (already visited): {' -> '.join(visited_node_ids)}"

    # Grid-specific variables
    grid_next_place = {
        "type": "ARRAY",
        "items": {"type": "INTEGER"}
    }
    text_next_place = {"type": "STRING"}

    # Position handling for grid
    matrix_representation = representations['matrix']
    locations: Dict[str, Optional[Tuple[str, str]]] = {"S": None, "P": None}
    for r, row in enumerate(matrix_representation):
        for c, val in enumerate(row):
            if val in locations:
                locations[val] = (str(r), str(c))

    # For subsequent steps, S = start, P = current position
    if current_sub_idx > 0 or locations['P'] is not None:
        your_position = f"[{locations['P'][0]}, {locations['P'][1]}] (marked as P) and start position [{locations['S'][0]}, {locations['S'][1]}] (marked as S)"
        location_legend = "S = Start Position, P = Current Position"
    else:
        your_position = f"[{locations['S'][0]}, {locations['S'][1]}] (marked as S)" if locations['S'] else "Unknown"
        location_legend = "S = Current Position"

    prompts = {}

    # Generate prompts for each method
    textual_prompt = TEXTUAL_NAVIGATOR_PROMPT.format(
        navigation_instruction=navigation_instruction + previous_path_str,
        navigator_message=sub_instruction_text,
        navigation_context=representations['textual'],
        planning_state=planning_state
    )

    grid_prompt = GRID_NAVIGATOR_PROMPT.format(
        navigation_instruction=navigation_instruction,
        navigator_message=sub_instruction_text,
        landmarks=landmarks_mapping,
        planning_state=planning_state,
        your_position=your_position,
        current_heading=f"{current_heading:.2f}",
        matrix_representation="\n".join(map(str, matrix_representation)),
        location_legend=location_legend,
        landmark_legend=landmark_legend
    )

    json_prompt = JSON_NAVIGATOR_PROMPT.format(
        navigation_instruction=navigation_instruction + previous_path_str,
        navigator_message=sub_instruction_text,
        navigation_context=f'''```json
{representations['json']}
```''',
        planning_state=planning_state
    )

    graph_vis_prompt = GRAPH_VIS_NAVIGATOR_PROMPT.format(
        navigation_instruction=navigation_instruction + previous_path_str,
        navigator_message=sub_instruction_text,
        navigation_context=representations['graphviz'],
        planning_state=planning_state,
    )

    # Only generate prompts for the target method (or all if "all")
    if target_method == "all" or target_method == "json":
        prompts['json'] = navigator_batch(key, json_prompt, text_next_place)
    if target_method == "all" or target_method == "grid":
        prompts['grid'] = navigator_batch(key, grid_prompt, grid_next_place)
    if target_method == "all" or target_method == "textual":
        prompts['textual'] = navigator_batch(key, textual_prompt, text_next_place)
    if target_method == "all" or target_method == "graph_vis":
        prompts['graph_vis'] = navigator_batch(key, graph_vis_prompt, text_next_place)

    return prompts


def process_results_for_method(
        method_name: str,
        results_file: str,
        navigation_instructions: Dict,
        split_file: str = "test_seen_200.json",
        output_prefix: str = "step2"
) -> None:
    """Process results from a single method and generate next step prompts."""

    print(f"\n=== Processing method: {method_name} ===")

    # Load results for this method
    results = load_method_results(results_file, method_name)
    print(f"Loaded {len(results)} results")

    prompts_by_method = defaultdict(list)

    for key, result in results.items():
        if key not in navigation_instructions:
            print(f"Warning: No navigation instructions found for key {key}")
            continue

        nav_inst = navigation_instructions[key]
        sub_goals = nav_inst.get('sub_goals', [])
        landmarks = nav_inst.get('landmarks', [])

        if not sub_goals:
            print(f"Warning: No sub_goals for key {key}")
            continue

        # Load area data
        try:
            area_data = get_data_by_instruction(
                int(key),
                split_file,
                base_path='./data/map2seq/',
                neighbor_degrees=20
            )
        except Exception as e:
            print(f"Error loading area data for key {key}: {e}")
            continue

        instruction_data = area_data.get("instruction_data", {})
        route = instruction_data.get("route", {})
        osm_path = route.get("osm_path", [])

        if not osm_path:
            print(f"Warning: No osm_path for key {key}")
            continue

        # Initial state
        start_node_id = osm_path[0]
        initial_heading = float(route.get("initial_heading", 0))

        # Get result from first step
        subplan_status = result.get('subplan_status', 'IN_PROGRESS')
        next_place = result.get('next_place')

        # Resolve next_place to node_id if needed (for grid method)
        if method_name == 'grid' and isinstance(next_place, list):
            prev_path_data, next_place = resolve_grid_next_place(
                next_place,
                area_data,
                start_node_id,
                initial_heading,
                []
            )
        else:
            prev_path_data = prev_path_fetcher(
                area_data,
                start_node_id,
                initial_heading,
            )

        # Determine current sub-instruction index
        if subplan_status == 'COMPLETED':
            current_sub_idx = 1  # Move to next sub-goal
        else:
            current_sub_idx = 0  # Stay on same sub-goal

        # Calculate new position and heading
        # Note: previous_visited must be in path_data format (list of dicts with node_id, etc.)
        if next_place and next_place in area_data['area_nodes']:
            current_node_id = next_place
            try:
                visited_route, current_heading = _get_heading_to_node(
                    area_links=area_data['area_links'],
                    area_nodes=area_data['area_nodes'],
                    start_node_id=start_node_id, end_node_id=next_place
                )
            except ValueError as e:
                print(f"Error getting heading for key {key}: {e}")
                continue

            if visited_route:
                previous_visited = _get_previous_nodes(prev_path_data, visited_route, current_node_id)

        else:
            # Fallback to start position
            current_node_id = start_node_id
            current_heading = initial_heading
            previous_visited = []

        # Check if we still have sub-goals to process
        if current_sub_idx >= len(sub_goals):
            print(f"Key {key}: Navigation completed (all sub-goals done)")
            continue

        # Generate next step prompts (only for the same method)
        try:
            prompts = generate_next_step_prompts(
                key=key,
                instruction_data=instruction_data,
                area_data=area_data,
                current_node_id=current_node_id,
                current_heading=current_heading,
                current_sub_idx=current_sub_idx,
                previous_visited=previous_visited,
                sub_goals=sub_goals,
                landmarks=landmarks,
                target_method=method_name,  # Only generate for same method
            )

            for method, prompt in prompts.items():
                prompts_by_method[method].append(prompt)

        except Exception as e:
            print(f"Error generating prompts for key {key}: {e}")
            continue

    # Write output files
    for method, prompts in prompts_by_method.items():
        output_file = f"{output_prefix}_{method}.jsonl"
        with open(output_file, 'w') as f:
            for prompt in prompts:
                f.write(json.dumps(prompt) + "\n")
        print(f"Written {len(prompts)} prompts to {output_file}")


def main():
    """Main entry point."""

    # Load navigation instructions (sub_goals and landmarks)
    nav_instructions_file = 'ablation_study/test_seen_200_predictions.jsonl'
    print(f"Loading navigation instructions from {nav_instructions_file}...")
    navigation_instructions = load_navigation_instructions(nav_instructions_file)
    print(f"Loaded {len(navigation_instructions)} navigation instructions")

    # Process each method's results
    methods = {
        'json': 'ablation_study/json.jsonl',
        'textual': 'ablation_study/textual.jsonl',
        'graph_vis': 'ablation_study/graph_vis.jsonl',
        'grid': 'ablation_study/grid.jsonl'
    }

    for method_name, results_file in methods.items():
        try:
            process_results_for_method(
                method_name=method_name,
                results_file=results_file,
                navigation_instructions=navigation_instructions,
                output_prefix="step2"  # Output will be step2_{method}.jsonl
            )
        except Exception as e:
            print(f"Error processing method {method_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()