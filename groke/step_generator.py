"""
Unified Step Generation Script.

This script reads from test_seen.json and generates next step prompts for all
presentation methods (json, textual, grid, graph_vis) that are still in progress.

Usage:
    python step_generator.py --input test_seen.json --output-dir ablation_study --step 2

The script:
1. Loads state from test_seen.json
2. For each instruction that has methods with status="PROGRESS":
   - Calculates previous path (if predicated_path > 1 and instruction_index > 0)
   - Generates the next step prompt with proper context
3. Outputs prompt files for each method: step{N}_{method}.jsonl
"""

import argparse
import json
import math
import os
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

from rapidfuzz import fuzz

from groke.prompts import (
    GRID_NAVIGATOR_PROMPT,
    TEXTUAL_NAVIGATOR_PROMPT,
    JSON_NAVIGATOR_PROMPT,
    GRAPH_VIS_NAVIGATOR_PROMPT
)
from groke.scorer.graph_context import navigate
from groke.scorer.grid_representation import convert2grid, get_node_id_from_position
from groke.scorer.presentation_formats import generate_all_representations
from groke.data_loader import get_data_by_instruction
from groke.templates import navigator_batch


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
            f"{tags.get('shop', '').replace('_', ' ')} "
            f"{tags.get('highway', '').replace('_', ' ')} "
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


def _get_current_instruction_poi(sub_instruction_text: str, landmarks_in_area: Dict) -> Dict[str, List[str]]:
    """Find landmarks mentioned in the current sub-instruction."""
    current_sub_instructions_landmarks = {}

    for lia_name, lia_node_id in landmarks_in_area.items():
        if lia_name.lower() in sub_instruction_text.lower():
            current_sub_instructions_landmarks[lia_name] = lia_node_id

    return current_sub_instructions_landmarks


def _get_heading_to_node(
        area_links: List[Dict],
        area_nodes: Dict,
        start_node_id: str,
        end_node_id: str
) -> Tuple[List[str], float]:
    """
    Find the path between two nodes and calculate the heading at the end.

    Returns:
        Tuple of (path_node_ids, final_heading)
    """
    # Build adjacency list
    graph = {}
    for link in area_links:
        src = link['source']
        tgt = link['target']
        if src not in graph:
            graph[src] = []
        graph[src].append(tgt)

    # BFS to find path
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
        raise ValueError(f"No path found between {start_node_id} and {end_node_id}")

    if len(found_path) < 2:
        raise ValueError("Start and End nodes are the same")

    # Calculate heading from second-to-last to last node
    prev_node_id = found_path[-2]
    lat1 = math.radians(area_nodes[prev_node_id]['lat'])
    lon1 = math.radians(area_nodes[prev_node_id]['lng'])
    lat2 = math.radians(area_nodes[end_node_id]['lat'])
    lon2 = math.radians(area_nodes[end_node_id]['lng'])

    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    bearing = math.degrees(math.atan2(y, x))
    compass_bearing = (bearing + 360) % 360

    return found_path, compass_bearing


def _get_previous_nodes(path_data: List[Dict], previous_visited: List[str]) -> List[Dict]:
    """
    Filter path_data to only include previously visited nodes with their connections.

    Args:
        path_data: Full path data from navigate()
        previous_visited: List of node IDs that were visited (the route)

    Returns:
        Filtered path_data containing only visited nodes with filtered connectivity
    """
    new_path_data = []

    for pd in path_data:
        if pd['node_id'] not in previous_visited:
            continue

        # Make a copy to avoid modifying the original
        pd_copy = pd.copy()

        new_connectivity = []
        connectivity = pd.get('connectivity', [])
        if connectivity:
            for con in connectivity:
                if con['node_id'] in previous_visited:
                    new_connectivity.append(con)
        pd_copy['connectivity'] = new_connectivity
        new_path_data.append(pd_copy)

    return new_path_data


def _build_previous_visited_path_data(
        area_data: Dict,
        predicated_path: List[str],
        initial_heading: float,
        pois: Dict = None,
        poi_mapping: Dict = None
) -> List[Dict]:
    """
    Build the previous_visited list (path_data format) from the predicated_path.

    This function reconstructs the path data for previously visited nodes,
    which is needed to properly render the "Previous Path" in all representations.

    The approach is:
    For each consecutive pair (n_i, n_{i+1}) in predicated_path:
    1. Calculate heading between n_i and n_{i+1}
    2. Call navigate() from n_i to get prev_path_data
    3. Get visited route between n_i and n_{i+1}
    4. Filter using _get_previous_nodes to only include visited nodes
    5. Combine all previous steps, merging connectivity for duplicate nodes

    Args:
        area_data: Full area data dict
        predicated_path: List of node_ids that have been visited (including current)
        initial_heading: Initial heading from the start position
        pois: Optional POI dict for landmarks
        poi_mapping: Optional POI letter mapping

    Returns:
        List of path_data dicts for previous nodes (excluding current position)
    """
    if len(predicated_path) <= 1:
        return []

    area_nodes = area_data.get('area_nodes', {})
    area_links = area_data.get('area_links', [])

    # Use a dict to accumulate nodes and merge connectivity
    nodes_by_id: Dict[str, Dict] = {}
    current_heading = initial_heading

    # Process each consecutive pair in the predicated path
    for i in range(len(predicated_path) - 1):
        from_node = predicated_path[i]
        to_node = predicated_path[i + 1]

        # Get the visited route and heading between this pair
        try:
            visited_route, new_heading = _get_heading_to_node(
                area_links, area_nodes, from_node, to_node
            )
        except ValueError:
            # If no path found, skip this segment
            continue

        if not visited_route:
            continue

        # Calculate the heading FROM from_node TO the first step in visited_route
        # This ensures navigate() explores in the right direction
        if len(visited_route) >= 2:
            first_step = visited_route[1]  # First node after from_node
            if from_node in area_nodes and first_step in area_nodes:
                from_coords = area_nodes[from_node]
                to_coords = area_nodes[first_step]

                lat1 = math.radians(from_coords['lat'])
                lon1 = math.radians(from_coords['lng'])
                lat2 = math.radians(to_coords['lat'])
                lon2 = math.radians(to_coords['lng'])

                dLon = lon2 - lon1
                y = math.sin(dLon) * math.cos(lat2)
                x = math.cos(lat1) * math.sin(lat2) - \
                    math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

                segment_heading = math.degrees(math.atan2(y, x))
                segment_heading = (segment_heading + 360) % 360
            else:
                segment_heading = current_heading
        else:
            segment_heading = current_heading

        # Get path data from this segment's start node using heading towards next node
        prev_path_data = navigate(
            map_json=area_data,
            starting_point=from_node,
            heading=int(segment_heading),
            pois=pois,
            poi_mapping=poi_mapping,
            units=1,
            last_instruction=False
        )

        # Filter to only include nodes in the visited route for this segment
        segment_visited = _get_previous_nodes(prev_path_data, visited_route)

        # Merge into accumulated dict
        for node_data in segment_visited:
            node_id = node_data['node_id']
            if node_id not in nodes_by_id:
                # New node - add it
                nodes_by_id[node_id] = node_data.copy()
            else:
                # Existing node - merge connectivity
                existing_connectivity = nodes_by_id[node_id].get('connectivity', [])
                new_connectivity = node_data.get('connectivity', [])

                # Add new connections that don't already exist
                existing_conn_ids = {c['node_id'] for c in existing_connectivity}
                for conn in new_connectivity:
                    if conn['node_id'] not in existing_conn_ids:
                        existing_connectivity.append(conn)

                nodes_by_id[node_id]['connectivity'] = existing_connectivity

        # Update heading for next segment
        current_heading = new_heading

    # Build ordered list by following the actual BFS paths
    # We need to maintain the order: start → [BFS path nodes] → waypoint1 → [BFS path nodes] → waypoint2 → ...
    all_previous_visited = []
    seen_ids = set()

    # Re-process each segment to get nodes in correct order
    current_heading = initial_heading
    for i in range(len(predicated_path) - 1):
        from_node = predicated_path[i]
        to_node = predicated_path[i + 1]

        try:
            visited_route, new_heading = _get_heading_to_node(
                area_links, area_nodes, from_node, to_node
            )
        except ValueError:
            continue

        if not visited_route:
            continue

        # Add nodes in BFS path order (excluding the last one which will be the start of next segment)
        # For the last segment, include all nodes except the final current position
        nodes_to_add = visited_route[:-1] if i < len(predicated_path) - 2 else visited_route[:-1]

        for node_id in nodes_to_add:
            if node_id in nodes_by_id and node_id not in seen_ids:
                all_previous_visited.append(nodes_by_id[node_id])
                seen_ids.add(node_id)

        current_heading = new_heading

    return all_previous_visited


def resolve_grid_next_place(
        next_place: Any,
        area_data: Dict,
        current_node_id: str,
        current_heading: float,
        previous_visited: List[Dict],
        pois: Dict = None,
        poi_mapping: Dict = None
) -> Optional[str]:
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
        return get_node_id_from_position(
            path_data,
            previous_visited,
            next_place,
            area_nodes=area_data.get('area_nodes'),
            area_pois=area_data.get('area_pois'),
            pois=pois,
            poi_mapping=poi_mapping
        )

    return None


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def generate_prompts_for_instruction(
        key: str,
        instruction_data: Dict,
        area_data: Dict,
        current_node_id: str,
        current_heading: float,
        current_sub_idx: int,
        n_retry_instruction: int,
        previous_visited: List[Dict],
        sub_goals: List[Dict],
        landmarks: List[Dict],
        target_methods: List[str] = None
) -> Dict[str, Dict]:
    """
    Generate prompts for the next navigation step.

    Args:
        key: Instruction ID
        instruction_data: Instruction metadata
        area_data: Full area data dict
        current_node_id: Current position node ID
        current_heading: Current heading in degrees
        current_sub_idx: Current sub-instruction index
        n_retry_instruction: Retry count
        previous_visited: List of previously visited path_data dicts
        sub_goals: List of sub-goals/instructions
        landmarks: List of landmark dicts
        target_methods: Which methods to generate prompts for (default: all)

    Returns:
        Dict mapping method_name -> prompt_dict
    """
    if target_methods is None:
        target_methods = ['json', 'textual', 'grid', 'graph_vis']

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

    # Build planning state
    def get_sub_description(sub):
        if isinstance(sub, str):
            return sub
        return sub.get('description', '') if isinstance(sub, dict) else str(sub)

    planning_state = "\n".join(
        f"{i + 1}. {get_sub_description(sub)} "
        f"({f'IN_PROGRESS, Iteration: {n_retry_instruction+1}' if i == current_sub_idx else 'COMPLETED' if i < current_sub_idx else 'TODO'})"
        for i, sub in enumerate(sub_goals)
    )

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

    # For subsequent steps (when we have previous path), S = start, P = current position
    if len(previous_visited) > 0 and locations['P'] is not None:
        your_position = f"[{locations['P'][0]}, {locations['P'][1]}] (marked as P) and start position [{locations['S'][0]}, {locations['S'][1]}] (marked as S)"
        location_legend = "S = Start Position, P = Current Position"
    else:
        your_position = f"[{locations['S'][0]}, {locations['S'][1]}] (marked as S)" if locations['S'] else "Unknown"
        location_legend = "S = Current Position"

    prompts = {}

    # Generate prompts for each target method
    if 'textual' in target_methods:
        textual_prompt = TEXTUAL_NAVIGATOR_PROMPT.format(
            navigation_instruction=navigation_instruction,
            navigator_message=sub_instruction_text,
            navigation_context=representations['textual'],
            planning_state=planning_state
        )
        prompts['textual'] = navigator_batch(key, textual_prompt, text_next_place)

    if 'grid' in target_methods:
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
        prompts['grid'] = navigator_batch(key, grid_prompt, grid_next_place)

    if 'json' in target_methods:
        json_prompt = JSON_NAVIGATOR_PROMPT.format(
            navigation_instruction=navigation_instruction,
            landmarks=landmarks_mapping,
            navigator_message=sub_instruction_text,
            navigation_context=f'''```json
{json.dumps(representations['json'])}
```''',
            planning_state=planning_state
        )
        prompts['json'] = navigator_batch(key, json_prompt, text_next_place)

    if 'graph_vis' in target_methods:
        graph_vis_prompt = GRAPH_VIS_NAVIGATOR_PROMPT.format(
            navigation_instruction=navigation_instruction,
            navigator_message=sub_instruction_text,
            navigation_context=representations['graphviz'],
            planning_state=planning_state,
        )
        prompts['graph_vis'] = navigator_batch(key, graph_vis_prompt, text_next_place)

    return prompts


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_state_file(
        input_file: str,
        output_dir: str,
        step_number: int,
        split_file: str = "test_seen_200.json",
        target_methods: List[str] = None
) -> Dict[str, int]:
    """
    Process the state file and generate next step prompts.

    Args:
        input_file: Path to test_seen.json state file
        output_dir: Directory to output prompt files
        step_number: Current step number (for output file naming)
        split_file: Data split file to use
        target_methods: Which methods to process (default: all)

    Returns:
        Dict mapping method_name -> count of prompts generated
    """
    if target_methods is None:
        target_methods = ['json', 'textual', 'grid', 'graph_vis']

    # Load state file
    print(f"Loading state from {input_file}...")
    with open(input_file, 'r') as f:
        state_data = json.load(f)
    print(f"Loaded {len(state_data)} instructions")

    # Collect prompts by method
    prompts_by_method = {method: [] for method in target_methods}

    # Process each instruction
    for key, instruction_state in state_data.items():
        sub_goals = instruction_state.get('sub_instructions', [])
        landmarks = instruction_state.get('landmarks', [])

        if not sub_goals:
            print(f"Warning: No sub_goals for key {key}, skipping")
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

        # Process each method that is still in progress
        for method in target_methods:
            method_state = instruction_state.get(method, {})

            # Skip if method is done or not present
            if method_state.get('status') == 'DONE':
                continue

            # Get current state
            predicated_path = method_state.get('predicated_path', [])
            if not predicated_path:
                print(f"Warning: No predicated_path for key {key} method {method}")
                continue

            current_node_id = predicated_path[-1]
            current_heading = method_state.get('current_heading', 0.0)
            current_sub_idx = method_state.get('current_instruction_index', 0)
            num_retry_current_instruction = method_state.get('num_retry_current_instruction', 0)

            # Validate current node exists in area
            if current_node_id not in area_data.get('area_nodes', {}):
                print(f"Warning: Current node {current_node_id} not in area for key {key}, skipping")
                continue

            # Check if we still have sub-goals to process
            if current_sub_idx >= len(sub_goals):
                print(f"Key {key} method {method}: Navigation completed (all sub-goals done)")
                continue

            # Build previous visited path
            # Key insight: previous path exists when predicated_path has > 1 node
            # AND either current_instruction_index > 0 OR we've moved from start
            if len(predicated_path) > 1:
                # Find landmarks for building previous path context
                area_pois = area_data.get("area_pois", {})
                current_sub = sub_goals[current_sub_idx]
                if isinstance(current_sub, str):
                    sub_text = current_sub
                else:
                    sub_text = current_sub.get("description", "")

                landmarks_in_area = _find_available_pois(area_pois, landmarks)
                sub_instruction_landmarks = _get_current_instruction_poi(sub_text, landmarks_in_area)
                poi_mapping = _get_unique_identifiers(sub_instruction_landmarks) if sub_instruction_landmarks else None

                # Get initial heading from instruction state
                initial_heading = instruction_state.get('initial_headings', 0)

                previous_visited = _build_previous_visited_path_data(
                    area_data=area_data,
                    predicated_path=predicated_path,
                    initial_heading=initial_heading,
                    pois=sub_instruction_landmarks,
                    poi_mapping=poi_mapping
                )
            else:
                previous_visited = []

            # Generate prompts for this method only
            try:
                prompts = generate_prompts_for_instruction(
                    key=key,
                    instruction_data=instruction_data,
                    area_data=area_data,
                    current_node_id=current_node_id,
                    current_heading=current_heading,
                    current_sub_idx=current_sub_idx,
                    n_retry_instruction=num_retry_current_instruction,
                    previous_visited=previous_visited,
                    sub_goals=sub_goals,
                    landmarks=landmarks,
                    target_methods=[method]
                )

                if method in prompts:
                    prompts_by_method[method].append(prompts[method])

            except Exception as e:
                print(f"Error generating prompts for key {key} method {method}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Write output files
    os.makedirs(output_dir, exist_ok=True)
    result_counts = {}

    for method, prompts in prompts_by_method.items():
        if prompts:
            output_file = os.path.join(output_dir, f"step{step_number}_{method}.jsonl")
            with open(output_file, 'w') as f:
                for prompt in prompts:
                    f.write(json.dumps(prompt) + "\n")
            print(f"Written {len(prompts)} prompts to {output_file}")
            result_counts[method] = len(prompts)
        else:
            result_counts[method] = 0

    return result_counts


def main(ttype):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate next step prompts from test_seen.json state file"
    )
    parser.add_argument(
        '--input', '-i',
        default=f'main_test_{ttype}.json',
        help='Input state file (default: test_seen.json)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=f'main_prompt_{ttype}',
        help='Output directory for prompt files (default: ablation_study)'
    )
    parser.add_argument(
        '--step', '-s',
        type=int,
        required=True,
        help='Step number for output file naming (e.g., 2 for step2_*.jsonl)'
    )
    parser.add_argument(
        '--split-file',
        default=f'test_{ttype}.json',
        help='Data split file to use (default: test_seen_200.json)'
    )
    parser.add_argument(
        '--methods', '-m',
        nargs='+',
        choices=['json', 'textual', 'grid', 'graph_vis'],
        default=None,
        help='Methods to generate prompts for (default: all)'
    )

    args = parser.parse_args()

    print(f"=== Step Generator ===")
    print(f"Input: {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Step: {args.step}")
    print(f"Methods: {args.methods or 'all'}")
    print()

    result_counts = process_state_file(
        input_file=args.input,
        output_dir=args.output_dir,
        step_number=args.step,
        split_file=args.split_file,
        target_methods=args.methods
    )

    print()
    print("=== Summary ===")
    total = 0
    for method, count in result_counts.items():
        print(f"  {method}: {count} prompts")
        total += count
    print(f"  Total: {total} prompts")


if __name__ == "__main__":
    for i in ['seen', 'unseen']:
        main(i)