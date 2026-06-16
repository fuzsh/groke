"""
Multi-Format Navigation Agent

This script demonstrates multi-step prompting using all four presentation formats:
1. Matrix representation (existing)
2. Textual/Natural-language representation
3. Structured JSON representation
4. Graphviz-style visual text representation

Follows the design pattern of second_agent.py with next-step generation capabilities.
"""

import json
import math
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

from rapidfuzz import fuzz

from groke.scorer.graph_context import navigate
from groke.scorer.grid_representation import convert2grid
from groke.scorer.presentation_formats import (
    generate_textual_representation,
    generate_json_representation,
    generate_json_representation_dict,
    generate_graphviz_representation,
    generate_all_representations,
    NavigationStepGenerator,
    RepresentationFormat,
    generate_batch_request
)
from groke.data_loader import get_data_by_instruction
from groke.templates import navigator_batch, navigator_batch_multi_format, multi_step_batch


OUTPUT_FILE = 'multi_format_navigator.jsonl'


def _get_heading_to_node(area_links, area_nodes, start_node_id, end_node_id):
    """Calculate heading from start node to end node using BFS pathfinding."""
    graph = {}
    for link in area_links:
        src = link['source']
        tgt = link['target']
        if src not in graph:
            graph[src] = []
        graph[src].append(tgt)

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
        return None, None

    if len(found_path) < 2:
        return found_path, 0

    prev_node_id = found_path[-2]

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


def _build_navigator_context(
    area_data: Dict,
    current_node_id: str,
    current_heading: float,
    pois: dict = None,
    poi_mapping: dict = None,
    previous_visited: List[Dict] = None,
    include_formats: List[str] = None
) -> Dict[str, Any]:
    """
    Build complete navigation context with all representation formats.

    Args:
        area_data: Dict containing area_nodes, area_links, area_pois
        current_node_id: Current position node ID
        current_heading: Current heading in degrees
        pois: Dict mapping landmark name to list of POI node IDs
        poi_mapping: Dict mapping landmark name to letter
        previous_visited: List of previously visited node dicts
        include_formats: List of format names to include

    Returns:
        Dict with path_data and all requested representations
    """
    if include_formats is None:
        include_formats = ["matrix", "textual", "json", "graphviz"]

    if previous_visited is None:
        previous_visited = []

    # Get path data from navigate()
    path_data = navigate(
        map_json=area_data,
        starting_point=current_node_id,
        heading=int(current_heading),
        pois=pois,
        poi_mapping=poi_mapping,
        units=1,
        last_instruction=False
    )

    result = {
        "path_data": path_data,
        "representations": {}
    }

    area_nodes = area_data.get('area_nodes', {})
    area_pois = area_data.get('area_pois', {})

    # Generate matrix representation
    if "matrix" in include_formats:
        matrix = convert2grid(
            path_data,
            previous_visited,
            area_nodes=area_nodes,
            area_pois=area_pois,
            pois=pois,
            poi_mapping=poi_mapping
        ).tolist()
        result["representations"]["matrix"] = matrix
        result["matrix_str"] = "\n".join(str(row) for row in matrix)

    # Generate textual representation
    if "textual" in include_formats:
        textual = generate_textual_representation(
            path_data,
            previous_visited,
            poi_mapping,
            current_heading
        )
        result["representations"]["textual"] = textual

    # Generate JSON representation
    if "json" in include_formats:
        json_repr = generate_json_representation(
            path_data,
            previous_visited,
            poi_mapping,
            current_heading,
            area_nodes,
            area_pois,
            include_coordinates=True
        )
        result["representations"]["json"] = json_repr

    # Generate Graphviz representation
    if "graphviz" in include_formats:
        graphviz = generate_graphviz_representation(
            path_data,
            previous_visited,
            poi_mapping,
            current_heading
        )
        result["representations"]["graphviz"] = graphviz

    return result


def _find_available_pois(area_pois, landmarks):
    """Find POIs in area that match the given landmarks using fuzzy matching."""
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
    """Generate unique single-letter identifiers for landmarks."""
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


def _get_current_instruction_poi(sub_instruction_text, landmarks_in_area):
    """Extract POIs mentioned in the current sub-instruction."""
    current_sub_instructions_landmarks = {}

    for lia_name, lia_node_id in landmarks_in_area.items():
        if lia_name.lower() in sub_instruction_text.lower():
            current_sub_instructions_landmarks[lia_name] = lia_node_id

    return current_sub_instructions_landmarks


class MultiFormatNavigationProcessor:
    """
    Processor for multi-step navigation with multiple representation formats.

    This class follows the design pattern of second_agent.py but extends it
    to support all four representation formats and next-step generation.
    """

    def __init__(
        self,
        instructions_file: str = 'predictions/test_seen_processed.jsonl',
        output_file: str = 'multi_format_navigator.jsonl',
        split_file: str = "test_seen.json",
        base_path: str = './data/map2seq/',
        neighbor_degrees: int = 20
    ):
        """
        Initialize the processor.

        Args:
            instructions_file: Path to processed instructions JSONL
            output_file: Path for output batch requests
            split_file: Data split file name
            base_path: Base path for map data
            neighbor_degrees: Degrees of neighbors to include
        """
        self.instructions_file = instructions_file
        self.output_file = output_file
        self.split_file = split_file
        self.base_path = base_path
        self.neighbor_degrees = neighbor_degrees

        self.navigation_instructions = []
        self._load_instructions()

    def _load_instructions(self):
        """Load navigation instructions from JSONL file."""
        try:
            with open(self.instructions_file, 'r') as f:
                for line in f:
                    self.navigation_instructions.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: Instructions file {self.instructions_file} not found")

    def process_single_instruction(
        self,
        instruction_data: Dict,
        representation_format: str = "all"
    ) -> List[Dict]:
        """
        Process a single navigation instruction and generate batch requests.

        Args:
            instruction_data: Dict with instruction details
            representation_format: Format to use ("all", "matrix", "textual", "json", "graphviz")

        Returns:
            List of batch request dicts
        """
        instruction_id = int(instruction_data['key'])
        sub_goals = instruction_data.get('sub_goals', [])
        landmarks = instruction_data.get('landmarks', [])

        # Load area data
        area_data = get_data_by_instruction(
            instruction_id,
            self.split_file,
            base_path=self.base_path,
            neighbor_degrees=self.neighbor_degrees
        )

        instruction_info = area_data.get("instruction_data", {})
        navigation_instruction = instruction_info.get("instructions", "")

        area_nodes = area_data.get("area_nodes", {})
        area_links = area_data.get("area_links", [])
        area_pois = area_data.get("area_pois", {})

        # Find available POIs
        landmarks_in_area = _find_available_pois(area_pois, landmarks)

        # Get route information
        route = instruction_info.get("route", {})
        osm_path = route.get("osm_path", [])

        if not osm_path:
            return []

        # Initialize navigation state
        current_node_id = osm_path[0]
        current_heading = float(route.get("initial_heading", 0))
        previous_visited_path = []

        requests = []

        # Process each sub-goal
        for sub_idx, current_sub in enumerate(sub_goals):
            sub_instruction_text = current_sub.get("description", "")

            # Get landmarks for this sub-instruction
            sub_instruction_landmarks = _get_current_instruction_poi(
                sub_instruction_text, landmarks_in_area
            )

            unique_identifiers_mapping = None
            landmarks_mapping = ""
            landmark_legend = ""

            if sub_instruction_landmarks:
                unique_identifiers_mapping = _get_unique_identifiers(sub_instruction_landmarks)
                landmarks_mapping = "Landmark: " + ", ".join(unique_identifiers_mapping.keys())
                landmark_legend = ", " + ", ".join(
                    f"{char} = {name} (landmark)" for name, char in unique_identifiers_mapping.items()
                )

            # Build context with all formats
            context = _build_navigator_context(
                area_data=area_data,
                current_node_id=current_node_id,
                current_heading=current_heading,
                pois=sub_instruction_landmarks,
                poi_mapping=unique_identifiers_mapping,
                previous_visited=previous_visited_path,
                include_formats=["matrix", "textual", "json", "graphviz"]
            )

            # Get position markers
            matrix = context["representations"]["matrix"]
            locations: Dict[str, Optional[Tuple[str, str]]] = {"S": None, "P": None}

            for r, row in enumerate(matrix):
                for c, val in enumerate(row):
                    if val in locations:
                        locations[val] = (str(r), str(c))

            # Determine position string
            if sub_idx == 0:
                location_legend = "S = Current Position"
                your_position = f"[{locations['S'][0]}, {locations['S'][1]}] (marked as S)" if locations['S'] else "Unknown"
            else:
                location_legend = "S = Start Position, P = Current Position"
                if locations['P']:
                    your_position = f"[{locations['P'][0]}, {locations['P'][1]}] (marked as P)"
                    if locations['S']:
                        your_position += f" and start position [{locations['S'][0]}, {locations['S'][1]}] (marked as S)"
                else:
                    your_position = "Unknown"

            # Build planning state
            planning_state = "\n".join(
                f"{i + 1}. {sub.get('description', '')} "
                f"({'IN_PROGRESS' if i == sub_idx else 'COMPLETED' if i < sub_idx else 'TODO'})"
                for i, sub in enumerate(sub_goals)
            )

            # Generate request based on format
            if representation_format == "all":
                request = navigator_batch_multi_format(
                    key=f"{instruction_data['key']}-{sub_idx}",
                    navigation_instruction=navigation_instruction,
                    navigator_message=sub_instruction_text,
                    current_heading=f"{current_heading:.2f}",
                    location_legend=location_legend,
                    landmarks=landmarks_mapping,
                    landmark_legend=landmark_legend,
                    planning_state=planning_state,
                    your_position=your_position,
                    representations={
                        "matrix": context["matrix_str"],
                        "textual": context["representations"]["textual"],
                        "json": context["representations"]["json"],
                        "graphviz": context["representations"]["graphviz"]
                    }
                )
                # Add all representations to metadata
                request["metadata"] = {
                    "key": instruction_data['key'],
                    "sub_goal_index": sub_idx,
                    "representations": context["representations"]
                }
            else:
                # Single format
                format_map = {
                    "matrix": context["matrix_str"],
                    "textual": context["representations"].get("textual", ""),
                    "json": context["representations"].get("json", ""),
                    "graphviz": context["representations"].get("graphviz", "")
                }
                request = navigator_batch(
                    key=f"{instruction_data['key']}-{sub_idx}",
                    navigation_instruction=navigation_instruction,
                    navigator_message=sub_instruction_text,
                    current_heading=f"{current_heading:.2f}",
                    location_legend=location_legend,
                    landmarks=landmarks_mapping,
                    landmark_legend=landmark_legend,
                    planning_state=planning_state,
                    your_position=your_position,
                    matrix_representation=context["matrix_str"],
                    textual_representation=context["representations"].get("textual"),
                    json_representation=context["representations"].get("json"),
                    graphviz_representation=context["representations"].get("graphviz"),
                    include_formats=[representation_format]
                )

            requests.append(request)

        return requests

    def process_all_instructions(self, representation_format: str = "all") -> List[Dict]:
        """
        Process all loaded instructions.

        Args:
            representation_format: Format to use for all requests

        Returns:
            List of all batch requests
        """
        all_requests = []

        for idx, ni in enumerate(self.navigation_instructions):
            print(f'Processing instruction {idx + 1}/{len(self.navigation_instructions)}')
            try:
                requests = self.process_single_instruction(ni, representation_format)
                all_requests.extend(requests)
            except Exception as e:
                print(f"Error processing instruction {idx}: {e}")

        return all_requests

    def save_requests(self, requests: List[Dict], output_file: str = None):
        """Save requests to JSONL file."""
        if output_file is None:
            output_file = self.output_file

        with open(output_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        print(f"Saved {len(requests)} requests to {output_file}")


class NextStepProcessor:
    """
    Processor for generating next-step prompts from previous step results.

    This follows the multi-step prompting workflow where output from one step
    is processed to generate the next step.
    """

    def __init__(self, area_data: Dict):
        """
        Initialize with area data.

        Args:
            area_data: Dict containing area_nodes, area_links, area_pois
        """
        self.area_data = area_data
        self.step_generator = None

    def initialize_navigation(
        self,
        start_node_id: str,
        initial_heading: float,
        poi_mapping: Dict[str, str] = None,
        pois: Dict[str, List[str]] = None,
        representation_format: RepresentationFormat = RepresentationFormat.MATRIX
    ):
        """
        Initialize navigation session.

        Args:
            start_node_id: Starting node ID
            initial_heading: Initial heading in degrees
            poi_mapping: Dict mapping landmark name to letter
            pois: Dict mapping landmark name to list of POI node IDs
            representation_format: Which format to use for context generation
        """
        self.step_generator = NavigationStepGenerator(
            area_data=self.area_data,
            poi_mapping=poi_mapping,
            pois=pois,
            representation_format=representation_format
        )
        self.step_generator.initialize(start_node_id, initial_heading)

    def generate_next_step(
        self,
        previous_result: Dict = None,
        sub_instruction: str = "",
        planning_state: str = "",
        include_all_formats: bool = True
    ) -> Dict[str, Any]:
        """
        Generate context for the next navigation step.

        Args:
            previous_result: Optional result from previous step
                Expected format: {"next_position": [row, col], "status": "...", "heading": ...}
            sub_instruction: Current sub-instruction text
            planning_state: Current planning state string
            include_all_formats: Whether to include all representation formats

        Returns:
            Dict with step context and all representations
        """
        if self.step_generator is None:
            raise ValueError("Navigation not initialized. Call initialize_navigation first.")

        # Process previous result if provided
        if previous_result:
            next_node_id = previous_result.get("next_node_id")
            new_heading = previous_result.get("heading")
            status = previous_result.get("status", "IN_PROGRESS")

            if next_node_id:
                self.step_generator.process_step_result(
                    next_node_id=next_node_id,
                    new_heading=new_heading,
                    sub_plan_status=status
                )

        # Generate context for current step
        return self.step_generator.generate_step_context(
            sub_instruction=sub_instruction,
            planning_state=planning_state,
            include_all_formats=include_all_formats
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get navigation summary."""
        if self.step_generator:
            return self.step_generator.get_navigation_summary()
        return {}


def main():
    """Main entry point for processing navigation instructions."""
    processor = MultiFormatNavigationProcessor()

    # Process all instructions with all formats
    print("Processing instructions with all representation formats...")
    requests = processor.process_all_instructions(representation_format="all")

    # Save to output file
    processor.save_requests(requests)

    # Also save format-specific files
    for format_name in ["matrix", "textual", "json", "graphviz"]:
        print(f"\nProcessing with {format_name} format only...")
        requests = processor.process_all_instructions(representation_format=format_name)
        processor.save_requests(requests, f"navigator_{format_name}.jsonl")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
