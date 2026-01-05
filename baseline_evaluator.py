"""
Baseline Evaluator for Navigation.

This script implements three baseline methods for OSM navigation:
1. Random Walker: Randomly selects outgoing edges at intersections
2. Heuristic Agent: Matches directional keywords to edge angles
3. Action Sampling: Samples actions based on dataset action distribution

Usage:
    python baseline_evaluator.py --input ablation_study/test_seen_100.json --output baseline_results.json
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any

from src.data_loader import get_data_by_instruction


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_heading_from_coords(
        from_lat: float, from_lng: float,
        to_lat: float, to_lng: float
) -> float:
    """Calculate compass heading (0-360 degrees) from one coordinate to another."""
    lat1 = math.radians(from_lat)
    lon1 = math.radians(from_lng)
    lat2 = math.radians(to_lat)
    lon2 = math.radians(to_lng)

    d_lon = lon2 - lon1

    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)

    initial_bearing = math.atan2(y, x)
    initial_bearing_deg = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing_deg + 360) % 360

    return compass_bearing


def get_relative_direction(current_heading: float, target_heading: float) -> str:
    """
    Returns relative direction (forward, left, right, back)
    based on the change in heading.
    """
    diff = (target_heading - current_heading + 180) % 360 - 180
    if -45 <= diff <= 45:
        return "forward"
    elif -135 <= diff < -45:
        return "left"
    elif 45 < diff <= 135:
        return "right"
    else:
        return "back"


def get_heading_diff(h1: float, h2: float) -> float:
    """Calculates minimal difference between two headings (0-360)."""
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)


def build_adjacency(area_links: List[Dict]) -> Dict[str, List[str]]:
    """Build adjacency list from area links."""
    adjacency = defaultdict(list)
    for link in area_links:
        src = link['source']
        tgt = link['target']
        adjacency[src].append(tgt)
    return adjacency


def count_connections(area_links: List[Dict]) -> Dict[str, int]:
    """Count connections for each node (for intersection detection)."""
    node_connections = defaultdict(set)
    for link in area_links:
        src = link['source']
        tgt = link['target']
        node_connections[src].add(tgt)
        node_connections[tgt].add(src)
    return {k: len(v) for k, v in node_connections.items()}


def is_intersection(node_id: str, connection_counts: Dict[str, int]) -> bool:
    """Determines if a node is an intersection (connects to > 2 unique nodes)."""
    return connection_counts.get(node_id, 0) > 2


def get_available_connections(
        current_node: str,
        current_heading: float,
        adjacency: Dict[str, List[str]],
        area_nodes: Dict[str, Dict],
        exclude_back: bool = True
) -> List[Dict]:
    """
    Get available connections from current node with their headings and directions.

    Returns:
        List of dicts with {node_id, heading, direction}
    """
    connections = []

    if current_node not in adjacency:
        return connections

    for target in adjacency[current_node]:
        if current_node not in area_nodes or target not in area_nodes:
            continue

        # Calculate heading to target
        from_coords = area_nodes[current_node]
        to_coords = area_nodes[target]
        target_heading = calculate_heading_from_coords(
            from_coords['lat'], from_coords['lng'],
            to_coords['lat'], to_coords['lng']
        )

        direction = get_relative_direction(current_heading, target_heading)

        # Optionally exclude going back
        if exclude_back and direction == "back":
            continue

        connections.append({
            "node_id": target,
            "heading": target_heading,
            "direction": direction
        })

    return connections


def get_best_forward_connection(
        connections: List[Dict],
        current_heading: float
) -> Optional[Dict]:
    """Get the connection that best matches going forward."""
    forward_candidates = []

    for conn in connections:
        diff = get_heading_diff(conn['heading'], current_heading)
        if diff < 100:  # Forward tolerance
            forward_candidates.append((diff, conn))

    if forward_candidates:
        forward_candidates.sort(key=lambda x: x[0])
        return forward_candidates[0][1]

    return None


# =============================================================================
# BASELINE 1: RANDOM WALKER
# =============================================================================

class RandomWalker:
    """
    Random Walker Baseline.

    Selects random outgoing edges at every intersection.
    This establishes the "chance" level of success.
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

    def navigate(
            self,
            area_data: Dict,
            max_steps: int = 100
    ) -> Dict:
        """
        Navigate through the graph by randomly selecting edges.

        Args:
            area_data: Area data containing nodes, links, instruction data
            max_steps: Maximum number of navigation steps

        Returns:
            Dict with navigation results
        """
        instruction_data = area_data.get("instruction_data", {})
        route = instruction_data.get("route", {})
        osm_path = route.get("osm_path", [])

        if not osm_path:
            return {"status": "ERROR", "predicated_path": [], "error": "No osm_path"}

        # Initialize
        area_nodes = area_data.get('area_nodes', {})
        area_links = area_data.get('area_links', [])
        adjacency = build_adjacency(area_links)
        connection_counts = count_connections(area_links)

        current_node = osm_path[0]
        current_heading = float(route.get("initial_heading", 0))
        predicated_path = [current_node]
        visited = {current_node}

        # Target: navigate similar number of steps as ground truth
        target_steps = len(osm_path)

        for step in range(max_steps):
            # Get available connections (excluding back)
            connections = get_available_connections(
                current_node, current_heading, adjacency, area_nodes, exclude_back=True
            )

            if not connections:
                # Try including back connections
                connections = get_available_connections(
                    current_node, current_heading, adjacency, area_nodes, exclude_back=False
                )

            if not connections:
                break  # Dead end

            # Random selection
            chosen = random.choice(connections)

            current_node = chosen['node_id']
            current_heading = chosen['heading']
            predicated_path.append(current_node)
            visited.add(current_node)

            # Stop after reaching similar path length as ground truth
            if len(predicated_path) >= target_steps:
                break

        return {
            "status": "DONE",
            "predicated_path": predicated_path,
            "num_retry_current_instruction": 0,
            "current_heading": current_heading
        }


# =============================================================================
# BASELINE 2: HEURISTIC AGENT
# =============================================================================

class HeuristicAgent:
    """
    Heuristic Agent Baseline.

    A rule-based agent that matches directional keywords (e.g., "left", "right",
    "straight", "move forward") to edge angles without semantic reasoning.
    This tests how much of the instruction is purely geometric.
    """

    # Direction keywords and their mappings
    DIRECTION_PATTERNS = {
        'left': [
            r'\bturn\s+left\b', r'\bgo\s+left\b', r'\btake\s+a\s+left\b',
            r'\bleft\s+at\b', r'\bleft\s+on\b', r'\bhang\s+a\s+left\b',
            r'\bbear\s+left\b', r'\bveer\s+left\b'
        ],
        'right': [
            r'\bturn\s+right\b', r'\bgo\s+right\b', r'\btake\s+a\s+right\b',
            r'\bright\s+at\b', r'\bright\s+on\b', r'\bhang\s+a\s+right\b',
            r'\bbear\s+right\b', r'\bveer\s+right\b'
        ],
        'forward': [
            r'\bgo\s+straight\b', r'\bcontinue\s+straight\b', r'\bstraight\s+ahead\b',
            r'\bkeep\s+going\b', r'\bwalk\s+forward\b', r'\bmove\s+forward\b',
            r'\bcontinue\s+forward\b', r'\bgo\s+forward\b', r'\bhead\s+straight\b',
            r'\bstraight\b', r'\bcontinue\b', r'\bkeep\s+straight\b',
            r'\bwalk\s+past\b', r'\bpass\b', r'\bcross\b'
        ]
    }

    def __init__(self):
        # Compile regex patterns
        self.compiled_patterns = {}
        for direction, patterns in self.DIRECTION_PATTERNS.items():
            self.compiled_patterns[direction] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def extract_directions(self, instruction: str) -> List[str]:
        """
        Extract directional commands from instruction text.

        Returns:
            List of directions in order of appearance
        """
        directions = []
        instruction_lower = instruction.lower()

        # Find all matches with their positions
        matches = []
        for direction, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(instruction_lower):
                    matches.append((match.start(), direction))

        # Sort by position and extract directions
        matches.sort(key=lambda x: x[0])
        directions = [m[1] for m in matches]

        # If no directions found, default to forward
        if not directions:
            directions = ['forward']

        return directions

    def navigate(
            self,
            area_data: Dict,
            max_steps: int = 100
    ) -> Dict:
        """
        Navigate through the graph using heuristic direction matching.

        Args:
            area_data: Area data containing nodes, links, instruction data
            max_steps: Maximum number of navigation steps

        Returns:
            Dict with navigation results
        """
        instruction_data = area_data.get("instruction_data", {})
        route = instruction_data.get("route", {})
        osm_path = route.get("osm_path", [])
        instruction_text = instruction_data.get("instructions", "")

        if not osm_path:
            return {"status": "ERROR", "predicated_path": [], "error": "No osm_path"}

        # Extract directions from instruction
        directions = self.extract_directions(instruction_text)
        direction_idx = 0

        # Initialize
        area_nodes = area_data.get('area_nodes', {})
        area_links = area_data.get('area_links', [])
        adjacency = build_adjacency(area_links)
        connection_counts = count_connections(area_links)

        current_node = osm_path[0]
        current_heading = float(route.get("initial_heading", 0))
        predicated_path = [current_node]

        # Target steps
        target_steps = len(osm_path)

        for step in range(max_steps):
            # Get available connections
            connections = get_available_connections(
                current_node, current_heading, adjacency, area_nodes, exclude_back=True
            )

            if not connections:
                break  # Dead end

            is_inter = is_intersection(current_node, connection_counts)

            # Get desired direction
            if direction_idx < len(directions):
                desired_direction = directions[direction_idx]
            else:
                desired_direction = 'forward'

            # Find best matching connection
            chosen = None

            if is_inter:
                # At intersection, try to match the desired direction
                for conn in connections:
                    if conn['direction'] == desired_direction:
                        chosen = conn
                        break

                # If found a matching direction at intersection, advance to next direction
                if chosen:
                    direction_idx += 1
                else:
                    # No exact match, choose forward or any available
                    chosen = get_best_forward_connection(connections, current_heading)
                    if not chosen:
                        chosen = connections[0]
            else:
                # Not at intersection, just go forward
                chosen = get_best_forward_connection(connections, current_heading)
                if not chosen and connections:
                    chosen = connections[0]

            if not chosen:
                break

            current_node = chosen['node_id']
            current_heading = chosen['heading']
            predicated_path.append(current_node)

            # Stop after reaching similar path length
            if len(predicated_path) >= target_steps:
                break

        return {
            "status": "DONE",
            "predicated_path": predicated_path,
            "num_retry_current_instruction": 0,
            "current_heading": current_heading
        }


# =============================================================================
# BASELINE 3: ACTION SAMPLING
# =============================================================================

class ActionSampler:
    """
    Action Sampling Baseline.

    Analyzes the statistical properties of the dataset and samples actions
    based on the action distribution. This baseline tests whether the agent
    can succeed by simply following the most common action patterns.
    """

    def __init__(self):
        self.action_distribution = None
        self.intersection_action_dist = None
        self.non_intersection_action_dist = None

    def analyze_dataset(self, test_data: List[Dict], base_path: str = './data/map2seq/', split_file: str = 'test_seen_200.json') -> Dict:
        """
        Analyze the dataset to compute action distribution.

        Args:
            test_data: List of test instructions
            base_path: Base path for data files

        Returns:
            Dict with action statistics
        """
        print("Analyzing dataset for action distribution...")

        # Track actions at intersections vs non-intersections
        intersection_actions = Counter()
        non_intersection_actions = Counter()
        total_actions = Counter()

        # Track path lengths
        path_lengths = []

        for item in test_data:
            instructions_id = item['instructions_id']
            osm_path = item['route']['osm_path']
            initial_heading = item['route']['initial_heading']

            if len(osm_path) < 2:
                continue

            path_lengths.append(len(osm_path))

            try:
                area_data = get_data_by_instruction(
                    instructions_id,
                    split_file,
                    base_path=base_path,
                    neighbor_degrees=20
                )
            except Exception as e:
                continue

            area_nodes = area_data.get('area_nodes', {})
            area_links = area_data.get('area_links', [])
            connection_counts = count_connections(area_links)

            current_heading = float(initial_heading)

            # Analyze each step in the ground truth path
            for i in range(len(osm_path) - 1):
                current_node = osm_path[i]
                next_node = osm_path[i + 1]

                if current_node not in area_nodes or next_node not in area_nodes:
                    continue

                # Calculate heading to next node
                from_coords = area_nodes[current_node]
                to_coords = area_nodes[next_node]
                next_heading = calculate_heading_from_coords(
                    from_coords['lat'], from_coords['lng'],
                    to_coords['lat'], to_coords['lng']
                )

                # Determine action direction
                action = get_relative_direction(current_heading, next_heading)

                # Track by intersection status
                is_inter = is_intersection(current_node, connection_counts)

                if is_inter:
                    intersection_actions[action] += 1
                else:
                    non_intersection_actions[action] += 1

                total_actions[action] += 1
                current_heading = next_heading

        # Convert to probabilities
        total_inter = sum(intersection_actions.values()) or 1
        total_non_inter = sum(non_intersection_actions.values()) or 1
        total_all = sum(total_actions.values()) or 1

        self.intersection_action_dist = {
            action: count / total_inter
            for action, count in intersection_actions.items()
        }

        self.non_intersection_action_dist = {
            action: count / total_non_inter
            for action, count in non_intersection_actions.items()
        }

        self.action_distribution = {
            action: count / total_all
            for action, count in total_actions.items()
        }

        avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 20

        stats = {
            "total_samples": len(test_data),
            "average_path_length": avg_path_length,
            "total_actions": dict(total_actions),
            "intersection_actions": dict(intersection_actions),
            "non_intersection_actions": dict(non_intersection_actions),
            "action_probabilities": self.action_distribution,
            "intersection_probabilities": self.intersection_action_dist,
            "non_intersection_probabilities": self.non_intersection_action_dist
        }

        print(f"  Analyzed {len(test_data)} samples")
        print(f"  Average path length: {avg_path_length:.1f}")
        print(f"  Action distribution: {self.action_distribution}")
        print(f"  Intersection actions: {self.intersection_action_dist}")

        return stats

    def sample_action(self, is_intersection: bool = False) -> str:
        """Sample an action based on learned distribution."""
        if is_intersection and self.intersection_action_dist:
            dist = self.intersection_action_dist
        elif not is_intersection and self.non_intersection_action_dist:
            dist = self.non_intersection_action_dist
        else:
            dist = self.action_distribution or {'forward': 0.7, 'left': 0.15, 'right': 0.15}

        actions = list(dist.keys())
        probs = [dist.get(a, 0) for a in actions]

        # Normalize probabilities
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(actions)] * len(actions)

        return random.choices(actions, weights=probs, k=1)[0]

    def navigate(
            self,
            area_data: Dict,
            max_steps: int = 100
    ) -> Dict:
        """
        Navigate through the graph by sampling actions from learned distribution.

        Args:
            area_data: Area data containing nodes, links, instruction data
            max_steps: Maximum number of navigation steps

        Returns:
            Dict with navigation results
        """
        instruction_data = area_data.get("instruction_data", {})
        route = instruction_data.get("route", {})
        osm_path = route.get("osm_path", [])

        if not osm_path:
            return {"status": "ERROR", "predicated_path": [], "error": "No osm_path"}

        # Initialize
        area_nodes = area_data.get('area_nodes', {})
        area_links = area_data.get('area_links', [])
        adjacency = build_adjacency(area_links)
        connection_counts = count_connections(area_links)

        current_node = osm_path[0]
        current_heading = float(route.get("initial_heading", 0))
        predicated_path = [current_node]

        # Target steps
        target_steps = len(osm_path)

        for step in range(max_steps):
            # Get available connections
            connections = get_available_connections(
                current_node, current_heading, adjacency, area_nodes, exclude_back=True
            )

            if not connections:
                break  # Dead end

            is_inter = is_intersection(current_node, connection_counts)

            # Sample an action based on the distribution
            sampled_action = self.sample_action(is_inter)

            # Find connection matching the sampled action
            chosen = None
            for conn in connections:
                if conn['direction'] == sampled_action:
                    chosen = conn
                    break

            # If no match, try forward, then any available
            if not chosen:
                chosen = get_best_forward_connection(connections, current_heading)
            if not chosen:
                chosen = connections[0]

            current_node = chosen['node_id']
            current_heading = chosen['heading']
            predicated_path.append(current_node)

            # Stop after reaching similar path length
            if len(predicated_path) >= target_steps:
                break

        return {
            "status": "DONE",
            "predicated_path": predicated_path,
            "num_retry_current_instruction": 0,
            "current_heading": current_heading
        }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_baselines(
        input_file: str,
        output_file: str,
        split_file: str = "test_seen_100.json",
        base_path: str = './data/map2seq/',
        seed: int = 42
) -> Dict:
    """
    Evaluate all baseline methods on the test set.

    Args:
        input_file: Path to test data JSON file
        output_file: Path to output results JSON file
        split_file: Name of split file for data loading
        base_path: Base path for data files
        seed: Random seed for reproducibility

    Returns:
        Dict with evaluation results
    """
    random.seed(seed)

    # Load test data
    print(f"Loading test data from {input_file}...")
    with open(input_file, 'r') as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test instructions")

    # Initialize baselines
    random_walker = RandomWalker(seed=seed)
    heuristic_agent = HeuristicAgent()
    action_sampler = ActionSampler()

    # Analyze dataset for action sampling
    action_stats = action_sampler.analyze_dataset(test_data, base_path, split_file)

    # Results container
    results = {}

    # Evaluate each instruction
    print("\nEvaluating baselines...")
    for idx, item in enumerate(test_data):
        if (idx + 1) % 10 == 0:
            print(f"  Processing {idx + 1}/{len(test_data)}...")

        instructions_id = str(item['instructions_id'])
        osm_path = item['route']['osm_path']
        initial_heading = item['route']['initial_heading']
        instruction_text = item.get('instructions', '')

        # Load area data
        try:
            area_data = get_data_by_instruction(
                int(instructions_id),
                split_file,
                base_path=base_path,
                neighbor_degrees=20
            )
        except Exception as e:
            print(f"  Error loading data for {instructions_id}: {e}")
            continue

        if not area_data:
            continue

        # Run each baseline
        random_result = random_walker.navigate(area_data)
        heuristic_result = heuristic_agent.navigate(area_data)
        action_sample_result = action_sampler.navigate(area_data)

        # Store results
        results[instructions_id] = {
            "instruction_id": instructions_id,
            "osm_path": osm_path,
            "initial_headings": initial_heading,
            "full_instructions": instruction_text,
            "random_walker": random_result,
            "heuristic_agent": heuristic_result,
            "action_sampling": action_sample_result
        }

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Print summary statistics
    print_summary(results, action_stats)

    return results


def print_summary(results: Dict, action_stats: Dict):
    """Print summary statistics for the evaluation."""
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal instructions evaluated: {len(results)}")

    print("\n--- Action Distribution (from dataset analysis) ---")
    print(f"Forward: {action_stats['action_probabilities'].get('forward', 0):.1%}")
    print(f"Left: {action_stats['action_probabilities'].get('left', 0):.1%}")
    print(f"Right: {action_stats['action_probabilities'].get('right', 0):.1%}")
    print(f"Back: {action_stats['action_probabilities'].get('back', 0):.1%}")

    print("\n--- Path Length Comparison ---")
    methods = ['random_walker', 'heuristic_agent', 'action_sampling']

    for method in methods:
        path_lengths = []
        gt_lengths = []

        for key, result in results.items():
            method_result = result.get(method, {})
            pred_path = method_result.get('predicated_path', [])
            gt_path = result.get('osm_path', [])

            if pred_path:
                path_lengths.append(len(pred_path))
            if gt_path:
                gt_lengths.append(len(gt_path))

        avg_pred = sum(path_lengths) / len(path_lengths) if path_lengths else 0
        avg_gt = sum(gt_lengths) / len(gt_lengths) if gt_lengths else 0

        print(f"\n{method}:")
        print(f"  Avg predicted path length: {avg_pred:.1f}")
        print(f"  Avg ground truth length: {avg_gt:.1f}")

    print("\n" + "=" * 60)
    print("Results saved. Run evaluation_metrics.py to compute metrics.")
    print("=" * 60)


def main(seen_type):
    parser = argparse.ArgumentParser(description="Evaluate baseline navigation methods")
    parser.add_argument(
        '--input', '-i',
        default=f'data/map2seq/splits/test_{seen_type}.json',
        help='Input test data file (default: ablation_study/test_seen_100.json)'
    )
    parser.add_argument(
        '--output', '-o',
        default=f'baseline_results_{seen_type}.json',
        help='Output results file (default: baseline_results.json)'
    )
    parser.add_argument(
        '--split-file',
        default=f'test_{seen_type}.json',
        help='Split file name for data loading (default: test_seen_200.json)'
    )
    parser.add_argument(
        '--base-path',
        default='./data/map2seq/',
        help='Base path for map2seq data (default: ./data/map2seq/)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    evaluate_baselines(
        input_file=args.input,
        output_file=args.output,
        split_file=args.split_file,
        base_path=args.base_path,
        seed=args.seed
    )


if __name__ == "__main__":
    for seen_type in ['seen', 'unseen']:
        main(seen_type)