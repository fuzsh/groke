"""
Graph context builder for the Navigator Agent.
Provides functions to traverse the graph and build context for LLM navigation decisions.
Uses the toon_format library for compact, LLM-friendly serialization.
"""

import json
import math
from typing import Dict, List, Optional, Any, Tuple

from toon_format import encode, decode


def calculate_heading_from_coords(
        from_lat: float, from_lng: float,
        to_lat: float, to_lng: float
) -> float:
    """
    Calculate compass heading (0-360 degrees) from one coordinate to another.
    Uses the forward azimuth formula.
    """
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


def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the Haversine distance between two points in meters.

    Args:
        lat1, lng1: First point coordinates
        lat2, lng2: Second point coordinates

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def find_nearest_path_nodes_for_pois(
        pois: Dict[str, List[str]],
        poi_mapping: Dict[str, str],
        area_nodes: Dict,
        area_pois: Dict,
        path_node_ids: set,
        max_distance: float = 50.0
) -> Dict[str, List[Tuple[str, str, float, float]]]:
    """
    Find the nearest path node for each POI based on lat/lng distance.

    Args:
        pois: Dict mapping landmark name to list of POI node IDs
        poi_mapping: Dict mapping landmark name to assigned letter
        area_nodes: All nodes in the area with lat/lng
        area_pois: POI data with lat/lng
        path_node_ids: Set of node IDs that are on the path
        max_distance: Maximum distance in meters to consider (default 50m)

    Returns:
        Dict mapping path_node_id -> [(landmark_name, letter, poi_node_id, distance)]
    """
    path_node_to_pois = {}

    for landmark_name, poi_node_ids in pois.items():
        letter = poi_mapping.get(landmark_name, '?')

        for poi_node_id in poi_node_ids:
            # Get POI coordinates
            poi_coords = None
            if poi_node_id in area_pois:
                poi_coords = area_pois[poi_node_id]
            elif poi_node_id in area_nodes:
                poi_coords = area_nodes[poi_node_id]

            if not poi_coords:
                continue

            poi_lat = poi_coords.get('lat')
            poi_lng = poi_coords.get('lng')

            if poi_lat is None or poi_lng is None:
                continue

            # Find nearest path node
            min_distance = float('inf')
            nearest_node = None

            for path_node_id in path_node_ids:
                if path_node_id not in area_nodes:
                    continue

                path_node = area_nodes[path_node_id]
                path_lat = path_node.get('lat')
                path_lng = path_node.get('lng')

                if path_lat is None or path_lng is None:
                    continue

                distance = calculate_distance(poi_lat, poi_lng, path_lat, path_lng)

                if distance < min_distance:
                    min_distance = distance
                    nearest_node = path_node_id

            # If nearest node is within max_distance, associate POI with it
            if nearest_node and min_distance <= max_distance:
                if nearest_node not in path_node_to_pois:
                    path_node_to_pois[nearest_node] = []
                path_node_to_pois[nearest_node].append(
                    (landmark_name, letter, poi_node_id, min_distance)
                )

    return path_node_to_pois


def navigate(
        map_json: Dict,
        starting_point: str,
        heading: int,
        pois: Dict[str, List[str]] = None,
        poi_mapping: Dict[str, str] = None,
        units: int = 1,
        last_instruction: bool = False
) -> List[Dict]:
    """
    Given a map_json representing a trajectory map, return a sequence of nodes
    in front of the starting point based on the heading.

    Args:
        map_json: Dictionary containing area_links, area_pois, area_nodes
        starting_point: Starting node ID
        heading: Initial heading in degrees (0-360)
        pois: Dict mapping landmark name to list of POI node IDs
        poi_mapping: Dict mapping landmark name to assigned letter
        units: Number of intersections to pass
        last_instruction: Whether this is the last instruction

    Returns:
        List of node step information dictionaries
    """
    # --- 1. Parse Data ---
    area_links = map_json.get('area_links', [])
    area_pois = map_json.get('area_pois', {})
    area_nodes = map_json.get('area_nodes', {})

    # Initialize POI structures
    if pois is None:
        pois = {}
    if poi_mapping is None:
        poi_mapping = {}

    # Build Adjacency List
    adjacency = {}
    node_connections = {}

    for link in area_links:
        src = link['source']
        tgt = link['target']

        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append({'target': tgt})

        # Track connections for intersection detection (degree check)
        node_connections[src] = node_connections.get(src, set())
        node_connections[src].add(tgt)
        node_connections[tgt] = node_connections.get(tgt, set())
        node_connections[tgt].add(src)

    def get_link_heading(from_node_id: str, to_node_id: str) -> float:
        """Calculate heading from one node to another using coordinates."""
        if from_node_id not in area_nodes or to_node_id not in area_nodes:
            return 0.0
        from_node = area_nodes[from_node_id]
        to_node = area_nodes[to_node_id]
        return calculate_heading_from_coords(
            from_node['lat'], from_node['lng'],
            to_node['lat'], to_node['lng']
        )

    # --- 2. Helper Functions ---

    def get_heading_diff(h1, h2):
        """Calculates minimal difference between two headings (0-360)."""
        diff = abs(h1 - h2) % 360
        return min(diff, 360 - diff)

    def get_relative_direction(current_heading, target_heading):
        """
        Returns relative direction (Forward, Left, Right, Back)
        based on the change in heading.
        """
        diff = (target_heading - current_heading + 180) % 360 - 180
        if -45 <= diff <= 45:
            return "Forward"
        elif -135 <= diff < -45:
            return "Left"
        elif 45 < diff <= 135:
            return "Right"
        else:
            return "Back"

    def is_intersection(node_id):
        """
        Determines if a node is an intersection (connects to > 2 unique nodes).
        """
        return len(node_connections.get(node_id, [])) > 2

    def get_connectivity_and_next(current_node, current_heading):
        """
        Returns:
        1. best_match: Dict with target and calculated heading for the best forward path.
        2. connections: A list of all available choices with relative directions.
        3. side_pois: Empty list (POIs handled separately now)
        """
        if current_node not in adjacency:
            return None, [], []

        candidates = adjacency[current_node]

        # 1. Identify best forward match using coordinate-based headings
        best_match = None
        forward_links = []

        for link in candidates:
            tgt = link['target']
            link_heading = get_link_heading(current_node, tgt)
            diff = get_heading_diff(link_heading, current_heading)
            if diff < 100:  # Forward tolerance
                forward_links.append((diff, {'target': tgt, 'heading': link_heading}))

        if forward_links:
            forward_links.sort(key=lambda x: x[0])
            best_match = forward_links[0][1]

        # 2. Build Connectivity
        connections = []

        for link in candidates:
            tgt = link['target']
            h = get_link_heading(current_node, tgt)
            rel_dir = get_relative_direction(current_heading, h)
            if rel_dir == "Back":
                continue

            connections.append({
                "node_id": tgt,
                "heading": h,
                "direction": rel_dir
            })

        return best_match, connections, []

    # --- 3. First Pass: Collect all nodes on the path ---

    path_node_ids = set()
    current_node = starting_point
    current_heading = heading

    MAX_ITERATIONS = 1000
    intersections_passed = 0
    target_units = units if units is not None else 1

    for _ in range(MAX_ITERATIONS):
        path_node_ids.add(current_node)

        is_inter = is_intersection(current_node)
        if is_inter:
            intersections_passed += 1
            if intersections_passed >= target_units:
                # Get a few more nodes after last intersection
                temp_node = current_node
                temp_heading = current_heading
                for i in range(3):
                    if temp_node not in adjacency:
                        break
                    best_match, _, _ = get_connectivity_and_next(temp_node, temp_heading)
                    if not best_match:
                        break
                    path_node_ids.add(best_match['target'])
                    temp_node = best_match['target']
                    temp_heading = best_match['heading']
                break

        best_match, _, _ = get_connectivity_and_next(current_node, current_heading)
        if not best_match:
            break

        current_node = best_match['target']
        current_heading = best_match['heading']

    # --- 4. Find nearest path nodes for each POI ---

    path_node_to_pois = find_nearest_path_nodes_for_pois(
        pois, poi_mapping, area_nodes, area_pois, path_node_ids, max_distance=50.0
    )

    # --- 5. Second Pass: Build path data with POI information ---

    path_data = []
    current_node = starting_point
    current_heading = heading

    intersections_passed = 0
    nodes_after_last_intersection = 0
    stop_traversal = False

    for _ in range(MAX_ITERATIONS):
        is_inter = is_intersection(current_node)

        # Get connectivity & forward path
        best_match, connections, _ = get_connectivity_and_next(current_node, current_heading)

        # Check if there are POIs near this node
        nearby_pois = path_node_to_pois.get(current_node, [])

        # Build side_pois list with relative directions
        side_pois = []
        current_poi_info = None

        for landmark_name, letter, poi_node_id, distance in nearby_pois:
            # Get POI coordinates
            poi_coords = area_pois.get(poi_node_id) or area_nodes.get(poi_node_id)
            if not poi_coords:
                continue

            # Calculate heading from current node to POI
            poi_heading = calculate_heading_from_coords(
                area_nodes[current_node]['lat'],
                area_nodes[current_node]['lng'],
                poi_coords['lat'],
                poi_coords['lng']
            )

            poi_direction = get_relative_direction(current_heading, poi_heading)

            # Determine position description
            position = poi_direction.lower()
            if is_inter and poi_direction in ["Left", "Right"]:
                position = "on the corner"

            poi_info = {
                "poi": landmark_name,
                "letter": letter,
                "direction": poi_direction,
                "position": position,
                "node_id": poi_node_id,
                "distance": round(distance, 1)
            }

            # If POI is very close (< 5m), treat it as being at the current node
            if distance < 5.0:
                current_poi_info = poi_info
            else:
                side_pois.append(poi_info)

        # Build Node Step
        node_step = {
            "node_id": current_node,
            "is_intersection": is_inter,
            "connectivity": connections,
            "side_pois": side_pois,
            "direction": current_heading
        }

        # Add POI info if current node has a POI right at it
        if current_poi_info:
            node_step["poi"] = current_poi_info["poi"]
            node_step["poi_letter"] = current_poi_info["letter"]
            node_step["poi_position"] = current_poi_info["position"]

        path_data.append(node_step)

        # Handle Intersection Logic
        if is_inter:
            intersections_passed += 1
            nodes_after_last_intersection = 0

            if intersections_passed >= target_units:
                if last_instruction:
                    stop_traversal = True
        else:
            nodes_after_last_intersection += 1

        # Stopping Conditions
        if stop_traversal:
            break

        if intersections_passed >= target_units and nodes_after_last_intersection >= 3:
            break

        if not best_match:
            break

        # Move to Next Node
        current_node = best_match['target']
        current_heading = best_match['heading']

    return path_data


def encode_node_ids_compact(node_ids: List[str], prefix: str = "...") -> str:
    """
    Encode a list of node IDs into a compact format using toon_format.

    Args:
        node_ids: List of node ID strings
        prefix: Prefix to indicate truncation (default: "...")

    Returns:
        Compact TOON-encoded string representation
    """
    if not node_ids:
        return encode({"nodes": []})

    # Create compact representation with prefix and list of IDs
    compact_data = {
        "in_front": prefix + encode(node_ids)
    }
    return encode(compact_data)


def build_navigation_context(
        map_json: Dict,
        current_node_id: str,
        current_heading: float,
        previous_node_id: str = None,
        poi_links: List[Dict] = None,
        max_depth: int = 3
) -> Dict:
    """
    Build a comprehensive navigation context for the current position.

    Args:
        map_json: Dictionary containing area_links, area_pois, area_nodes
        current_node_id: Current node ID
        current_heading: Current heading in degrees
        previous_node_id: Previous node ID (to exclude from forward options)
        poi_links: Optional list of POI links for detailed POI lookup
        max_depth: Maximum depth of nodes to include

    Returns:
        Dictionary with navigation context
    """
    area_links = map_json.get('area_links', [])
    area_nodes = map_json.get('area_nodes', {})
    area_pois = map_json.get('area_pois', {})

    # Build adjacency - just store targets, we'll calculate headings from coordinates
    adjacency = {}
    for link in area_links:
        src = link['source']
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append({
            'target': link['target']
        })

    def calc_heading_to_node(from_node_id: str, to_node_id: str) -> float:
        """Calculate heading from one node to another using coordinates."""
        if from_node_id not in area_nodes or to_node_id not in area_nodes:
            return 0.0
        from_node = area_nodes[from_node_id]
        to_node = area_nodes[to_node_id]
        return calculate_heading_from_coords(
            from_node['lat'], from_node['lng'],
            to_node['lat'], to_node['lng']
        )

    def get_relative_direction(from_heading, to_heading):
        diff = (to_heading - from_heading + 180) % 360 - 180
        if -45 <= diff <= 45:
            return "forward"
        elif -135 <= diff < -45:
            return "left"
        elif 45 < diff <= 135:
            return "right"
        else:
            return "back"

    def get_node_pois(node_id):
        """Get POIs for a node, trying both area_pois and poi_links."""
        pois = []

        # Check if node_id is directly in area_pois
        if node_id in area_pois:
            poi_data = area_pois[node_id]
            tags = poi_data.get('tags', {})
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except json.JSONDecodeError:
                    tags = {}
            pois.append({
                'poi_id': node_id,
                'name': tags.get('name', tags.get('amenity', tags.get('shop', 'unnamed'))),
                'tags': tags
            })

        # Check poi_links if available
        if poi_links:
            for poi_link in poi_links:
                if poi_link.get('osm_id') == node_id:
                    poi_id = poi_link.get('poi_id')
                    if poi_id in area_pois:
                        poi_data = area_pois[poi_id]
                        tags = poi_data.get('tags', {})
                        if isinstance(tags, str):
                            try:
                                tags = json.loads(tags)
                            except json.JSONDecodeError:
                                tags = {}
                        pois.append({
                            'poi_id': poi_id,
                            'name': tags.get('name', tags.get('amenity', tags.get('shop', 'unnamed'))),
                            'tags': tags
                        })

        return pois

    # Get current node info
    current_node_info = area_nodes.get(current_node_id, {})
    current_pois = get_node_pois(current_node_id)

    # Get available directions
    available_directions = []
    if current_node_id in adjacency:
        for link in adjacency[current_node_id]:
            target = link['target']
            heading = calc_heading_to_node(current_node_id, target)

            # Skip going back to previous node
            if target == previous_node_id:
                continue

            rel_dir = get_relative_direction(current_heading, heading)
            if rel_dir == "back":
                continue

            target_info = area_nodes.get(target, {})
            target_pois = get_node_pois(target)

            # Count connections at target (is it an intersection?)
            target_connections = sum(1 for l in area_links if l['source'] == target or l['target'] == target)

            available_directions.append({
                'node_id': target,
                'heading': heading,
                'relative_direction': rel_dir,
                'lat': target_info.get('lat'),
                'lng': target_info.get('lng'),
                'is_intersection': target_connections > 2,
                'pois': target_pois
            })

    # Count current node connections
    current_connections = sum(1 for l in area_links if l['source'] == current_node_id or l['target'] == current_node_id)

    # Extract in-front node IDs and encode compactly using toon_format
    in_front_node_ids = [d['node_id'] for d in available_directions]
    in_front_encoded = encode_node_ids_compact(in_front_node_ids)

    return {
        'current_node': {
            'node_id': current_node_id,
            'lat': current_node_info.get('lat'),
            'lng': current_node_info.get('lng'),
            'heading': current_heading,
            'is_intersection': current_connections > 2,
            'pois': current_pois
        },
        'available_directions': available_directions,
        'in_front_nodes_encoded': in_front_encoded
    }


def encode_navigation_context(path_data: List[Dict]) -> str:
    """
    Encode navigation path data into compact TOON format for LLM consumption.
    Uses the toon_format library for efficient serialization.

    Args:
        path_data: List of node step dictionaries from navigate()

    Returns:
        TOON-formatted string representation
    """
    if not path_data:
        return encode({"error": "No path data available"})

    # Structure the data for optimal TOON encoding
    context = {
        "steps": path_data
    }
    return encode(context)


def decode_navigation_context(encoded_str: str) -> List[Dict]:
    """
    Decode a TOON-formatted navigation context string back to path data.

    Args:
        encoded_str: TOON-encoded navigation context string

    Returns:
        List of node step dictionaries
    """
    decoded = decode(encoded_str)
    return decoded.get("steps", [])


def encode_for_decision(
        current_step: Dict,
        sub_instruction: str,
        mentioned_landmarks: List[str]
) -> str:
    """
    Encode a single navigation step for LLM decision making using TOON format.

    Args:
        current_step: Single step dictionary from navigate()
        sub_instruction: The current sub-instruction text
        mentioned_landmarks: List of landmarks mentioned in the instruction

    Returns:
        TOON-formatted context string for the LLM
    """
    decision_context = {
        "current_node": current_step.get('node_id', 'unknown'),
        "heading": current_step.get('direction', 0),
        "is_intersection": current_step.get('is_intersection', False),
        "poi": current_step.get('poi'),
        "poi_letter": current_step.get('poi_letter'),
        "side_pois": current_step.get('side_pois', []),
        "available_paths": current_step.get('connectivity', []),
        "instruction": sub_instruction,
        "landmarks_to_find": mentioned_landmarks
    }

    return encode(decision_context)