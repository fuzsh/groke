"""
Presentation Formats for Grid-Based OSM Data.

This module provides three alternative representation formats for nearby OSM nodes and POIs:
1. Textual / Natural-Language Representation
2. Structured JSON Representation
3. Visual Text (Graphviz-Style) Representation

These representations complement the existing matrix_representation and are designed
for multi-step prompting workflows.
"""

import math
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class RepresentationFormat(Enum):
    """Enumeration of available representation formats."""
    TEXTUAL = "textual"
    JSON = "json"
    GRAPHVIZ = "graphviz"
    MATRIX = "matrix"


@dataclass
class NodeConnection:
    """Represents a connection between two nodes."""
    from_node: str
    to_node: str
    heading: float
    direction: str
    is_poi: bool = False
    poi_name: Optional[str] = None
    poi_type: Optional[str] = None
    distance: Optional[float] = None


@dataclass
class BranchInfo:
    """Represents a branch from an intersection with extended nodes."""
    direction: str  # "Left", "Right", "Forward"
    heading: float
    nodes: List[str]  # List of node IDs along this branch (1-2 nodes)


@dataclass
class NodeInfo:
    """Represents information about a single node."""
    node_id: str
    is_intersection: bool
    heading: float
    connections: List[NodeConnection]
    nearby_pois: List[Dict]
    poi_at_node: Optional[str] = None
    poi_letter: Optional[str] = None
    branches: List[BranchInfo] = field(default_factory=list)  # For intersections


@dataclass
class PresentationContext:
    """Container for all presentation data."""
    nodes: List[NodeInfo]
    poi_mapping: Dict[str, str]
    current_node_id: str
    current_heading: float
    start_node_id: Optional[str] = None
    intersection_branches: Dict[str, List[BranchInfo]] = field(default_factory=dict)


def _heading_to_compass(heading: float) -> str:
    """Convert heading degrees to compass direction."""
    heading = heading % 360
    if 337.5 <= heading or heading < 22.5:
        return "North"
    elif 22.5 <= heading < 67.5:
        return "Northeast"
    elif 67.5 <= heading < 112.5:
        return "East"
    elif 112.5 <= heading < 157.5:
        return "Southeast"
    elif 157.5 <= heading < 202.5:
        return "South"
    elif 202.5 <= heading < 247.5:
        return "Southwest"
    elif 247.5 <= heading < 292.5:
        return "West"
    else:
        return "Northwest"


def _format_node_id(node_id: str, max_length: int = 50) -> str:
    """Format node ID for display (truncate if too long)."""
    if len(node_id) <= max_length:
        return node_id
    return f"...{node_id[-max_length:]}"


def _calculate_heading_from_coords(
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
    compass_bearing = (math.degrees(initial_bearing) + 360) % 360

    return compass_bearing


def _get_relative_direction(current_heading: float, target_heading: float) -> str:
    """Returns relative direction (Forward, Left, Right, Back) based on heading difference."""
    diff = (target_heading - current_heading + 180) % 360 - 180
    if -45 <= diff <= 45:
        return "Forward"
    elif -135 <= diff < -45:
        return "Left"
    elif 45 < diff <= 135:
        return "Right"
    else:
        return "Back"


def _explore_intersection_branches(
    intersection_node_id: str,
    intersection_heading: float,
    area_nodes: Dict,
    area_links: List[Dict],
    path_node_ids: set,
    branch_depth: int = 2
) -> List[BranchInfo]:
    """
    Explore branches from an intersection node.

    For each direction (Left, Right, Forward), find 1-2 nodes along that branch.

    Args:
        intersection_node_id: The intersection node ID
        intersection_heading: Current heading at the intersection
        area_nodes: Dict of all nodes with lat/lng
        area_links: List of all links in the area
        path_node_ids: Set of node IDs already in the main path (to identify forward)
        branch_depth: Number of nodes to explore in each branch (default 2)

    Returns:
        List of BranchInfo for each direction
    """
    if not area_nodes or not area_links:
        return []

    # Build adjacency list
    adjacency = {}
    for link in area_links:
        src = link['source']
        tgt = link['target']
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append(tgt)

    if intersection_node_id not in adjacency:
        return []

    branches = []
    intersection_coords = area_nodes.get(intersection_node_id)

    if not intersection_coords:
        return []

    # Group neighbors by direction
    direction_branches = {"Left": [], "Right": [], "Forward": []}

    for neighbor_id in adjacency.get(intersection_node_id, []):
        neighbor_coords = area_nodes.get(neighbor_id)
        if not neighbor_coords:
            continue

        # Calculate heading to this neighbor
        heading_to_neighbor = _calculate_heading_from_coords(
            intersection_coords['lat'], intersection_coords['lng'],
            neighbor_coords['lat'], neighbor_coords['lng']
        )

        # Determine relative direction
        direction = _get_relative_direction(intersection_heading, heading_to_neighbor)

        if direction == "Back":
            continue  # Skip going back

        # Explore this branch
        branch_nodes = [neighbor_id]
        current_node = neighbor_id
        current_heading = heading_to_neighbor

        for _ in range(branch_depth - 1):
            if current_node not in adjacency:
                break

            # Find the best forward node from this position
            best_next = None
            best_diff = float('inf')

            for next_node in adjacency.get(current_node, []):
                if next_node == intersection_node_id:
                    continue  # Don't go back to intersection
                if next_node in branch_nodes:
                    continue  # Don't revisit

                next_coords = area_nodes.get(next_node)
                if not next_coords:
                    continue

                current_coords = area_nodes.get(current_node)
                if not current_coords:
                    continue

                heading_to_next = _calculate_heading_from_coords(
                    current_coords['lat'], current_coords['lng'],
                    next_coords['lat'], next_coords['lng']
                )

                # Prefer nodes that continue in roughly the same direction
                diff = abs((heading_to_next - current_heading + 180) % 360 - 180)
                if diff < best_diff and diff < 100:  # Forward tolerance
                    best_diff = diff
                    best_next = next_node

            if best_next:
                branch_nodes.append(best_next)
                current_node = best_next
            else:
                break

        direction_branches[direction].append(BranchInfo(
            direction=direction,
            heading=heading_to_neighbor,
            nodes=branch_nodes
        ))

    # Flatten and return all branches
    for direction in ["Forward", "Left", "Right"]:
        branches.extend(direction_branches[direction])

    return branches


def extract_presentation_context(
    path_data: List[Dict],
    previous_visited: List[Dict],
    poi_mapping: Dict[str, str],
    current_heading: float,
    area_nodes: Dict = None,
    area_links: List[Dict] = None,
    branch_depth: int = 2
) -> PresentationContext:
    """
    Extract presentation context from path_data for generating representations.

    Args:
        path_data: List of current path node dictionaries from navigate()
        previous_visited: List of previously visited node dictionaries
        poi_mapping: Dict mapping landmark name to assigned letter
        current_heading: Current heading in degrees
        area_nodes: Optional dict of all nodes with lat/lng (for branch exploration)
        area_links: Optional list of all links (for branch exploration)
        branch_depth: Number of nodes to explore in each intersection branch (default 2)

    Returns:
        PresentationContext with structured node and POI information
    """
    nodes = []
    current_node_id = path_data[0]['node_id'] if path_data else None
    start_node_id = previous_visited[0]['node_id'] if previous_visited else current_node_id
    intersection_branches = {}

    # Collect all path node IDs for reference
    path_node_ids = set()
    for node_data in (previous_visited or []) + (path_data or []):
        path_node_ids.add(node_data['node_id'])

    # Process all nodes (previous + current)
    all_nodes = (previous_visited or []) + (path_data or [])

    for node_data in all_nodes:
        connections = []
        branches = []

        # Process connectivity (node-to-node connections)
        for conn in node_data.get('connectivity', []):
            connections.append(NodeConnection(
                from_node=node_data['node_id'],
                to_node=conn['node_id'],
                heading=conn.get('heading', 0),
                direction=conn.get('direction', 'Forward'),
                is_poi=False
            ))

        # If this is an intersection, explore branches
        if node_data.get('is_intersection', False) and area_nodes and area_links:
            node_heading = node_data.get('direction', current_heading)
            branches = _explore_intersection_branches(
                intersection_node_id=node_data['node_id'],
                intersection_heading=node_heading,
                area_nodes=area_nodes,
                area_links=area_links,
                path_node_ids=path_node_ids,
                branch_depth=branch_depth
            )
            if branches:
                intersection_branches[node_data['node_id']] = branches

        # Process side POIs (node-to-POI connections)
        nearby_pois = []
        for poi_info in node_data.get('side_pois', []):
            poi_name = poi_info.get('poi', 'Unknown POI')
            poi_letter = poi_info.get('letter', '?')

            # Create connection for POI
            connections.append(NodeConnection(
                from_node=node_data['node_id'],
                to_node=poi_info.get('node_id', poi_name),
                heading=0,  # POI heading calculated from direction
                direction=poi_info.get('direction', 'Forward'),
                is_poi=True,
                poi_name=poi_name,
                distance=poi_info.get('distance', 0)
            ))

            nearby_pois.append({
                'name': poi_name,
                'letter': poi_letter,
                'direction': poi_info.get('direction', 'Forward'),
                'position': poi_info.get('position', 'nearby'),
                'distance': poi_info.get('distance', 0),
                'node_id': poi_info.get('node_id')
            })

        node_info = NodeInfo(
            node_id=node_data['node_id'],
            is_intersection=node_data.get('is_intersection', False),
            heading=node_data.get('direction', current_heading),
            connections=connections,
            nearby_pois=nearby_pois,
            poi_at_node=node_data.get('poi'),
            poi_letter=node_data.get('poi_letter'),
            branches=branches
        )
        nodes.append(node_info)

    return PresentationContext(
        nodes=nodes,
        poi_mapping=poi_mapping or {},
        current_node_id=current_node_id,
        current_heading=current_heading,
        start_node_id=start_node_id,
        intersection_branches=intersection_branches
    )


# =============================================================================
# 1. TEXTUAL / NATURAL-LANGUAGE REPRESENTATION
# =============================================================================

def _build_simplified_previous_path_textual(previous_visited: List[Dict]) -> str:
    """
    Build a simplified previous path string with connection info (direction, heading).

    Format for graphviz: node1 --[direction, heading°]--> node2 --[direction, heading°]--> current_position
    """
    if not previous_visited:
        return ""

    parts = []
    prev_node_ids = {n['node_id'] for n in previous_visited}

    for i, node in enumerate(previous_visited):
        node_id = _format_node_id(node['node_id'])
        connectivity = node.get('connectivity', [])

        # For all but the last node, find the connection to the next node
        if i < len(previous_visited) - 1:
            next_node_id = previous_visited[i + 1]['node_id']

            # Find connection info to next node
            conn_info = None
            for conn in connectivity:
                if conn.get('node_id') == next_node_id:
                    conn_info = conn
                    break

            if conn_info:
                direction = conn_info.get('direction', 'forward').lower()
                heading = conn_info.get('heading', 0)
                parts.append(f"{node_id} --[{direction}, {heading:.0f}°]-->")
            else:
                parts.append(f"{node_id} -->")
        else:
            # Last node - check for connection to current position
            conn_to_current = None
            for conn in connectivity:
                if conn.get('node_id') not in prev_node_ids:
                    conn_to_current = conn
                    break

            if conn_to_current:
                direction = conn_to_current.get('direction', 'forward').lower()
                heading = conn_to_current.get('heading', 0)
                parts.append(f"{node_id} --[{direction}, {heading:.0f}°]--> (current position)")
            else:
                parts.append(node_id)

    return " ".join(parts)


def _build_simplified_previous_path_natural_language(previous_visited: List[Dict]) -> str:
    """
    Build a natural language description of the previous path with connection info.

    Example: "Started at node X, moved forward (299°) to node Y, then right (29°) to current position node Z"
    """
    if not previous_visited:
        return ""

    if len(previous_visited) == 1:
        node = previous_visited[0]
        node_id = _format_node_id(node['node_id'])
        # Check if there's a connection to current position
        connectivity = node.get('connectivity', [])
        if connectivity:
            conn = connectivity[0]  # Connection to current position
            direction = conn.get('direction', 'forward').lower()
            heading = conn.get('heading', 0)
            compass = _heading_to_compass(heading)
            return f"Started at node {node_id}, then {direction} ({heading:.0f}°, {compass}) to current position"
        return f"Started at node {node_id}"

    parts = []
    for i, node in enumerate(previous_visited):
        node_id = _format_node_id(node['node_id'])
        connectivity = node.get('connectivity', [])

        if i == 0:
            # First node - find connection to next
            next_node_id = previous_visited[i + 1]['node_id']
            conn_info = None
            for conn in connectivity:
                if conn.get('node_id') == next_node_id:
                    conn_info = conn
                    break

            if conn_info:
                direction = conn_info.get('direction', 'forward').lower()
                heading = conn_info.get('heading', 0)
                compass = _heading_to_compass(heading)
                parts.append(f"Started at node {node_id}, moved {direction} ({heading:.0f}°, {compass})")
            else:
                parts.append(f"Started at node {node_id}, moved")
        elif i < len(previous_visited) - 1:
            # Middle nodes
            next_node_id = previous_visited[i + 1]['node_id']
            conn_info = None
            for conn in connectivity:
                if conn.get('node_id') == next_node_id:
                    conn_info = conn
                    break

            if conn_info:
                direction = conn_info.get('direction', 'forward').lower()
                heading = conn_info.get('heading', 0)
                compass = _heading_to_compass(heading)
                parts.append(f"to node {node_id}, then {direction} ({heading:.0f}°, {compass})")
            else:
                parts.append(f"to node {node_id}, then")
        else:
            # Last node - check for connection to current position
            # Find connection that's not to a previous node (it's to current position)
            prev_node_ids = {n['node_id'] for n in previous_visited}
            conn_to_current = None
            for conn in connectivity:
                if conn.get('node_id') not in prev_node_ids:
                    conn_to_current = conn
                    break

            if conn_to_current:
                direction = conn_to_current.get('direction', 'forward').lower()
                heading = conn_to_current.get('heading', 0)
                compass = _heading_to_compass(heading)
                parts.append(f"to node {node_id}, then {direction} ({heading:.0f}°, {compass}) to current position")
            else:
                parts.append(f"to node {node_id}")

    return " ".join(parts)


def generate_textual_representation(
    path_data: List[Dict],
    previous_visited: List[Dict] = None,
    poi_mapping: Dict[str, str] = None,
    current_heading: float = 0,
    area_nodes: Dict = None,
    area_links: List[Dict] = None,
    branch_depth: int = 2,
    verbose: bool = True
) -> str:
    """
    Generate a human-readable textual representation of the navigation context.

    Describes:
    - Simplified previous path with connection info (direction, heading)
    - Node-to-node connections with headings (for current path only)
    - Node-to-POI connections with directions
    - Intersection information with branch exploration (1-2 nodes per direction)
    - POI type/name where applicable

    Args:
        path_data: List of current path node dictionaries from navigate()
        previous_visited: List of previously visited node dictionaries
        poi_mapping: Dict mapping landmark name to assigned letter
        current_heading: Current heading in degrees
        area_nodes: Optional dict of all nodes with lat/lng (for branch exploration)
        area_links: Optional list of all links (for branch exploration)
        branch_depth: Number of nodes to explore in each intersection branch (default 2)
        verbose: If True, includes more detailed descriptions

    Returns:
        Human-readable textual description string
    """
    if previous_visited is None:
        previous_visited = []
    if poi_mapping is None:
        poi_mapping = {}

    # Extract context only for current path (not previous_visited for detailed display)
    context = extract_presentation_context(
        path_data, [], poi_mapping, current_heading,
        area_nodes=area_nodes, area_links=area_links, branch_depth=branch_depth
    )

    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("NAVIGATION CONTEXT - TEXTUAL REPRESENTATION")
    lines.append("=" * 60)
    lines.append("")

    # Current position info
    if context.current_node_id:
        lines.append(f"Current Position: Node {_format_node_id(context.current_node_id)}")
        lines.append(f"Current Heading: {context.current_heading:.1f}° ({_heading_to_compass(context.current_heading)})")
        lines.append("")

    # Simplified previous path with natural language description
    if previous_visited:
        path_str = _build_simplified_previous_path_natural_language(previous_visited)
        lines.append(f"Previous Path (already visited): {path_str}")
        lines.append("")

    # POI Legend
    if context.poi_mapping:
        lines.append("POI Legend:")
        for name, letter in context.poi_mapping.items():
            lines.append(f"  {letter} = {name}")
        lines.append("")

    # Node descriptions (only for current path, not previous visited)
    lines.append("-" * 40)
    lines.append("CURRENT PATH - NODE CONNECTIONS AND POIS")
    lines.append("-" * 40)

    for node in context.nodes:
        node_label = _format_node_id(node.node_id)
        node_type = "Intersection" if node.is_intersection else "Node"

        lines.append(f"\n{node_type} {node_label}:")

        # POI at this node
        if node.poi_at_node:
            letter = node.poi_letter or '?'
            lines.append(f"  * POI at this location: {node.poi_at_node} ({letter})")

        # Node-to-node connections
        node_connections = [c for c in node.connections if not c.is_poi]
        if node_connections:
            lines.append("  Connected to nodes:")
            for conn in node_connections:
                target = _format_node_id(conn.to_node)
                heading = conn.heading
                direction = conn.direction
                compass = _heading_to_compass(heading)
                lines.append(
                    f"    - Node {target} is to the {direction.lower()} "
                    f"(heading: {heading:.1f}°, {compass})"
                )

        # Intersection branch exploration (extended nodes)
        if node.is_intersection and node.branches:
            lines.append("  Branches from this intersection:")
            for branch in node.branches:
                direction = branch.direction
                heading = branch.heading
                compass = _heading_to_compass(heading)
                node_chain = " -> ".join(_format_node_id(n) for n in branch.nodes)
                lines.append(
                    f"    - {direction} branch (heading: {heading:.1f}°, {compass}):"
                )
                lines.append(f"      Path: {node_chain}")

        # Node-to-POI connections
        if node.nearby_pois:
            lines.append("  Nearby POIs:")
            for poi in node.nearby_pois:
                poi_name = poi['name']
                letter = poi['letter']
                direction = poi['direction']
                distance = poi.get('distance', 0)
                position = poi.get('position', direction.lower())

                if distance > 0:
                    lines.append(
                        f"    - {poi_name} ({letter}) is {position}, "
                        f"~{distance:.1f}m to the {direction.lower()}"
                    )
                else:
                    lines.append(
                        f"    - {poi_name} ({letter}) is {position} to the {direction.lower()}"
                    )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# =============================================================================
# 2. STRUCTURED JSON REPRESENTATION
# =============================================================================

def _build_simplified_previous_path_json(previous_visited: List[Dict]) -> List[Dict]:
    """
    Build a simplified previous path list with connection info for JSON.

    Returns list of: [{"node_id": "...", "to_next": {"direction": "forward", "heading": 299.1}}]
    The last node includes "to_current_position" if there's a connection to the current position.
    """
    if not previous_visited:
        return []

    prev_node_ids = {n['node_id'] for n in previous_visited}
    result = []

    for i, node in enumerate(previous_visited):
        node_entry = {"node_id": node['node_id']}
        connectivity = node.get('connectivity', [])

        if i < len(previous_visited) - 1:
            # Find connection to next node in path
            next_node_id = previous_visited[i + 1]['node_id']
            for conn in connectivity:
                if conn.get('node_id') == next_node_id:
                    node_entry["to_next"] = {
                        "direction": conn.get('direction', 'Forward'),
                        "heading": round(conn.get('heading', 0), 1)
                    }
                    break
        else:
            # Last node - check for connection to current position
            for conn in connectivity:
                if conn.get('node_id') not in prev_node_ids:
                    node_entry["to_current_position"] = {
                        "direction": conn.get('direction', 'Forward'),
                        "heading": round(conn.get('heading', 0), 1)
                    }
                    break

        result.append(node_entry)

    return result


def generate_json_representation(
    path_data: List[Dict],
    previous_visited: List[Dict] = None,
    poi_mapping: Dict[str, str] = None,
    current_heading: float = 0,
    area_nodes: Dict = None,
    area_pois: Dict = None,
    area_links: List[Dict] = None,
    branch_depth: int = 2,
    include_coordinates: bool = False,
    pretty_print: bool = True
) -> str:
    """
    Generate a structured JSON representation of the navigation context.

    The JSON includes:
    - Simplified previous path with connection info (direction, heading)
    - Clearly separated nodes and POIs (for current path only)
    - Connections, headings, and identifiers
    - Intersection branches with extended nodes (1-2 per direction)
    - Machine-readable and consistent format

    Args:
        path_data: List of current path node dictionaries from navigate()
        previous_visited: List of previously visited node dictionaries
        poi_mapping: Dict mapping landmark name to assigned letter
        current_heading: Current heading in degrees
        area_nodes: Optional dict of all nodes with lat/lng for coordinates
        area_pois: Optional dict of POI data
        area_links: Optional list of all links (for branch exploration)
        branch_depth: Number of nodes to explore in each intersection branch (default 2)
        include_coordinates: If True, include lat/lng for nodes
        pretty_print: If True, format JSON with indentation

    Returns:
        JSON string representation
    """
    if previous_visited is None:
        previous_visited = []
    if poi_mapping is None:
        poi_mapping = {}
    if area_nodes is None:
        area_nodes = {}
    if area_pois is None:
        area_pois = {}

    # Build simplified previous path with connection info
    previous_path_data = _build_simplified_previous_path_json(previous_visited)

    # Extract context only for current path (not previous_visited for detailed display)
    context = extract_presentation_context(
        path_data, [], poi_mapping, current_heading,
        area_nodes=area_nodes, area_links=area_links, branch_depth=branch_depth
    )

    # Build nodes section (only for current path)
    nodes_list = []
    for node in context.nodes:
        node_dict = {
            "id": node.node_id,
            "type": "intersection" if node.is_intersection else "waypoint",
            "heading": round(node.heading, 2),
            "connections": []
        }

        # Add coordinates if available and requested
        if include_coordinates and node.node_id in area_nodes:
            node_coords = area_nodes[node.node_id]
            node_dict["coordinates"] = {
                "lat": node_coords.get('lat'),
                "lng": node_coords.get('lng')
            }

        # Add POI at node if present
        if node.poi_at_node:
            node_dict["poi_at_location"] = {
                "name": node.poi_at_node,
                "letter": node.poi_letter or "?"
            }

        # Add connections (node-to-node only)
        for conn in node.connections:
            if not conn.is_poi:
                conn_dict = {
                    "target_node_id": conn.to_node,
                    "heading": round(conn.heading, 2),
                    "direction": conn.direction
                }
                node_dict["connections"].append(conn_dict)

        # Add intersection branches (extended nodes per direction)
        if node.is_intersection and node.branches:
            node_dict["branches"] = []
            for branch in node.branches:
                branch_dict = {
                    "direction": branch.direction,
                    "heading": round(branch.heading, 2),
                    "compass": _heading_to_compass(branch.heading),
                    "nodes": branch.nodes
                }
                node_dict["branches"].append(branch_dict)

        nodes_list.append(node_dict)

    # Build POIs section
    pois_list = []
    seen_pois = set()

    for node in context.nodes:
        for poi in node.nearby_pois:
            poi_key = (poi['name'], poi.get('node_id'))
            if poi_key in seen_pois:
                continue
            seen_pois.add(poi_key)

            poi_dict = {
                "name": poi['name'],
                "letter": poi['letter'],
                "nearby_node_id": node.node_id,
                "direction": poi['direction'],
                "position": poi.get('position', poi['direction'].lower()),
                "distance_meters": round(poi.get('distance', 0), 1)
            }

            # Add POI coordinates if available
            poi_node_id = poi.get('node_id')
            if include_coordinates and poi_node_id:
                if poi_node_id in area_pois:
                    poi_coords = area_pois[poi_node_id]
                    poi_dict["coordinates"] = {
                        "lat": poi_coords.get('lat'),
                        "lng": poi_coords.get('lng')
                    }

            pois_list.append(poi_dict)

    # Build the complete JSON structure
    result = {
        "navigation_context": {
            "current_position": {
                "node_id": context.current_node_id,
                "heading": round(context.current_heading, 2),
                "compass_direction": _heading_to_compass(context.current_heading)
            },
            "previous_path": previous_path_data if previous_path_data else None,
            "poi_legend": {
                letter: name for name, letter in context.poi_mapping.items()
            },
            "current_path_nodes": nodes_list,
            "pois": pois_list,
            "summary": {
                "previous_path_length": len(previous_path_data),
                "current_path_nodes": len(nodes_list),
                "total_pois": len(pois_list),
                "intersections": sum(1 for n in nodes_list if n["type"] == "intersection")
            }
        }
    }

    if pretty_print:
        return json.dumps(result, indent=2)
    return json.dumps(result)


def generate_json_representation_dict(
    path_data: List[Dict],
    previous_visited: List[Dict] = None,
    poi_mapping: Dict[str, str] = None,
    current_heading: float = 0,
    area_nodes: Dict = None,
    area_pois: Dict = None,
    area_links: List[Dict] = None,
    branch_depth: int = 2,
    include_coordinates: bool = False
) -> Dict:
    """
    Generate a structured JSON representation as a Python dict.
    Same as generate_json_representation but returns dict instead of string.
    """
    json_str = generate_json_representation(
        path_data, previous_visited, poi_mapping, current_heading,
        area_nodes, area_pois, area_links, branch_depth,
        include_coordinates, pretty_print=False
    )
    return json.loads(json_str)


# =============================================================================
# 3. VISUAL TEXT (GRAPHVIZ-STYLE) REPRESENTATION
# =============================================================================

def generate_graphviz_representation(
    path_data: List[Dict],
    previous_visited: List[Dict] = None,
    poi_mapping: Dict[str, str] = None,
    current_heading: float = 0,
    area_nodes: Dict = None,
    area_links: List[Dict] = None,
    branch_depth: int = 2,
    include_legend: bool = True,
    format_style: str = "arrow"  # "arrow" or "dot"
) -> str:
    """
    Generate a Graphviz-style text representation of the navigation graph.

    Example output:
        # Previous Path: Node_A --[forward, 299°]--> Node_B --[forward, 299°]--> Node_C
        Node_X --> Node_Y [heading: 50°, direction: Forward]
        Node_X --> restaurant_1 (POI) [heading: 30°, direction: Left]

    Args:
        path_data: List of current path node dictionaries from navigate()
        previous_visited: List of previously visited node dictionaries
        poi_mapping: Dict mapping landmark name to assigned letter
        current_heading: Current heading in degrees
        area_nodes: Optional dict of all nodes with lat/lng (for branch exploration)
        area_links: Optional list of all links (for branch exploration)
        branch_depth: Number of nodes to explore in each intersection branch (default 2)
        include_legend: If True, include POI legend at the top
        format_style: "arrow" for -->, "dot" for Graphviz DOT format

    Returns:
        Graphviz-style text representation string
    """
    if previous_visited is None:
        previous_visited = []
    if poi_mapping is None:
        poi_mapping = {}

    # Extract context only for current path (not previous_visited for detailed display)
    context = extract_presentation_context(
        path_data, [], poi_mapping, current_heading,
        area_nodes=area_nodes, area_links=area_links, branch_depth=branch_depth
    )

    lines = []

    if format_style == "dot":
        # Full Graphviz DOT format
        lines.append("digraph NavigationGraph {")
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box];")
        lines.append("")

        # Define special nodes
        if context.current_node_id:
            lines.append(f'    "{_format_node_id(context.current_node_id)}" [style=filled, fillcolor=lightblue, label="P (Current)"];')

        # Simplified previous path with connection info as a comment
        if previous_visited:
            path_str = _build_simplified_previous_path_textual(previous_visited)
            lines.append(f'    // Previous Path (already visited): {path_str}')
            lines.append("")

        # Define POI nodes
        seen_pois = set()
        for node in context.nodes:
            for poi in node.nearby_pois:
                if poi['name'] not in seen_pois:
                    seen_pois.add(poi['name'])
                    letter = poi['letter']
                    lines.append(f'    "{letter}_{poi["name"]}" [shape=ellipse, style=filled, fillcolor=lightyellow, label="{letter}: {poi["name"]}"];')

        lines.append("")

        # Define edges (only for current path nodes)
        seen_edges = set()
        for node in context.nodes:
            from_label = _format_node_id(node.node_id)

            # Node-to-node edges
            for conn in node.connections:
                if not conn.is_poi:
                    to_label = _format_node_id(conn.to_node)
                    edge_key = (from_label, to_label)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        lines.append(
                            f'    "{from_label}" -> "{to_label}" '
                            f'[label="h:{conn.heading:.0f}° {conn.direction}"];'
                        )

            # Node-to-POI edges
            for poi in node.nearby_pois:
                letter = poi['letter']
                poi_label = f"{letter}_{poi['name']}"
                edge_key = (from_label, poi_label)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    distance = poi.get('distance', 0)
                    lines.append(
                        f'    "{from_label}" -> "{poi_label}" '
                        f'[style=dashed, label="{poi["direction"]} ~{distance:.0f}m"];'
                    )

        lines.append("}")
    else:
        # Arrow-style format (simpler, more readable)
        lines.append("# Navigation Graph - Graphviz-Style Representation")
        lines.append("# Format: Source --> Target [attributes]")
        lines.append("")

        # Legend
        if include_legend and context.poi_mapping:
            lines.append("# POI Legend:")
            for name, letter in context.poi_mapping.items():
                lines.append(f"#   {letter} = {name}")
            lines.append("")

        # Position markers
        if context.current_node_id:
            lines.append(f"# Current Position (P): {_format_node_id(context.current_node_id)}")
        lines.append(f"# Current Heading: {context.current_heading:.1f}° ({_heading_to_compass(context.current_heading)})")
        lines.append("")

        # Simplified previous path with connection info
        if previous_visited:
            path_str = _build_simplified_previous_path_textual(previous_visited)
            lines.append(f"# Previous Path (already visited): {path_str}")
            lines.append("")

        # Graph edges (only for current path nodes)
        lines.append("# Current Path - Node Connections:")
        seen_edges = set()

        for node in context.nodes:
            from_label = _format_node_id(node.node_id)

            # Mark intersection nodes
            node_marker = " [INTERSECTION]" if node.is_intersection else ""

            # Node-to-node edges
            for conn in node.connections:
                if not conn.is_poi:
                    to_label = _format_node_id(conn.to_node)
                    edge_key = (from_label, to_label)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        lines.append(
                            f"{from_label}{node_marker} --> {to_label} "
                            f"[heading: {conn.heading:.0f}°, direction: {conn.direction}]"
                        )

        # Intersection branches (extended nodes)
        lines.append("")
        lines.append("# Intersection Branches (extended nodes):")
        for node in context.nodes:
            if node.is_intersection and node.branches:
                from_label = _format_node_id(node.node_id)
                for branch in node.branches:
                    direction = branch.direction
                    heading = branch.heading
                    compass = _heading_to_compass(heading)
                    node_chain = " --> ".join(_format_node_id(n) for n in branch.nodes)
                    lines.append(
                        f"{from_label} [INTERSECTION] --({direction})--> {node_chain} "
                        f"[heading: {heading:.0f}°, {compass}]"
                    )

        # POI edges
        lines.append("")
        lines.append("# POI Connections:")
        seen_poi_edges = set()

        for node in context.nodes:
            from_label = _format_node_id(node.node_id)

            for poi in node.nearby_pois:
                letter = poi['letter']
                poi_name = poi['name']
                edge_key = (from_label, poi_name)
                if edge_key not in seen_poi_edges:
                    seen_poi_edges.add(edge_key)
                    distance = poi.get('distance', 0)
                    position = poi.get('position', poi['direction'].lower())
                    lines.append(
                        f"{from_label} --> {poi_name} (POI:{letter}) "
                        f"[direction: {poi['direction']}, position: {position}, distance: ~{distance:.0f}m]"
                    )

    return "\n".join(lines)


# =============================================================================
# MULTI-FORMAT GENERATOR
# =============================================================================

def generate_all_representations(
    path_data: List[Dict],
    previous_visited: List[Dict] = None,
    poi_mapping: Dict[str, str] = None,
    current_heading: float = 0,
    area_nodes: Dict = None,
    area_pois: Dict = None,
    area_links: List[Dict] = None,
    branch_depth: int = 2,
    matrix_representation: List[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate all presentation formats at once.

    Args:
        path_data: List of current path node dictionaries from navigate()
        previous_visited: List of previously visited node dictionaries
        poi_mapping: Dict mapping landmark name to assigned letter
        current_heading: Current heading in degrees
        area_nodes: Optional dict of all nodes with lat/lng
        area_pois: Optional dict of POI data
        area_links: Optional list of all links (for branch exploration)
        branch_depth: Number of nodes to explore in each intersection branch (default 2)
        matrix_representation: Optional pre-computed matrix representation

    Returns:
        Dict containing all four representation formats:
        - 'textual': Natural language description
        - 'json': Structured JSON (as dict)
        - 'graphviz': Graphviz-style text
        - 'matrix': Matrix representation (if provided or computed)
    """
    if previous_visited is None:
        previous_visited = []
    if poi_mapping is None:
        poi_mapping = {}

    result = {
        'textual': generate_textual_representation(
            path_data, previous_visited, poi_mapping, current_heading,
            area_nodes=area_nodes, area_links=area_links, branch_depth=branch_depth
        ),
        'json': generate_json_representation_dict(
            path_data, previous_visited, poi_mapping, current_heading,
            area_nodes, area_pois, area_links, branch_depth, include_coordinates=False
        ),
        'graphviz': generate_graphviz_representation(
            path_data, previous_visited, poi_mapping, current_heading,
            area_nodes=area_nodes, area_links=area_links, branch_depth=branch_depth
        )
    }

    if matrix_representation is not None:
        result['matrix'] = matrix_representation

    return result


# =============================================================================
# NEXT-STEP GENERATOR FOR MULTI-STEP PROMPTING
# =============================================================================

class NavigationStepGenerator:
    """
    Generator for multi-step navigation prompting workflow.

    Consumes results from one navigation step and produces prompts/data
    for the subsequent step. Follows the design pattern of second_agent.py.
    """

    def __init__(
        self,
        area_data: Dict,
        poi_mapping: Dict[str, str] = None,
        pois: Dict[str, List[str]] = None,
        representation_format: RepresentationFormat = RepresentationFormat.MATRIX,
        branch_depth: int = 2,
        include_coordinates: bool = True
    ):
        """
        Initialize the step generator.

        Args:
            area_data: Dict containing area_nodes, area_links, area_pois
            poi_mapping: Dict mapping landmark name to assigned letter
            pois: Dict mapping landmark name to list of POI node IDs
            representation_format: Which format to use for context generation
            branch_depth: Number of nodes to explore in each intersection branch (default 2)
            include_coordinates: If True, include lat/lng coordinates in JSON output
        """
        self.area_data = area_data
        self.area_nodes = area_data.get('area_nodes', {})
        self.area_links = area_data.get('area_links', [])
        self.area_pois = area_data.get('area_pois', {})
        self.poi_mapping = poi_mapping or {}
        self.pois = pois or {}
        self.representation_format = representation_format
        self.branch_depth = branch_depth
        self.include_coordinates = include_coordinates

        # State tracking
        self.current_node_id: Optional[str] = None
        self.current_heading: float = 0
        self.previous_visited: List[Dict] = []
        self.step_count: int = 0
        self.navigation_history: List[Dict] = []

    def initialize(self, start_node_id: str, initial_heading: float):
        """
        Initialize the generator at the starting position.

        Args:
            start_node_id: Starting node ID
            initial_heading: Initial heading in degrees
        """
        self.current_node_id = start_node_id
        self.current_heading = initial_heading
        self.previous_visited = []
        self.step_count = 0
        self.navigation_history = []

    def generate_step_context(
        self,
        sub_instruction: str = "",
        planning_state: str = "",
        include_all_formats: bool = False
    ) -> Dict[str, Any]:
        """
        Generate context for the current navigation step.

        Args:
            sub_instruction: Current sub-instruction text
            planning_state: Current planning state string
            include_all_formats: If True, include all representation formats

        Returns:
            Dict containing:
            - 'path_data': Raw path data from navigate()
            - 'representation': Selected representation format output
            - 'all_representations': All formats (if include_all_formats=True)
            - 'metadata': Step metadata
        """
        from scorer.graph_context import navigate
        from scorer.grid_repsentation import convert2grid

        # Get path data for current position
        path_data = navigate(
            map_json=self.area_data,
            starting_point=self.current_node_id,
            heading=int(self.current_heading),
            pois=self.pois,
            poi_mapping=self.poi_mapping,
            units=1,
            last_instruction=False
        )

        # Generate matrix representation
        matrix = convert2grid(
            path_data,
            self.previous_visited,
            area_nodes=self.area_nodes,
            area_pois=self.area_pois,
            pois=self.pois,
            poi_mapping=self.poi_mapping
        ).tolist()

        # Generate selected representation
        if self.representation_format == RepresentationFormat.TEXTUAL:
            representation = generate_textual_representation(
                path_data, self.previous_visited, self.poi_mapping, self.current_heading,
                area_nodes=self.area_nodes,
                area_links=self.area_links,
                branch_depth=self.branch_depth
            )
        elif self.representation_format == RepresentationFormat.JSON:
            representation = generate_json_representation(
                path_data, self.previous_visited, self.poi_mapping, self.current_heading,
                area_nodes=self.area_nodes,
                area_pois=self.area_pois,
                area_links=self.area_links,
                branch_depth=self.branch_depth,
                include_coordinates=self.include_coordinates
            )
        elif self.representation_format == RepresentationFormat.GRAPHVIZ:
            representation = generate_graphviz_representation(
                path_data, self.previous_visited, self.poi_mapping, self.current_heading,
                area_nodes=self.area_nodes,
                area_links=self.area_links,
                branch_depth=self.branch_depth
            )
        else:  # MATRIX
            representation = "\n".join(str(row) for row in matrix)

        result = {
            'path_data': path_data,
            'matrix_representation': matrix,
            'representation': representation,
            'metadata': {
                'step_number': self.step_count,
                'current_node_id': self.current_node_id,
                'current_heading': self.current_heading,
                'sub_instruction': sub_instruction,
                'planning_state': planning_state,
                'format': self.representation_format.value
            }
        }

        if include_all_formats:
            result['all_representations'] = generate_all_representations(
                path_data, self.previous_visited, self.poi_mapping,
                self.current_heading,
                area_nodes=self.area_nodes,
                area_pois=self.area_pois,
                area_links=self.area_links,
                branch_depth=self.branch_depth,
                matrix_representation=matrix
            )

        return result

    def process_step_result(
        self,
        next_node_id: str,
        new_heading: Optional[float] = None,
        sub_plan_status: str = "IN_PROGRESS"
    ) -> Dict[str, Any]:
        """
        Process the result of a navigation step and prepare for the next step.

        Args:
            next_node_id: The node ID to move to
            new_heading: Optional new heading (calculated if not provided)
            sub_plan_status: Status of the current sub-plan

        Returns:
            Dict containing:
            - 'success': Whether the transition was valid
            - 'previous_node_id': The node we moved from
            - 'current_node_id': The node we're now at
            - 'heading_change': Change in heading
        """
        previous_node_id = self.current_node_id
        previous_heading = self.current_heading

        # Calculate new heading if not provided
        if new_heading is None and next_node_id != previous_node_id:
            new_heading = self._calculate_heading(previous_node_id, next_node_id)
        elif new_heading is None:
            new_heading = self.current_heading

        # Update state
        self.current_node_id = next_node_id
        self.current_heading = new_heading
        self.step_count += 1

        # Add previous position to visited list
        # Find the path data for the previous node
        from scorer.graph_context import navigate

        prev_path_data = navigate(
            map_json=self.area_data,
            starting_point=previous_node_id,
            heading=int(previous_heading),
            pois=self.pois,
            poi_mapping=self.poi_mapping,
            units=1,
            last_instruction=False
        )

        if prev_path_data:
            self.previous_visited.append(prev_path_data[0])

        # Record history
        step_record = {
            'step': self.step_count,
            'from_node': previous_node_id,
            'to_node': next_node_id,
            'from_heading': previous_heading,
            'to_heading': new_heading,
            'status': sub_plan_status
        }
        self.navigation_history.append(step_record)

        return {
            'success': True,
            'previous_node_id': previous_node_id,
            'current_node_id': self.current_node_id,
            'heading_change': new_heading - previous_heading,
            'step_record': step_record
        }

    def _calculate_heading(self, from_node_id: str, to_node_id: str) -> float:
        """Calculate heading from one node to another."""
        import math

        if from_node_id not in self.area_nodes or to_node_id not in self.area_nodes:
            return self.current_heading

        from_node = self.area_nodes[from_node_id]
        to_node = self.area_nodes[to_node_id]

        lat1 = math.radians(from_node['lat'])
        lon1 = math.radians(from_node['lng'])
        lat2 = math.radians(to_node['lat'])
        lon2 = math.radians(to_node['lng'])

        d_lon = lon2 - lon1
        y = math.sin(d_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)

        initial_bearing = math.atan2(y, x)
        compass_bearing = (math.degrees(initial_bearing) + 360) % 360

        return compass_bearing

    def get_navigation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the navigation so far.

        Returns:
            Dict containing navigation statistics and history
        """
        return {
            'total_steps': self.step_count,
            'start_node': self.navigation_history[0]['from_node'] if self.navigation_history else self.current_node_id,
            'current_node': self.current_node_id,
            'current_heading': self.current_heading,
            'visited_nodes': len(self.previous_visited),
            'history': self.navigation_history
        }

    def reset(self):
        """Reset the generator state."""
        self.current_node_id = None
        self.current_heading = 0
        self.previous_visited = []
        self.step_count = 0
        self.navigation_history = []


# =============================================================================
# BATCH REQUEST GENERATOR (following second_agent.py pattern)
# =============================================================================

def generate_batch_request(
    key: str,
    navigation_instruction: str,
    sub_instruction: str,
    current_heading: float,
    matrix_representation: List[List[str]],
    textual_representation: str,
    json_representation: Dict,
    graphviz_representation: str,
    planning_state: str,
    poi_legend: str = "",
    landmarks: str = "",
    current_position: str = "",
    format_type: str = "all"
) -> Dict[str, Any]:
    """
    Generate a batch request with all representation formats.
    Follows the pattern established in second_agent.py.

    Args:
        key: Unique identifier for this request
        navigation_instruction: Full navigation instruction
        sub_instruction: Current sub-instruction
        current_heading: Current heading in degrees
        matrix_representation: Matrix representation as list of lists
        textual_representation: Natural language representation
        json_representation: Structured JSON representation (as dict)
        graphviz_representation: Graphviz-style representation
        planning_state: Current planning state string
        poi_legend: POI legend string
        landmarks: Landmarks mapping string
        current_position: Current position description
        format_type: Which format to include ("all", "matrix", "textual", "json", "graphviz")

    Returns:
        Dict formatted as a batch request
    """
    request = {
        "custom_id": f"navigation-{key}",
        "method": "POST",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an embodied navigation agent."
                },
                {
                    "role": "user",
                    "content": ""
                }
            ]
        },
        "metadata": {
            "key": key,
            "navigation_instruction": navigation_instruction,
            "sub_instruction": sub_instruction,
            "heading": current_heading,
            "planning_state": planning_state
        }
    }

    # Build context based on format_type
    context_parts = []

    context_parts.append(f"[NAVIGATION INSTRUCTION]\n{navigation_instruction}\n")
    context_parts.append(f"[CURRENT SUB-GOAL]\n{sub_instruction}\n")
    context_parts.append(f"[CURRENT HEADING]\n{current_heading}° ({_heading_to_compass(current_heading)})\n")

    if landmarks:
        context_parts.append(f"[LANDMARKS]\n{landmarks}\n")

    if poi_legend:
        context_parts.append(f"[POI LEGEND]\n{poi_legend}\n")

    if current_position:
        context_parts.append(f"[YOUR POSITION]\n{current_position}\n")

    if planning_state:
        context_parts.append(f"[PLANNING STATE]\n{planning_state}\n")

    # Add representations based on format_type
    if format_type in ["all", "matrix"]:
        matrix_str = "\n".join(str(row) for row in matrix_representation)
        context_parts.append(f"[MATRIX REPRESENTATION]\n{matrix_str}\n")
        request["metadata"]["matrix"] = matrix_representation

    if format_type in ["all", "textual"]:
        context_parts.append(f"[TEXTUAL REPRESENTATION]\n{textual_representation}\n")
        request["metadata"]["textual"] = textual_representation

    if format_type in ["all", "json"]:
        json_str = json.dumps(json_representation, indent=2)
        context_parts.append(f"[JSON REPRESENTATION]\n{json_str}\n")
        request["metadata"]["json"] = json_representation

    if format_type in ["all", "graphviz"]:
        context_parts.append(f"[GRAPHVIZ REPRESENTATION]\n{graphviz_representation}\n")
        request["metadata"]["graphviz"] = graphviz_representation

    request["body"]["messages"][1]["content"] = "\n".join(context_parts)

    return request