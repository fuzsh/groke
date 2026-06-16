"""
Utility functions for graph navigation and POI operations.
"""

import math
import json
from typing import Dict, List, Tuple, Set, Any


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth in meters.

    Args:
        lat1, lng1: Latitude and longitude of first point
        lat2, lng2: Latitude and longitude of second point

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = math.sin(delta_phi/2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def calculate_bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the bearing (direction) from point 1 to point 2 in degrees (0-360).

    Args:
        lat1, lng1: Latitude and longitude of first point
        lat2, lng2: Latitude and longitude of second point

    Returns:
        Bearing in degrees (0-360, where 0 is North)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lng = math.radians(lng2 - lng1)

    y = math.sin(delta_lng) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lng)

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


def get_neighbors(current_node: str, area_links: List[Dict]) -> List[Dict]:
    """
    Get all neighboring nodes that can be reached from current node.

    Args:
        current_node: OSM ID of current node
        area_links: List of all links in the area

    Returns:
        List of dicts with 'target' and 'heading' keys
    """
    neighbors = []
    for link in area_links:
        if link['source'] == current_node:
            neighbors.append({
                'target': link['target'],
                'heading': link['heading']
            })
    return neighbors


def find_pois_near_node(node_id: str, area_pois: Dict, area_nodes: Dict,
                       max_distance: float = 100) -> List[Dict]:
    """
    Find all POIs within max_distance meters of a given node.

    Args:
        node_id: OSM ID of the node
        area_pois: Dictionary of POIs
        area_nodes: Dictionary of nodes
        max_distance: Maximum distance in meters

    Returns:
        List of dicts with POI information and distance
    """
    if node_id not in area_nodes:
        return []

    node = area_nodes[node_id]
    nearby_pois = []

    for poi_id, poi_data in area_pois.items():
        distance = haversine_distance(
            node['lat'], node['lng'],
            poi_data['lat'], poi_data['lng']
        )

        if distance <= max_distance:
            # Parse tags if they're in JSON format
            try:
                tags = json.loads(poi_data['tags']) if isinstance(poi_data['tags'], str) else poi_data['tags']
            except:
                tags = {}

            nearby_pois.append({
                'poi_id': poi_id,
                'distance': distance,
                'tags': tags,
                'lat': poi_data['lat'],
                'lng': poi_data['lng']
            })

    return sorted(nearby_pois, key=lambda x: x['distance'])


def extract_poi_keywords(tags: Dict) -> List[str]:
    """
    Extract relevant keywords from POI tags.

    Args:
        tags: Dictionary of OSM tags

    Returns:
        List of keywords
    """
    keywords = []

    # Common OSM tag keys that provide useful description
    relevant_keys = ['name', 'amenity', 'shop', 'building', 'leisure',
                    'tourism', 'highway', 'natural', 'historic']

    for key in relevant_keys:
        if key in tags:
            value = tags[key]
            if isinstance(value, str):
                # Split on common separators and add words
                words = value.lower().replace('_', ' ').replace('-', ' ').split()
                keywords.extend(words)

    return list(set(keywords))  # Remove duplicates


def calculate_path_distance(path: List[str], area_nodes: Dict) -> float:
    """
    Calculate total distance of a path in meters.

    Args:
        path: List of OSM node IDs
        area_nodes: Dictionary of nodes

    Returns:
        Total distance in meters
    """
    total_distance = 0
    for i in range(len(path) - 1):
        if path[i] in area_nodes and path[i+1] in area_nodes:
            node1 = area_nodes[path[i]]
            node2 = area_nodes[path[i+1]]
            total_distance += haversine_distance(
                node1['lat'], node1['lng'],
                node2['lat'], node2['lng']
            )
    return total_distance


def heading_difference(heading1: float, heading2: float) -> float:
    """
    Calculate the absolute difference between two headings (0-180 degrees).

    Args:
        heading1, heading2: Headings in degrees (0-360)

    Returns:
        Absolute difference in degrees (0-180)
    """
    diff = abs(heading1 - heading2)
    if diff > 180:
        diff = 360 - diff
    return diff


def get_direction_description(heading: float) -> str:
    """
    Convert a heading to a human-readable direction description.

    Args:
        heading: Heading in degrees (0-360)

    Returns:
        Direction string (e.g., "north", "northeast", "east", etc.)
    """
    directions = [
        "north", "northeast", "east", "southeast",
        "south", "southwest", "west", "northwest"
    ]

    # Divide 360 degrees into 8 sections
    index = round(heading / 45) % 8
    return directions[index]


def build_poi_description(poi: Dict) -> str:
    """
    Build a human-readable description of a POI.

    Args:
        poi: POI dictionary with tags

    Returns:
        Description string
    """
    tags = poi.get('tags', {})

    # Try to get name first
    if 'name' in tags:
        description = tags['name']

        # Add type if available
        for key in ['amenity', 'shop', 'building', 'leisure', 'tourism']:
            if key in tags:
                description += f" ({tags[key]})"
                break

        return description

    # If no name, use type
    for key in ['amenity', 'shop', 'building', 'leisure', 'tourism', 'highway']:
        if key in tags:
            return tags[key]

    return "unnamed location"