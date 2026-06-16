from typing import List, Dict, Tuple
import math
import numpy as np


def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the Haversine distance between two points in meters.
    """
    R = 6371000  # Earth's radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


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


def get_delta_from_heading(h):
    h = h % 360
    if 45 <= h < 135:
        return (0, 1)  # East
    elif 135 <= h < 225:
        return (1, 0)  # South
    elif 225 <= h < 315:
        return (0, -1)  # West
    else:
        return (-1, 0)  # North


def get_corner_delta_from_heading(h):
    """
    Get diagonal corner position (delta row, delta col) based on heading.
    Returns position for corner placement (not on route segments).
    
    Quadrants:
    - 0-90: Northeast corner -> (-1, 1)
    - 90-180: Southeast corner -> (1, 1)
    - 180-270: Southwest corner -> (1, -1)
    - 270-360: Northwest corner -> (-1, -1)
    """
    h = h % 360
    if 0 <= h < 90:
        return (-1, 1)  # Northeast corner
    elif 90 <= h < 180:
        return (1, 1)  # Southeast corner
    elif 180 <= h < 270:
        return (1, -1)  # Southwest corner
    else:  # 270 <= h < 360
        return (-1, -1)  # Northwest corner


def get_relative_direction_from_heading(current_heading, target_heading):
    """Returns relative direction based on heading difference."""
    diff = (target_heading - current_heading + 180) % 360 - 180
    if -45 <= diff <= 45:
        return "Forward"
    elif -135 <= diff < -45:
        return "Left"
    elif 45 < diff <= 135:
        return "Right"
    else:
        return "Back"


def find_poi_grid_positions(
        pois: Dict[str, List[str]],
        poi_mapping: Dict[str, str],
        area_nodes: Dict,
        area_pois: Dict,
        path_locations: Dict[str, Tuple[int, int]],
        max_distance: float = 50.0,
        intersection_nodes: set = None,
        intersection_threshold: float = 30.0
) -> Dict[Tuple[int, int], str]:
    """
    Find grid positions for POIs by placing them near their nearest path nodes.
    If a POI is near an intersection (within threshold), place it at a corner adjacent to the intersection.

    Args:
        pois: Dict mapping landmark name to list of POI node IDs
        poi_mapping: Dict mapping landmark name to assigned letter
        area_nodes: All nodes with lat/lng
        area_pois: POI data with lat/lng
        path_locations: Dict mapping node_id to (row, col) in grid
        max_distance: Maximum distance in meters to consider
        intersection_nodes: Set of node IDs that are intersections
        intersection_threshold: Distance threshold in meters for placing POI at corner near intersection

    Returns:
        Dict mapping (row, col) -> POI letter
    """
    if intersection_nodes is None:
        intersection_nodes = set()
    
    poi_positions = {}

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

            # First, check if POI is near an intersection
            nearest_intersection_id = None
            min_intersection_distance = float('inf')
            nearest_intersection_node = None
            
            for intersection_node_id in intersection_nodes:
                if intersection_node_id not in area_nodes or intersection_node_id not in path_locations:
                    continue
                
                intersection_node = area_nodes[intersection_node_id]
                intersection_lat = intersection_node.get('lat')
                intersection_lng = intersection_node.get('lng')
                
                if intersection_lat is None or intersection_lng is None:
                    continue
                
                distance = calculate_distance(poi_lat, poi_lng, intersection_lat, intersection_lng)
                
                if distance < min_intersection_distance and distance <= intersection_threshold:
                    min_intersection_distance = distance
                    nearest_intersection_id = intersection_node_id
                    nearest_intersection_node = intersection_node

            # If POI is near an intersection, place it at a corner (diagonal, not on route)
            if nearest_intersection_id and min_intersection_distance <= intersection_threshold:
                # Calculate heading from intersection to POI to determine corner direction
                intersection_lat = nearest_intersection_node.get('lat')
                intersection_lng = nearest_intersection_node.get('lng')
                
                # Calculate heading from intersection to POI
                poi_heading = calculate_heading_from_coords(
                    intersection_lat, intersection_lng, poi_lat, poi_lng
                )
                
                # Get diagonal corner offset (not on route segments)
                dr, dc = get_corner_delta_from_heading(poi_heading)
                
                # Place POI at corner (diagonal to intersection, not on route)
                base_row, base_col = path_locations[nearest_intersection_id]
                poi_row = base_row + dr
                poi_col = base_col + dc
                
                poi_positions[(poi_row, poi_col)] = letter
                continue

            # Otherwise, find nearest path node (original logic)
            min_distance = float('inf')
            nearest_node_id = None

            for path_node_id, _ in path_locations.items():
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
                    nearest_node_id = path_node_id

            # If nearest node is within max_distance, place POI in grid
            if nearest_node_id and min_distance <= max_distance:
                # If POI is very close (<5m), place it at the same grid position
                if min_distance < 5.0:
                    poi_row, poi_col = path_locations[nearest_node_id]
                    poi_positions[(poi_row, poi_col)] = letter
                else:
                    # Place POI in adjacent cell based on direction
                    nearest_node = area_nodes[nearest_node_id]
                    nearest_lat = nearest_node.get('lat')
                    nearest_lng = nearest_node.get('lng')

                    # Calculate heading from path node to POI
                    poi_heading = calculate_heading_from_coords(
                        nearest_lat, nearest_lng, poi_lat, poi_lng
                    )

                    # Get direction offset
                    dr, dc = get_delta_from_heading(poi_heading)

                    # Place POI in adjacent cell
                    base_row, base_col = path_locations[nearest_node_id]
                    poi_row = base_row + dr
                    poi_col = base_col + dc

                    poi_positions[(poi_row, poi_col)] = letter

    return poi_positions


def setup_grid_dynamically(prev_list, curr_list, area_nodes=None, area_pois=None,
                           pois=None, poi_mapping=None):
    """
    Traverses the whole graph to find the bounds, then returns
    a configured grid and the start locations for both lists.

    Args:
        prev_list: List of previous visited nodes
        curr_list: List of current path nodes
        area_nodes: Dict of all nodes with lat/lng (for POI positioning)
        area_pois: Dict of POI data (for POI positioning)
        pois: Dict mapping landmark name to list of POI node IDs
        poi_mapping: Dict mapping landmark name to assigned letter
    """
    # Initialize POI structures
    if pois is None:
        pois = {}
    if poi_mapping is None:
        poi_mapping = {}
    if area_nodes is None:
        area_nodes = {}
    if area_pois is None:
        area_pois = {}

    # 1. Map all node data by ID for easy lookup
    all_nodes = {}
    for n in prev_list: all_nodes[n['node_id']] = n
    for n in curr_list: all_nodes[n['node_id']] = n

    # 2. Determine Start Node (Root of the traversal)
    if prev_list:
        root_id = prev_list[0]['node_id']
    else:
        root_id = curr_list[0]['node_id']

    # 3. BFS Traversal to map relative coordinates (0,0 is root)
    relative_coords = {root_id: (0, 0)}
    queue = [root_id]
    visited = {root_id}

    while queue:
        curr_id = queue.pop(0)
        curr_r, curr_c = relative_coords[curr_id]

        if curr_id in all_nodes:
            node = all_nodes[curr_id]
            for conn in node['connectivity']:
                target_id = conn['node_id']
                heading = conn['heading']

                # Calculate where target is relative to current
                dr, dc = get_delta_from_heading(heading)
                new_r, new_c = curr_r + dr, curr_c + dc

                # Save and enqueue if not visited
                if target_id not in visited:
                    visited.add(target_id)
                    relative_coords[target_id] = (new_r, new_c)
                    if target_id in all_nodes:
                        queue.append(target_id)

    # 4. Calculate initial bounds from path nodes
    rs = [pos[0] for pos in relative_coords.values()]
    cs = [pos[1] for pos in relative_coords.values()]

    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)

    # 5. Calculate offset for path nodes
    offset_r = -min_r + 1
    offset_c = -min_c + 1

    # 6. Create path locations with offsets
    path_locations = {}
    for nid, (r, c) in relative_coords.items():
        path_locations[nid] = (r + offset_r, c + offset_c)

    # 6.5. Identify intersection nodes
    intersection_nodes = set()
    for n in prev_list:
        if n.get('is_intersection', False):
            intersection_nodes.add(n['node_id'])
    for n in curr_list:
        if n.get('is_intersection', False):
            intersection_nodes.add(n['node_id'])

    # 7. Find POI positions (they may extend the grid)
    poi_grid_positions = find_poi_grid_positions(
        pois, poi_mapping, area_nodes, area_pois, path_locations, 
        max_distance=50.0, intersection_nodes=intersection_nodes,
        intersection_threshold=30.0
    )

    # 8. Recalculate bounds including POIs
    all_rows = [loc[0] for loc in path_locations.values()] + [pos[0] for pos in poi_grid_positions.keys()]
    all_cols = [loc[1] for loc in path_locations.values()] + [pos[1] for pos in poi_grid_positions.keys()]

    final_min_r, final_max_r = min(all_rows), max(all_rows)
    final_min_c, final_max_c = min(all_cols), max(all_cols)

    # 9. Determine Grid Size (with +2 padding)
    H = (final_max_r - final_min_r) + 3
    W = (final_max_c - final_min_c) + 3

    # 10. Adjust locations if POIs extended the grid
    if final_min_r < 0 or final_min_c < 0:
        extra_offset_r = max(0, 1 - final_min_r)
        extra_offset_c = max(0, 1 - final_min_c)

        # Update path locations
        final_locations = {}
        for nid, (r, c) in path_locations.items():
            final_locations[nid] = (r + extra_offset_r, c + extra_offset_c)

        # Update POI positions
        updated_poi_positions = {}
        for (r, c), letter in poi_grid_positions.items():
            updated_poi_positions[(r + extra_offset_r, c + extra_offset_c)] = letter
        poi_grid_positions = updated_poi_positions
    else:
        final_locations = path_locations

    # 11. Create Matrix
    grid = np.full((H, W), '0', dtype=object)

    return grid, H, W, final_locations, poi_grid_positions


def convert2grid(path_data, previous_visited, area_nodes=None, area_pois=None,
                 pois=None, poi_mapping=None):
    """
    Convert path data to grid representation with POI support.

    Args:
        path_data: List of current path node dictionaries
        previous_visited: List of previously visited node dictionaries
        area_nodes: Dict of all nodes with lat/lng (needed for POI positioning)
        area_pois: Dict of POI data (needed for POI positioning)
        pois: Dict mapping landmark name to list of POI node IDs
        poi_mapping: Dict mapping landmark name to assigned letter

    Returns:
        numpy array representing the grid
    """
    # Initialize structures
    if area_nodes is None:
        area_nodes = {}
    if area_pois is None:
        area_pois = {}

    # If pois and poi_mapping are not provided, extract from path_data.side_pois
    if pois is None or poi_mapping is None:
        extracted_pois = {}
        extracted_mapping = {}

        # Extract from path_data
        for node in (path_data or []):
            for side_poi in node.get('side_pois', []):
                poi_name = side_poi.get('poi')
                poi_node_id = side_poi.get('node_id')
                poi_letter = side_poi.get('letter')

                if poi_name and poi_node_id:
                    if poi_name not in extracted_pois:
                        extracted_pois[poi_name] = []
                    if poi_node_id not in extracted_pois[poi_name]:
                        extracted_pois[poi_name].append(poi_node_id)

                if poi_name and poi_letter:
                    extracted_mapping[poi_name] = poi_letter

            # Also check for POI at node
            if node.get('poi') and node.get('poi_letter'):
                poi_name = node['poi']
                poi_letter = node['poi_letter']
                # For POI at node, use the node_id itself or a nearby POI
                if poi_name not in extracted_pois:
                    extracted_pois[poi_name] = []
                # Try to find associated node_id from side_pois or use current node
                if poi_name not in extracted_mapping:
                    extracted_mapping[poi_name] = poi_letter

        # Extract from previous_visited
        for node in (previous_visited or []):
            for side_poi in node.get('side_pois', []):
                poi_name = side_poi.get('poi')
                poi_node_id = side_poi.get('node_id')
                poi_letter = side_poi.get('letter')

                if poi_name and poi_node_id:
                    if poi_name not in extracted_pois:
                        extracted_pois[poi_name] = []
                    if poi_node_id not in extracted_pois[poi_name]:
                        extracted_pois[poi_name].append(poi_node_id)

                if poi_name and poi_letter:
                    extracted_mapping[poi_name] = poi_letter

        # Use extracted values if not provided
        if pois is None:
            pois = extracted_pois
        if poi_mapping is None:
            poi_mapping = extracted_mapping

    if pois is None:
        pois = {}
    if poi_mapping is None:
        poi_mapping = {}

    # Calculate everything dynamically including POI positions
    grid_matrix, H, W, locations, poi_positions = setup_grid_dynamically(
        previous_visited, path_data, area_nodes, area_pois, pois, poi_mapping
    )

    # Identify the start node ID for marking 'P' and the initial start node as 'S'
    start_node_id = path_data[0]['node_id'] if path_data else None
    initial_start_node = previous_visited[0]['node_id'] if previous_visited else None

    # A. Process Previous Visited
    for node in previous_visited:
        curr_id = node['node_id']
        if curr_id in locations:
            r, c = locations[curr_id]

            if grid_matrix[r, c] == '0':
                grid_matrix[r, c] = '1'

            # Connections
            for conn in node['connectivity']:
                target_id = conn['node_id']
                if target_id in locations:
                    tr, tc = locations[target_id]
                    if grid_matrix[tr, tc] == '0':
                        grid_matrix[tr, tc] = '1'

            if initial_start_node and curr_id == initial_start_node:
                grid_matrix[r, c] = 'S'

    # B. Process Current Path
    for node in path_data:
        curr_id = node['node_id']
        if curr_id in locations:
            r, c = locations[curr_id]

            # 1. Draw Forward Connections (branches)
            for conn in node['connectivity']:
                target_id = conn['node_id']
                if target_id in locations:
                    tr, tc = locations[target_id]
                    if grid_matrix[tr, tc] == '0':
                        grid_matrix[tr, tc] = '2'

            # 2. Mark Current Node
            if curr_id == start_node_id:
                grid_matrix[r, c] = 'P' if initial_start_node else 'S'
            elif node['is_intersection']:
                grid_matrix[r, c] = '3'
            else:
                if grid_matrix[r, c] not in ['P', 'S']:
                    grid_matrix[r, c] = '2'

    # C. Place POIs in the grid
    for (poi_row, poi_col), letter in poi_positions.items():
        # Only place POI if the cell is empty or has a path marker (not S or P)
        if 0 <= poi_row < H and 0 <= poi_col < W:
            current_cell = grid_matrix[poi_row, poi_col]
            # Don't override S (start) or P (position) markers
            if current_cell not in ['S', 'P']:
                grid_matrix[poi_row, poi_col] = letter

    return grid_matrix


def get_node_id_from_position(path_data, previous_visited, next_move: List[int],
                              area_nodes=None, area_pois=None,
                              pois=None, poi_mapping=None):
    """
    Reverse engineers the node_id from a specific (row, col) position in the grid
    by re-calculating the relative coordinates and offsets used during grid construction.

    Args:
        path_data: List of current path node dictionaries
        previous_visited: List of previously visited node dictionaries
        next_move: [row, col] position in grid
        area_nodes: Dict of all nodes with lat/lng
        area_pois: Dict of POI data
        pois: Dict mapping landmark name to list of POI node IDs
        poi_mapping: Dict mapping landmark name to assigned letter
    """
    target_r, target_c = next_move

    # Initialize structures
    if pois is None:
        pois = {}
    if poi_mapping is None:
        poi_mapping = {}
    if area_nodes is None:
        area_nodes = {}
    if area_pois is None:
        area_pois = {}

    # 1. Map all node data by ID
    all_nodes = {}
    for n in previous_visited: all_nodes[n['node_id']] = n
    for n in path_data: all_nodes[n['node_id']] = n

    # 2. Determine Start Node (Root)
    if previous_visited:
        root_id = previous_visited[0]['node_id']
    elif path_data:
        root_id = path_data[0]['node_id']
    else:
        return None

    # 3. BFS Traversal to map relative coordinates (0,0 is root)
    relative_coords = {root_id: (0, 0)}
    queue = [root_id]
    visited = {root_id}

    while queue:
        curr_id = queue.pop(0)
        curr_r, curr_c = relative_coords[curr_id]

        if curr_id in all_nodes:
            node = all_nodes[curr_id]
            for conn in node['connectivity']:
                target_id = conn['node_id']
                heading = conn['heading']

                h = heading % 360
                if 45 <= h < 135:
                    dr, dc = (0, 1)  # East
                elif 135 <= h < 225:
                    dr, dc = (1, 0)  # South
                elif 225 <= h < 315:
                    dr, dc = (0, -1)  # West
                else:
                    dr, dc = (-1, 0)  # North

                new_r, new_c = curr_r + dr, curr_c + dc

                if target_id not in visited:
                    visited.add(target_id)
                    relative_coords[target_id] = (new_r, new_c)
                    if target_id in all_nodes:
                        queue.append(target_id)

    # 4. Calculate Offsets
    if not relative_coords:
        return None

    rs = [pos[0] for pos in relative_coords.values()]
    cs = [pos[1] for pos in relative_coords.values()]

    min_r = min(rs)
    min_c = min(cs)

    offset_r = -min_r + 1
    offset_c = -min_c + 1

    # 5. Create path locations
    path_locations = {}
    for nid, (r, c) in relative_coords.items():
        path_locations[nid] = (r + offset_r, c + offset_c)

    # 5.5. Identify intersection nodes
    intersection_nodes = set()
    for n in previous_visited:
        if n.get('is_intersection', False):
            intersection_nodes.add(n['node_id'])
    for n in path_data:
        if n.get('is_intersection', False):
            intersection_nodes.add(n['node_id'])

    # 6. Find POI positions
    poi_grid_positions = find_poi_grid_positions(
        pois, poi_mapping, area_nodes, area_pois, path_locations, 
        max_distance=50.0, intersection_nodes=intersection_nodes,
        intersection_threshold=30.0
    )

    # 7. Check for adjustments due to POI extension
    all_rows = [loc[0] for loc in path_locations.values()] + [pos[0] for pos in poi_grid_positions.keys()]
    all_cols = [loc[1] for loc in path_locations.values()] + [pos[1] for pos in poi_grid_positions.keys()]

    final_min_r = min(all_rows)
    final_min_c = min(all_cols)

    if final_min_r < 0 or final_min_c < 0:
        extra_offset_r = max(0, 1 - final_min_r)
        extra_offset_c = max(0, 1 - final_min_c)

        for nid in path_locations:
            r, c = path_locations[nid]
            path_locations[nid] = (r + extra_offset_r, c + extra_offset_c)

    # 8. Reverse lookup
    req_rel_r = target_r - offset_r
    req_rel_c = target_c - offset_c

    for node_id, (grid_r, grid_c) in path_locations.items():
        if grid_r == target_r and grid_c == target_c:
            return node_id

    return None

    # [
    # ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 
    # ['0', '0', '0', '0', '2', '0', '0', '0', '0', '0'], 
    # ['0', 'S', '2', '2', 'B', '2', '2', '2', '2', '0'], 
    # ['0', '0', '0', '0', '2', '0', '0', '0', '0', '0'],
    # ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
    #]