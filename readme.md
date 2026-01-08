# GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap

---

## 1. System Overview

The system implements a **Multi-Agent Agentic Navigation Framework** for route navigability on OpenStreetMap (OSM) graph data.

![plot](./assets/overall_architecture.png)

---

## 2. Reproducibility

## 2.1. Instruction Generation

## 2.2. Step Generator

## 2.3. Evaluation

## 2.3.0. How ?

## 2.3.1. Performance Report Paper

## 2.3.2. computational Report Paper



---

## 3. The Navigate Function: Visible Area Construction

The `navigate()` function is the **core algorithm** that constructs the agent's visible surroundings from the current position. It simulates what a human would see while walking forward from a given starting point.

![plot](./assets/visible_data_construction.png)

### 3.1 Function Signature

```python
def navigate(
    map_json: Dict,           # Contains area_links, area_pois, area_nodes
    starting_point: str,      # Current OSM node ID
    heading: int,             # Current heading in degrees (0-360)
    pois: Dict[str, List[str]] = None,    # Landmark name → POI node IDs
    poi_mapping: Dict[str, str] = None,   # Landmark name → Letter (A, B, C)
    units: int = 1,           # Number of intersections to traverse
    last_instruction: bool = False        # Whether this is the final sub-goal
) -> List[Dict]:              # Returns path_data
```


### 3.2 Key Helper Functions

#### 3.3.1 Heading Calculation (Spherical Bearing)
```python
def calculate_heading_from_coords(from_lat, from_lng, to_lat, to_lng) -> float:
    """Calculate compass heading (0-360°) from one coordinate to another."""
    lat1, lon1 = radians(from_lat), radians(from_lng)
    lat2, lon2 = radians(to_lat), radians(to_lng)

    d_lon = lon2 - lon1

    y = sin(d_lon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(d_lon)

    initial_bearing = atan2(y, x)
    compass_bearing = (degrees(initial_bearing) + 360) % 360

    return compass_bearing
```

#### 3.3.2 Heading Difference (Minimal Angular Distance)
```python
def get_heading_diff(h1, h2) -> float:
    """Calculate minimal difference between two headings (0-180)."""
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)
```

#### 3.3.3 Relative Direction Classification
```python
def get_relative_direction(current_heading, target_heading) -> str:
    """
    Classify target direction relative to current heading.

    Returns: "Forward" | "Left" | "Right" | "Back"
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
```

**Direction Classification Diagram:**
```
                    Forward
                   [-45°, +45°]
                       ▲
                       │
         Left          │          Right
      [-135°, -45°] ◄──┼──► [+45°, +135°]
                       │
                       ▼
                     Back
               [+135°, +180°] ∪ [-180°, -135°]
```

#### 3.3.4 Intersection Detection
```python
def is_intersection(node_id) -> bool:
    """Node is an intersection if it connects to > 2 unique nodes."""
    return len(node_connections.get(node_id, [])) > 2
```

#### 3.3.5 Forward Path Selection
```python
def get_connectivity_and_next(current_node, current_heading):
    """
    Returns:
    1. best_match: Best forward neighbor (within ±100° tolerance)
    3. connections: All non-back neighbors with directions
    """
    candidates = adjacency[current_node]

    # Find best forward match
    forward_links = []
    for link in candidates:
        tgt = link['target']
        link_heading = bearing(current_node → tgt)
        diff = heading_diff(link_heading, current_heading)

        if diff < 100:  # Forward tolerance
            forward_links.append((diff, {target: tgt, heading: link_heading}))

    best_match = min(forward_links, key=lambda x: x[0]) if forward_links else None

    # Build connectivity (exclude Back direction)
    connections = []
    for link in candidates:
        rel_dir = get_relative_direction(current_heading, bearing(current → link.target))
        if rel_dir != "Back":
            connections.append({
                node_id: link.target,
                heading: bearing,
                direction: rel_dir
            })

    return best_match, connections
```

### 3.3 Visibility Depth Control

The `units` parameter controls how far the agent can "see":

```
units = 1 (default):
┌─────────────────────────────────────────────────────────────────┐
│  [Start] ──► [Node] ──► [Node] ──► [INTERSECTION] ──► [+3 nodes]│
│                                          │                      │
│                                    STOP HERE                    │
│                              (1 intersection passed)            │
└─────────────────────────────────────────────────────────────────┘

units = 2:
┌─────────────────────────────────────────────────────────────────┐
│  [Start] ──► [INTER_1] ──► [Node] ──► [INTER_2] ──► [+3 nodes] │
│                                            │                    │
│                                      STOP HERE                  │
│                               (2 intersections passed)          │
└─────────────────────────────────────────────────────────────────┘
```

**Post-intersection extension**: After reaching the target intersection count, the algorithm continues for 3 more nodes to provide context about what lies ahead.

### 3.4 POI Association Logic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POI PROXIMITY MAPPING                                │
└─────────────────────────────────────────────────────────────────────────────┘

                    POI_A (Bank)
                        ●
                        │ 30m
                        │
    ═══════════════════●═══════════════════════●════════════════
                   path_node_1            path_node_2
                        │                       │
                        │ 8m                    │ 45m
                        │                       │
                        ●                       ●
                    POI_B (Cafe)            POI_C (Shop)

    Result:
    - POI_A → Associated with path_node_1 (nearest, 30m < 50m)
    - POI_B → Associated with path_node_1 (nearest, 8m < 50m)
    - POI_C → Associated with path_node_2 (nearest, 45m < 50m)

    In path_data:
    - path_node_1.side_pois = [POI_A (Right, 30m), POI_B (Left, 8m)]
    - path_node_3.side_pois = [POI_C (Right, 45m)]
```

**Distance Thresholds:**
- `< 5m`: POI is AT the node (stored as `poi`, `poi_letter`)
- `5m - 50m`: POI is NEARBY (stored in `side_pois`)
- `> 50m`: POI not associated with any path node

**Special Case - Intersection Corners:**
```python
if is_intersection and poi_direction in ["Left", "Right"]:
    position = "on the corner"
```
