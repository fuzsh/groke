# OSM Agentic Navigation System Architecture
## Comprehensive Description for Academic Figure Generation

---

## 1. System Overview

The system implements a **Multi-Agent Agentic Navigation Framework** for vision-and-language navigation on OpenStreetMap (OSM) graph data. It employs a hierarchical decomposition strategy where natural language instructions are first parsed into executable sub-goals, then iteratively executed through a navigation agent that reasons over multiple spatial representations.

---

## 2. High-Level Architecture Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INPUT: Natural Language Instruction                   │
│           "Go straight to the bank, then turn right at the cafe"            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: INSTRUCTION PARSING AGENT                       │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │   Sub-Goal Extraction       │    │     Landmark Detection              │ │
│  │   • Decompose instruction   │    │     • Extract POI references        │ │
│  │   • Sequential actions      │    │     • Categorize: amenity, traffic  │ │
│  │   • MOVE_FORWARD, TURN_*    │    │     • Spatial relations (at, past)  │ │
│  └─────────────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │     Sub-Goals List    │           │    Landmarks List     │
        │  1. Go to the bank    │           │  A = "bank" (amenity) │
        │  2. Turn right @ cafe │           │  B = "cafe" (amenity) │
        └───────────────────────┘           └───────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 2: DATA PREPARATION                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        OSM Data Loading                                 │ │
│  │   • nodes.txt → {node_id: (lat, lng)}                                  │ │
│  │   • links.txt → [(source, heading, target)]                            │ │
│  │   • pois.txt → {poi_id: (lat, lng, tags)}                              │ │
│  │   • poi_links.txt → [(osm_node, poi_id)]                               │ │
│  │   • Neighbor expansion: N-degree graph around ground-truth path        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                       Landmark Grounding                                │ │
│  │   • Fuzzy matching (RapidFuzz partial_ratio > 70%)                     │ │
│  │   • Map instruction landmarks → OSM POI nodes                          │ │
│  │   • Assign unique identifiers: A, B, C, ... (excluding S, P)           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: AGENTIC NAVIGATION LOOP                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FOR each sub_goal in sub_goals:                                     │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │  │ 3.1 FETCH SURROUNDINGS                                          │ │    │
│  │  │     navigate(current_node, heading, pois, poi_mapping)          │ │    │
│  │  │     → Build path_data with:                                     │ │    │
│  │  │       • Node connectivity (Forward/Left/Right)                  │ │    │
│  │  │       • Intersection detection (degree > 2)                     │ │    │
│  │  │       • POI proximity mapping (< 50m threshold)                 │ │    │
│  │  └─────────────────────────────────────────────────────────────────┘ │    │
│  │                               │                                       │    │
│  │                               ▼                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │  │ 3.2 FORMAT CONVERSION                                     │ │    │
│  │  │                                                                  │ │    │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │ │    │
│  │  │  │     │ │    JSON    │ │        │ │   │   │ │    │
│  │  │  │     │ │ Structured │ │      │ │       │   │ │    │
│  │  │  │    │ │  Topology  │ │    │ │    │   │ │    │
│  │  │  │   │ │   Graph    │ │         │ │       │   │ │    │
│  │  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │ │    │
│  │  └─────────────────────────────────────────────────────────────────┘ │    │
│  │                               │                                       │    │
│  │                               ▼                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │  │ 3.3 LLM NAVIGATION AGENT QUERY                                  │ │    │
│  │  │     Input: Representation + Current Sub-Goal + Planning State   │ │    │
│  │  │     Output: {subplan_status, next_place}                        │ │    │
│  │  │     • subplan_status: "IN_PROGRESS" | "COMPLETED"               │ │    │
│  │  │     • next_place: node_id (str) or [row, col] (grid)            │ │    │
│  │  └─────────────────────────────────────────────────────────────────┘ │    │
│  │                               │                                       │    │
│  │                               ▼                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │  │ 3.4 STATE UPDATE                                                │ │    │
│  │  │     • Parse next_place (resolve grid→node if needed)            │ │    │
│  │  │     • Calculate new heading: bearing(current → next)            │ │    │
│  │  │     • Update predicted_path.append(next_node)                   │ │    │
│  │  │     • IF COMPLETED → increment sub_goal_index                   │ │    │
│  │  │     • IF IN_PROGRESS → continue with same sub_goal              │ │    │
│  │  └─────────────────────────────────────────────────────────────────┘ │    │
│  │                               │                                       │    │
│  │           ┌───────────────────┴───────────────────┐                   │    │
│  │           ▼                                       ▼                   │    │
│  │  ┌─────────────────┐                   ┌─────────────────┐           │    │
│  │  │   COMPLETED     │                   │   IN_PROGRESS   │           │    │
│  │  │ Move to next    │                   │ Same sub-goal   │           │    │
│  │  │ sub-goal        │                   │ New position    │           │    │
│  │  └─────────────────┘                   └─────────────────┘           │    │
│  │                                                                       │    │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  TERMINATION CONDITIONS:                                                     │
│  • All sub-goals completed (sub_goal_index >= len(sub_goals))               │
│  • Max total steps exceeded (steps >= 100)                                   │
│  • Max retries per sub-goal exceeded (retry >= 15)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 4: EVALUATION                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Navigation Metrics:                                                     │ │
│  │ • NE (Navigation Error): Haversine(predicted_end, goal)                │ │
│  │ • SR (Success Rate): NE < 25m                                          │ │
│  │ • OSR (Oracle Success Rate): Best method per instruction               │ │
│  │ • SDTW (Success weighted by DTW): Path similarity scoring              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Path Metrics:                                                           │ │
│  │ • Path Overlap: BFS expansion + Precision/Recall/F1                    │ │
│  │ • Multi-threshold analysis: [25m, 50m, 100m, 150m]                     │ │
│  │ • UpSet plots: Method agreement patterns                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Navigate Function: Visible Area Construction

The `navigate()` function is the **core algorithm** that constructs the agent's visible surroundings from the current position. It simulates what a human would see while walking forward from a given starting point.

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

### 3.2 Algorithm Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NAVIGATE FUNCTION: VISIBLE AREA BUILDER                   │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: starting_point, heading, pois, poi_mapping, units
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: BUILD GRAPH STRUCTURES                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ 1.1 Parse area_links → Adjacency List                               │  │
│  │     adjacency[source] = [{target: node_id}, ...]                    │  │
│  │                                                                      │  │
│  │ 1.2 Build node_connections for intersection detection               │  │
│  │     node_connections[node] = {neighbor1, neighbor2, ...}            │  │
│  │     Intersection = node where len(connections) > 2                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: FIRST PASS - COLLECT PATH NODES                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Starting from starting_point, traverse forward:                      │  │
│  │                                                                      │  │
│  │ WHILE iterations < MAX (1000) AND intersections_passed < units:     │  │
│  │   1. Add current_node to path_node_ids                              │  │
│  │   2. IF is_intersection(current_node):                              │  │
│  │        intersections_passed++                                        │  │
│  │        IF intersections_passed >= units:                            │  │
│  │           Add 3 more nodes after intersection, then BREAK           │  │
│  │   3. Find best_match (forward neighbor within ±100° tolerance)      │  │
│  │   4. Move: current_node = best_match.target                         │  │
│  │           current_heading = best_match.heading                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  OUTPUT: path_node_ids = {node1, node2, node3, ...}                       │
└───────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: POI PROXIMITY MAPPING                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ find_nearest_path_nodes_for_pois(pois, path_node_ids, max_dist=50m) │  │
│  │                                                                      │  │
│  │ FOR each landmark in pois:                                          │  │
│  │   FOR each poi_node_id in landmark's POI list:                      │  │
│  │     1. Get POI coordinates (lat, lng)                               │  │
│  │     2. FOR each path_node in path_node_ids:                         │  │
│  │          Calculate Haversine distance(POI, path_node)               │  │
│  │          Track minimum distance                                      │  │
│  │     3. IF min_distance <= 50m:                                      │  │
│  │          Associate POI with nearest_path_node                       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  OUTPUT: path_node_to_pois = {node_id: [(landmark, letter, poi_id, dist)]}│
└───────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: SECOND PASS - BUILD PATH_DATA WITH CONTEXT                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Starting from starting_point again, build detailed node info:       │  │
│  │                                                                      │  │
│  │ FOR each node along the path:                                       │  │
│  │   1. Detect if intersection: is_intersection(current_node)          │  │
│  │   2. Get connectivity: all neighbors with directions                │  │
│  │      ┌─────────────────────────────────────────────────────────┐   │  │
│  │      │ FOR each neighbor:                                       │   │  │
│  │      │   heading_to_neighbor = bearing(current → neighbor)     │   │  │
│  │      │   relative_dir = classify(heading, heading_to_neighbor) │   │  │
│  │      │   IF relative_dir != "Back": add to connections         │   │  │
│  │      └─────────────────────────────────────────────────────────┘   │  │
│  │   3. Get nearby POIs from path_node_to_pois                        │  │
│  │      ┌─────────────────────────────────────────────────────────┐   │  │
│  │      │ FOR each POI near this node:                            │   │  │
│  │      │   poi_heading = bearing(current → POI)                  │   │  │
│  │      │   poi_direction = classify(heading, poi_heading)        │   │  │
│  │      │   IF distance < 5m: POI is AT this node                 │   │  │
│  │      │   ELSE: POI is SIDE_POI with direction                  │   │  │
│  │      │   IF is_intersection AND (Left/Right): "on the corner"  │   │  │
│  │      └─────────────────────────────────────────────────────────┘   │  │
│  │   4. Build node_step dictionary                                    │  │
│  │   5. Advance to next node (best forward match)                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: path_data = [node_step_1, node_step_2, ...]                       │
└───────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Key Helper Functions

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
    2. connections: All non-back neighbors with directions
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

### 3.4 Output Structure: path_data

Each element in `path_data` represents one node along the visible path:

```python
node_step = {
    # Core node info
    "node_id": "osm_123456789",       # OSM node identifier
    "is_intersection": True,           # True if degree > 2
    "direction": 45.5,                 # Current heading at this node

    # Connectivity - available directions from this node
    "connectivity": [
        {
            "node_id": "osm_234567890",
            "heading": 45.5,           # Absolute heading to this neighbor
            "direction": "Forward"     # Relative direction
        },
        {
            "node_id": "osm_345678901",
            "heading": 135.0,
            "direction": "Right"
        },
        {
            "node_id": "osm_456789012",
            "heading": 315.0,
            "direction": "Left"
        }
    ],

    # POIs visible from this node (> 5m away)
    "side_pois": [
        {
            "poi": "bank",             # Landmark name from instruction
            "letter": "A",             # Assigned identifier
            "direction": "Right",      # Relative to current heading
            "position": "on the corner", # or "left", "right", "forward"
            "node_id": "poi_567890",   # POI node ID
            "distance": 15.3           # Distance in meters
        }
    ],

    # POI at this exact node (< 5m away) - optional
    "poi": "cafe",                     # Landmark name
    "poi_letter": "B",                 # Assigned identifier
    "poi_position": "forward"          # Position description
}
```

### 3.5 Visibility Depth Control

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

### 3.6 POI Association Logic

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
    - path_node_2.side_pois = [POI_C (Right, 45m)]
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

### 3.7 Complete Flow Example

```
INPUT:
  starting_point = "osm_100"
  heading = 0° (North)
  pois = {"bank": ["poi_200"], "cafe": ["poi_300"]}
  poi_mapping = {"bank": "A", "cafe": "B"}
  units = 1

PHASE 1 - Build Adjacency:
  adjacency = {
    "osm_100": [{target: "osm_101"}],
    "osm_101": [{target: "osm_102"}, {target: "osm_103"}, {target: "osm_104"}],
    "osm_102": [{target: "osm_105"}],
    ...
  }
  node_connections = {
    "osm_100": {"osm_101"},
    "osm_101": {"osm_100", "osm_102", "osm_103", "osm_104"},  # degree=4 → INTERSECTION
    ...
  }

PHASE 2 - First Pass (Collect Nodes):
  Iteration 1: current_node = osm_100 → not intersection → move to osm_101
  Iteration 2: current_node = osm_101 → IS INTERSECTION → intersections_passed=1
               Since units=1, add 3 more nodes: osm_102, osm_105, osm_106
               BREAK

  path_node_ids = {osm_100, osm_101, osm_102, osm_105, osm_106}

PHASE 3 - POI Mapping:
  poi_200 (bank): nearest to osm_101, distance=25m → Associate
  poi_300 (cafe): nearest to osm_102, distance=40m → Associate

  path_node_to_pois = {
    "osm_101": [("bank", "A", "poi_200", 25.0)],
    "osm_102": [("cafe", "B", "poi_300", 40.0)]
  }

PHASE 4 - Second Pass (Build path_data):
  Node osm_100:
    is_intersection = False
    connectivity = [{node_id: osm_101, heading: 5°, direction: Forward}]
    side_pois = []

  Node osm_101:
    is_intersection = True
    connectivity = [
      {node_id: osm_102, heading: 10°, direction: Forward},
      {node_id: osm_103, heading: 90°, direction: Right},
      {node_id: osm_104, heading: 270°, direction: Left}
    ]
    side_pois = [{poi: "bank", letter: "A", direction: Right, position: "on the corner", distance: 25.0}]

  Node osm_102:
    is_intersection = False
    connectivity = [{node_id: osm_105, heading: 15°, direction: Forward}]
    side_pois = [{poi: "cafe", letter: "B", direction: Right, distance: 40.0}]

  ... (osm_105, osm_106)

OUTPUT: path_data = [node_step_100, node_step_101, node_step_102, node_step_105, node_step_106]
```

---

## 4. Detailed Component Descriptions

### 4.1 Stage 1: Instruction Parsing Agent

**Purpose**: Parse natural language navigation instruction into structured, machine-readable components.

**Agent**: `InstructionDivider`

**Inputs**:
- Full navigation instruction (natural language string)

**Outputs**:
```json
{
  "landmarks": [
    {"name": "bank", "category": "amenity"},
    {"name": "cafe", "category": "amenity"}
  ],
  "sub_goals": [
    {"description": "Go straight to the bank", "status": "TODO"},
    {"description": "Turn right at the cafe", "status": "TODO"}
  ]
}
```

**Processing**:
1. **Landmark Extraction**: Identify physical entities (traffic lights, stop signs, banks, shops, restaurants, parks, etc.)
2. **Action Decomposition**: Break into atomic actions using only: `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`
3. **Relation Mapping**: Extract spatial relationships (at, past, before, after)

**Prompt Template** (from `prompts.py`):
```
[SYSTEM ROLE]
You are a Navigation Instruction Parser. Your goal is to translate natural
language navigation instructions into structured, machine-readable sub-goals
compatible with OpenStreetMap (OSM) data.

[DEFINITIONS]
1. LANDMARKS (OSM POIs): Traffic lights, stop signs, banks, shops, restaurants...
2. ACTIONS: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
3. RELATIONS: Spatial relationship between Action and Landmark
```

---

### 4.2 Stage 2: Data Preparation

#### 4.2.1 OSM Data Loading (`src/data_loader.py`)

**Function**: `get_data_by_instruction(instruction_id, split_file, base_path, neighbor_degrees=2)`

**Process**:
1. Load ground-truth path (`osm_path`) from split file
2. Load OSM graph data:
   - **nodes.txt**: `{osm_id, lat, lng}`
   - **links.txt**: `{source, heading, target}`
   - **pois.txt**: `{poi_id, lat, lng, tags}`
   - **poi_links.txt**: `{osm_node, poi_id}`
3. Expand area by N neighbor degrees (BFS from path nodes)
4. Filter to connected subgraph

**Output**:
```python
{
  "instruction_data": {...},
  "area_nodes": {osm_id: {"lat": float, "lng": float}},
  "area_links": [{"source": str, "target": str, "heading": float}],
  "area_pois": {poi_id: {"lat": float, "lng": float, "tags": dict}},
  "area_poi_links": [{"osm_id": str, "poi_id": str}]
}
```

#### 4.2.2 Landmark Grounding

**Function**: `_find_available_pois(area_pois, landmarks)`

**Method**: Fuzzy string matching using RapidFuzz
- Threshold: `partial_ratio > 70%`
- Maps instruction landmarks → OSM POI node IDs
- Assigns unique letter identifiers (A, B, C, ..., excluding S and P)

---

### 4.3 Stage 3: Agentic Navigation Loop

#### 4.3.1 Fetch Surroundings (Graph Traversal)

**Function**: `navigate(map_json, starting_point, heading, pois, poi_mapping, units=1)`

**Location**: `scorer/graph_context.py`

**Algorithm**:
1. Build adjacency list from `area_links`
2. Detect intersections: nodes with degree > 2
3. Heading-relative traversal:
   - For each neighbor, calculate bearing from current position
   - Classify by relative direction:
     - **Forward**: heading ± 45°
     - **Left**: heading - 135° to heading - 45°
     - **Right**: heading + 45° to heading + 135°
4. Track POIs within 50m of path nodes

**Output** (`path_data`):
```python
[
  {
    "node_id": "osm_123",
    "is_intersection": False,
    "connectivity": [
      {"node_id": "osm_124", "heading": 45.5, "direction": "Forward"}
    ],
    "side_pois": [
      {"poi": "bank", "letter": "A", "direction": "Right", "distance": 30.5}
    ],
    "direction": 45.0  # current heading
  },
  ...
]
```

#### 4.3.2 Multi-Format Conversion

**Location**: `scorer/json_formats.py`

The system generates **four parallel representations** of the navigation context:

##### JSON Representation
Structured graph topology with explicit connections.

```json
{
  "current_position": {
    "node_id": "osm_123",
    "is_intersection": false,
    "heading_degrees": 45.0
  },
  "nodes": [
    {
      "node_id": "osm_123",
      "type": "waypoint",
      "connections": [
        {"target_node_id": "osm_124", "heading": 45.5, "direction": "Forward"}
      ]
    }
  ],
  "pois": [
    {"poi_id": "poi_456", "name": "Bank", "letter": "A", "direction": "Right"}
  ]
}
```


#### 4.3.3 LLM Navigation Agent Query

**Agent**: `NodeNavigator`

**Input Structure**:
```
[TASK DESCRIPTION]
You are an embodied agent navigating using a {format_type} representation.

[INPUT FORMAT]
Instruction: {full_instruction}
Current Sub-Goal: {current_sub_goal}
Sub-Goal State: IN PROGRESS
{landmark_legend}

Navigation Context:
{representation}

Planning State:
{planning_state}

[OUTPUT FORMAT - STRICT JSON]
1. subplan_status: "IN_PROGRESS" | "COMPLETED"
2. next_place: node_id (string) OR [row, col] (grid)
```

**Output Schema** (from `templates.py`):
```json
{
  "type": "OBJECT",
  "properties": {
    "subplan_status": {
      "type": "STRING",
      "enum": ["IN_PROGRESS", "COMPLETED"]
    },
    "next_place": {"type": "STRING"}  // or ARRAY for grid
  },
  "required": ["subplan_status", "next_place"]
}
```

#### 4.3.4 State Update

**Functions**: `_get_heading_to_node()`, `process_results_for_method()`

**Location**: `data_manager.py`, `step_generator.py`

**Heading Calculation (Spherical Bearing)**:
```python
def calculate_bearing(lat1, lng1, lat2, lng2):
    dLon = radians(lng2 - lng1)
    lat1, lat2 = radians(lat1), radians(lat2)

    y = sin(dLon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)

    bearing = degrees(atan2(y, x))
    compass_bearing = (bearing + 360) % 360  # Normalize to [0, 360)
    return compass_bearing
```

**State Update Logic**:
```python
if subplan_status == "COMPLETED":
    # Move to next sub-goal
    current_sub_goal_index += 1
    num_retry_current_instruction = 0
else:  # IN_PROGRESS
    # Continue with same sub-goal from new position
    current_node = next_node
    current_heading = calculate_bearing(prev_node, next_node)
    predicted_path.append(next_node)
```

**Grid Resolution** (`resolve_grid_next_place()`):
- Converts `[row, col]` grid coordinates to OSM node ID
- Uses `get_node_id_from_position()` from grid representation

---

### 4.4 Stage 4: Evaluation

**Location**: `evaluation_metrics.py`

#### 4.4.1 Navigation Metrics

##### Navigation Error (NE)
Distance between agent's stopping point and goal destination.

```python
def calculate_navigation_error(predicted_path, osm_path, area_nodes):
    pred_end = predicted_path[-1]
    osm_end = osm_path[-1]
    return haversine_distance(
        area_nodes[pred_end]['lat'], area_nodes[pred_end]['lng'],
        area_nodes[osm_end]['lat'], area_nodes[osm_end]['lng']
    )
```

##### Success Rate (SR)
Binary success metric.
```
SR = 1 if NE < 25m else 0
```

##### Oracle Success Rate (OSR)
Best-case success when optimal method is selected per instruction.
```
OSR = max(SR_json, SR_textual, SR_grid, SR_graphvis)
```

##### Success weighted by Path Length (SDTW)
Dynamic Time Warping based path similarity.

#### 4.4.2 Path Metrics

**Path Expansion** (`expand_predicted_path()`):
- Uses BFS to find intermediate nodes between predicted waypoints
- Removes duplicates while preserving order

**Path Overlap**:
```python
overlap = len(set(expanded_path) & set(osm_path))
precision = overlap / len(expanded_path)
recall = overlap / len(osm_path)
f1 = 2 * precision * recall / (precision + recall)
```

#### 4.4.3 Haversine Distance Formula

```python
def haversine_distance(lat1, lng1, lat2, lng2):
    R = 6371000  # Earth's radius in meters

    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lng2 - lng1)

    a = sin(delta_phi/2)**2 + cos(phi1)*cos(phi2)*sin(delta_lambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c  # Distance in meters
```

---

## 4. Key Algorithms Summary

| Algorithm | Purpose | Formula/Method |
|-----------|---------|----------------|
| **Spherical Bearing** | Calculate heading between two coordinates | `atan2(sin(dLon)*cos(lat2), cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dLon))` |
| **Haversine Distance** | Calculate distance between two coordinates | `R * 2 * atan2(sqrt(a), sqrt(1-a))` |
| **BFS Path Finding** | Find shortest path in graph | Standard BFS with path reconstruction |
| **Fuzzy Matching** | Match landmarks to POIs | RapidFuzz `partial_ratio > 70%` |
| **Heading-Relative Direction** | Classify neighbor direction | Forward: ±45°, Left: -135° to -45°, Right: +45° to +135° |
| **Intersection Detection** | Identify decision points | Node degree > 2 |
| **Grid Cell Mapping** | Convert coordinates to grid | Delta-based stepping with heading normalization |

---

## 5. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COMPLETE DATA FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

1. INPUT
   └── Natural Language Instruction
         │
         ▼
2. INSTRUCTION PARSING (Agent 1: InstructionDivider)
   ├── Sub-Goals: ["Go to bank", "Turn right at cafe"]
   └── Landmarks: [{name: "bank", category: "amenity"}, ...]
         │
         ▼
3. DATA LOADING
   ├── OSM Graph: nodes, links, pois
   └── Neighbor Expansion: N-degree subgraph
         │
         ▼
4. LANDMARK GROUNDING
   └── Fuzzy Match: "bank" → osm_poi_123 → Letter "A"
         │
         ▼
5. NAVIGATION LOOP (Agent 2: NodeNavigator)
   │
   ├──► 5a. GRAPH TRAVERSAL (navigate())
   │        └── path_data with connectivity & POIs
   │              │
   │              ▼
   ├──► 5b. MULTI-FORMAT CONVERSION
   │        ├── Textual: Natural language
   │        ├── JSON: Structured graph
   │        ├── Grid: Matrix map
   │        └── GraphViz: ASCII graph
   │              │
   │              ▼
   ├──► 5c. LLM QUERY
   │        └── Output: {status, next_place}
   │              │
   │              ▼
   ├──► 5d. STATE UPDATE
   │        ├── New position
   │        ├── New heading (bearing calculation)
   │        └── Sub-goal progress
   │              │
   │              ▼
   └──► TERMINATION CHECK
        ├── All sub-goals done? → EXIT
        ├── Max steps reached? → EXIT
        └── Else → LOOP BACK TO 5a
               │
               ▼
6. EVALUATION
   ├── Navigation Error (NE): Haversine distance
   ├── Success Rate (SR): NE < 25m
   ├── Path Overlap: Precision/Recall/F1
   └── Method Comparison: UpSet analysis
```

---

## 6. File Structure for Reference

| File | Purpose |
|------|---------|
| `data_manager.py` | State management, heading calculation, results processing |
| `step_generator.py` | Prompt generation, multi-format conversion |
| `scorer/orchestrator.py` | Main agentic loop, multi-agent coordination |
| `scorer/graph_context.py` | Graph traversal, navigate() function |
| `scorer/presentation_formats.py` | 4 representation generators |
| `scorer/grid_representation.py` | Grid conversion & POI placement |
| `prompts.py` | LLM prompt templates |
| `templates.py` | Batch request formatting |
| `evaluation_metrics.py` | NE, SR, OSR, SDTW, path overlap |
| `src/data_loader.py` | OSM data loading |

---

## 7. Figure Annotation Suggestions

### Main Architecture Figure Elements:

1. **Input Box**: Natural language instruction with example
2. **Agent 1 Block**: InstructionDivider with sub-goals + landmarks output
3. **Data Loading Block**: OSM components (nodes, links, POIs)
4. **Navigation Loop Block**:
   - Graph traversal sub-block
   - 4 parallel representation boxes (Textual/JSON/Grid/GraphViz)
   - LLM query arrow
   - State update feedback loop
5. **Decision Diamond**: Status check (COMPLETED/IN_PROGRESS)
6. **Evaluation Block**: Metrics (NE, SR, OSR, SDTW)

### Color Coding Suggestion:
- **Blue**: Input/Output data
- **Green**: Agent processing blocks
- **Orange**: OSM data components
- **Purple**: Multi-format representations
- **Red**: Evaluation metrics

### Arrow Labels:
- Solid arrows: Data flow
- Dashed arrows: Control flow / feedback loops
- Thick arrows: Main pipeline
- Thin arrows: Internal processing
