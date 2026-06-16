# GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap

---

## 1. System Overview

The system implements a **Multi-Agent Agentic Navigation Framework** for route navigability on OpenStreetMap (OSM) graph data.

![plot](./assets/overall_architecture.png)

---

## 2. Reproducibility

### 2.0. Installation

```bash
# Clone the repository
git clone <repository-url>
cd groke

# Install the package (editable) — this puts `groke` on the import path
pip install -e .

# For the exact, fully-pinned environment used to produce the paper results:
pip install -r requirements.txt
```

All Python modules live in the importable `groke/` package and are run with
`python -m groke.<module>` (see commands below). `pip install -e .` makes those
imports resolve from any working directory; run the commands from the repo root
so the relative data paths (`data/`, `predictions/`, `annotations/`, …) resolve.

**Key Dependencies:**
- `google-adk` - Google Agent Development Kit for multi-agent orchestration
- `google-genai` - Google Generative AI SDK
- `rapidfuzz` - Fuzzy string matching for POI detection
- `matplotlib`, `numpy`, `scipy` - Visualization and analysis
- `networkx` - Graph operations

### 2.0.1. Project Structure

```
groke/                  # importable package (pip install -e .)
├── prompts.py          # LLM prompt strings
├── templates.py        # batch-request builders
├── data_loader.py      # dataset access (get_data_by_instruction)
├── visualize.py        # matplotlib area visualization
├── data_manager.py     # orchestrates per-step data prep
├── step_generator.py   # builds per-step navigation prompts
├── process_results.py  # parses agent outputs into next-step state
├── scorer/             # graph context, grid representation, POI extraction, agents
├── agents/             # pipeline entry points (run as scripts)
│   ├── first_agent.py      # instruction divider
│   ├── second_agent.py     # navigator
│   ├── multi_format_agent.py
│   └── runner.py           # single-instruction ADK runner (was main.py)
└── evaluation/
    ├── metrics.py          # navigation metrics (was evaluation_metrics.py)
    ├── baseline.py         # baseline evaluator
    └── usage_metadata.py   # token/cost analysis

analysis/               # standalone analysis & plotting scripts
scripts/                # operational utilities (download_dataset.sh, batch.py, merge.py)
data/                   # map2seq dataset (OSM graph + splits)
generated_prompts/      # generated step{N}_{method}.jsonl prompts
predictions/            # model prediction outputs
annotations/            # human/auto annotation files (correctness, POI, errors)
paper_results/          # step-by-step results behind the paper
evaluation_results/     # computed metrics + plots
assets/                 # figures used in this README
scratch/                # throwaway / exploratory scripts (not part of the pipeline)
```

### 2.1. Instruction Generation (`groke/agents/first_agent.py`)

The first agent parses navigation instructions from raw text and generates sub-goals and landmarks.

**Input:** Raw navigation instructions from Map2Seq dataset
**Output:** Parsed sub-goals and identified landmarks in JSONL format

```bash
# Process navigation instructions
python -m groke.agents.first_agent
```

This reads from `main_results/main_test_seen.jsonl` and outputs parsed navigation data with:
- `sub_goals`: List of sequential navigation sub-instructions
- `landmarks`: Identified POIs mentioned in the instructions

### 2.2. Step Generator (`groke/step_generator.py`)

Generates navigation prompts for each step based on the current agent state. Supports multiple representation formats for ablation studies.

**Usage:**
```bash
python -m groke.step_generator --input main_test_seen.json --output-dir main_prompt_seen --step <STEP_NUMBER>
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--input, -i` | Input state file (JSON) | `main_test_seen.json` |
| `--output-dir, -o` | Output directory for prompts | `main_prompt_seen` |
| `--step, -s` | Step number (required) | - |
| `--split-file` | Data split file | `test_seen.json` |
| `--methods, -m` | Methods to generate: `json`, `textual`, `grid`, `graph_vis` | all |


**Example - Generate step 2 prompts for JSON representation:**
```bash
python -m groke.step_generator -i main_test_seen.json -o main_prompt_seen -s 2 -m json
```

**Output:** JSONL files per method: `step{N}_{method}.jsonl`

**Representation Formats:**
- `json`: Structured JSON navigation context
- `textual`: Natural language description
- `grid`: 2D grid matrix representation
- `graph_vis`: GraphViz DOT format

### 2.3. Evaluation

#### 2.3.1. Running Evaluation (`groke/evaluation/metrics.py`)

Computes navigation metrics comparing predicted paths against ground truth OSM paths.

**Usage:**
```bash
python -m groke.evaluation.metrics --input <predictions.json> --output <output_dir> --split-file <split.json>
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--input, -i` | Predictions JSON file | `test_seen_divider.json` |
| `--output, -o` | Output directory | `evaluation_results` |
| `--split-file` | Data split file | `test_seen_200.json` |
| `--threshold` | Distance threshold (meters) | `25` |
| `--difficulty-file` | Optional difficulty assessment file | `None` |

**Example - Evaluate on test_seen split:**
```bash
python -m groke.evaluation.metrics \
    --input paper_results/main_test_seen.json \
    --output evaluation_results/seen \
    --split-file test_seen.json \
    --difficulty-file annotations/correctness_hardness.json
```

**Output Files:**
```
evaluation_results/
├── detailed_results.json          # Aggregated metrics
├── individual_results.json        # Per-instruction results
├── results_by_difficulty.json     # Difficulty-based analysis
├── correlation_results.json       # Human annotation correlations
├── correctness_by_subgoals.png    # Bar chart by sub-goal count
├── endpoint_distance_distribution.png
├── navigation_metrics.png         # NE, SR, OSR, SDTW plots
├── accuracy_by_threshold.png      # 25m, 50m, 100m, 150m
└── upset_plot_*.png               # Method agreement patterns
```

#### 2.3.2. Performance Metrics

The evaluation computes standard VLN (Vision-and-Language Navigation) metrics:

| Metric | Description |
|--------|-------------|
| **NE** (Navigation Error) | Distance (m) between predicted endpoint and goal |
| **SR** (Success Rate) | % predictions within 25m of goal |
| **OSR** (Oracle Success Rate) | % paths passing within 25m of goal at any point |
| **SDTW** (Success-weighted DTW) | Path similarity weighted by success |
| **nDTW** (Normalized DTW) | Normalized Dynamic Time Warping distance |

**Path Overlap Metrics:**
- Overlap Ratio: Intersection / Union of path nodes
- F1 Score: Harmonic mean of precision and recall
- Sequence Overlap: Longest common subsequence ratio

#### 2.3.3. Computational Cost Analysis (`groke/evaluation/usage_metadata.py`)

Analyzes token usage and computational costs from inference results.

**Usage:**
```bash
python -m groke.evaluation.usage_metadata <folder_path> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `folder` | Path(s) to folder(s) containing JSONL results |
| `-v, --verbose` | Show per-instruction details |
| `-o, --output` | Export results to JSON file |
| `-c, --compare` | Compare statistics across multiple folders |

**Examples:**
```bash
# Analyze single experiment
python -m groke.evaluation.usage_metadata paper_results/main_results_seen

# Compare multiple experiments
python -m groke.evaluation.usage_metadata paper_results/ablation_study_json \
    paper_results/ablation_study_low \
    paper_results/ablation_study_high \
    --compare

# Export to JSON
python -m groke.evaluation.usage_metadata paper_results/main_results_seen -o token_analysis.json
```

**Output Statistics:**
- Steps per instruction (mean, median, min, max)
- Token counts: prompt, candidates, thoughts, total
- Percentiles: P25, P75, P90, P95
- Per-step breakdown

**Sample Output:**
```
============================================================
SUMMARY STATISTICS (per instruction totals)
============================================================

Number of instructions: 200

Steps per instruction:
  Mean:   4.32
  Median: 4
  Min:    1
  Max:    10

Total Tokens (per instruction):
  Sum:    2,450,000
  Mean:   12,250
  Median: 11,500
```

### 2.4. Running the Full Pipeline

**Step 1: Generate sub-instructions from raw navigation text**
```bash
python -m groke.agents.first_agent
```

**Step 2: Generate navigation prompts for each step**
```bash
# For test_seen split
# For step in 1 2 3 4 5 6 7 8 9 10; do
python -m groke.data_manager 
python -m groke.step_generator -i main_test_seen.json -o main_prompt_seen -s $step
#
```

**Step 3: Run the navigation agent (single instruction)**
```bash
python -m groke.agents.runner
```

**Step 4: Process results and generate next steps**
```bash
python -m groke.process_results
```

**Step 5: Evaluate predictions**
```bash
python -m groke.evaluation.metrics \
    --input paper_results/main_test_seen.json \
    --output evaluation_results \
    --split-file test_seen.json
```

**Step 6: Analyze computational costs**
```bash
python -m groke.evaluation.usage_metadata paper_results/main_results_seen -v
```

### 2.5. Data Structure

```
data/
└── map2seq/
    ├── osm/                    # OpenStreetMap area data
    │   ├── graph/
    │   │   ├── nodes.txt       # OSM nodes with coordinates
    │   │   ├── links.txt       # OSM way connections
    │   │   ├── pois.txt        # Points of interest
    │   │   └── poi_links.txt   # POI-to-node associations
    │   ├── command.txt
    │   └── map2seq_new_york-171204.osm
    ├── splits/
    │   ├── test_seen.json      # Test split (seen areas)
    │   ├── test_unseen.json    # Test split (unseen areas)
    │   ├── train.json / val.json
    │   └── *_200.json          # 200-item evaluation subsets
    └── readme.txt

paper_results/
├── main_test_seen.json         # Prediction state file
├── main_results_seen/          # Step-by-step results
│   ├── step1_json.jsonl
│   ├── step2_json.jsonl
│   └── ...
└── ablation_study_*/           # Ablation experiments
```

> **Known limitations.** Some pipeline scripts still carry data-path literals
> from an earlier directory layout that no longer exists in this snapshot —
> e.g. `groke/agents/first_agent.py` reads `main_results/main_test_seen.jsonl`,
> and several scripts reference an `ablation_study/` directory. The experiment
> data now lives under `paper_results/` (e.g. `paper_results/main_test_seen.json`,
> `paper_results/ablation_study_*/`). Point these scripts at the corresponding
> `paper_results/` paths before running them. Module imports, by contrast, are
> fully wired through the `groke` package and work after `pip install -e .`.



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
