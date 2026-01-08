import json
from collections import defaultdict

import json_repair

from scorer import navigate, get_node_id_from_position
from src.data_loader import get_data_by_instruction

json_rows = []

with open(f"ablation_study/json.jsonl", 'r') as f:
    for l in f.readlines():
        l = json.loads(l)
        try:
            for candidate in l['response']['candidates']:
                for part in candidate['content']['parts']:
                    if 'thought' not in part:
                        json_rows.append(
                            {"key": l['key'], **json_repair.repair_json(part['text'], return_objects=True)}
                        )
        except:
            continue

textual_rows = []
with open(f"ablation_study/textual.jsonl", 'r') as f:
    for l in f.readlines():
        l = json.loads(l)
        try:
            for candidate in l['response']['candidates']:
                for part in candidate['content']['parts']:
                    if 'thought' not in part:
                        textual_rows.append(
                            {"key": l['key'], **json_repair.repair_json(part['text'], return_objects=True)}
                        )
        except:
            continue

graph_vis = []
with open(f"ablation_study/graph_vis.jsonl", 'r') as f:
    for l in f.readlines():
        l = json.loads(l)
        try:
            for candidate in l['response']['candidates']:
                for part in candidate['content']['parts']:
                    if 'thought' not in part:
                        graph_vis.append(
                            {"key": l['key'], **json_repair.repair_json(part['text'], return_objects=True)}
                        )
        except:
            continue

grid_old = []
with open(f"ablation_study/grid.jsonl", 'r') as f:
    for l in f.readlines():
        l = json.loads(l)
        try:
            for candidate in l['response']['candidates']:
                for part in candidate['content']['parts']:
                    if 'thought' not in part:
                        grid_old.append(
                            {"key": l['key'], **json_repair.repair_json(part['text'], return_objects=True)}
                        )
        except:
            continue

grid = []
for idx, ni in enumerate(grid_old):
    split_file = "paper_results/test_seen.json"

    instruction_id = int(ni['key'])
    subgoal_status = ni['subplan_status']
    next_place = ni['next_place']

    area_data = get_data_by_instruction(
        instruction_id,
        split_file,
        base_path='./data/map2seq/',
        neighbor_degrees=20
    )

    instruction_data = area_data.get("instruction_data", {})

    # Get route information for initialization
    route = instruction_data.get("route", {})
    osm_path = route.get("osm_path", [])

    # Initialize at start node
    current_node_id = osm_path[0]
    current_heading = float(route.get("initial_heading", 0))
    previous_visited_path = []

    path_data = navigate(
        map_json=area_data,
        starting_point=current_node_id,
        heading=int(current_heading),
        pois=None,
        poi_mapping=None,
        units=1,
        last_instruction=False
    )


    next_node_id = get_node_id_from_position(path_data, previous_visited_path, next_place)
    grid.append(
        {"key": str(instruction_id), 'subplan_status': subgoal_status, 'next_place': next_node_id}
    )

# print(grid)

def index_by_key(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["key"]].append(r)
    return grouped

json_by_key = index_by_key(json_rows)
textual_by_key = index_by_key(textual_rows)
graph_by_key = index_by_key(graph_vis)
grid_by_key = index_by_key(grid)


all_keys = set(json_by_key) | set(textual_by_key) | set(graph_by_key) | set(grid_by_key)

comparisons = []

for key in all_keys:
    comparisons.append({
        "key": key,
        "json": json_by_key.get(key, []),
        "textual": textual_by_key.get(key, []),
        "graph_vis": graph_by_key.get(key, []),
        "grid": grid_by_key.get(key, []),
    })

for c in comparisons:
    print(f"\n=== Key: {c['key']} ===")
    print("JSON:", c["json"])
    print("TEXTUAL:", c["textual"])
    print("GRAPH:", c["graph_vis"])
    print("GRID:", c["grid"])
