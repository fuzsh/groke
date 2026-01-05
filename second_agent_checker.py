import json

import json_repair

from scorer.graph_context import navigate
from scorer.grid_repsentation import get_node_id_from_position
from src.data_loader import get_data_by_instruction
from src.visualize import visualize_area

rows = []
with open(f"predications/test_seen_navigator_1.jsonl", 'r') as f:
    for l in f.readlines():
        rows.append(json.loads(l))

navigation_instructions = []

for r in rows:
    for candidate in r['response']['candidates']:
        for part in candidate['content']['parts']:
            if 'thought' not in part:
                navigation_instructions.append(
                    {"key": r['key'], **json_repair.repair_json(part['text'], return_objects=True)}
                )

wrong_outputs = 0
for idx, ni in enumerate(navigation_instructions):
    split_file = "test_seen.json"

    instruction_id = int(ni['key'])
    if instruction_id != 2184:
        continue
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
    print(next_node_id)

    # Visualize the data
    if area_data:
        print(f"Data for instruction {instruction_id}:")
        print(f"  Route length: {len(area_data['instruction_data']['route']['osm_path'])}")
        print(f"  Number of nodes in area: {len(area_data['area_nodes'])}")
        print(f"  Number of links in area: {len(area_data['area_links'])}")
        print(f"  Number of POIs in area: {len(area_data['area_pois'])}")
        visualize_area(area_data, [next_node_id])


    if next_node_id not in osm_path:
        print(f"instruction id: {instruction_id} --> provides a wrong next node")
        wrong_outputs += 1

print(f'wrong outputs: {wrong_outputs}')


