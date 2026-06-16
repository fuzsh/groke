import json
import math
import os
from collections import defaultdict, deque
from typing import Dict

import json_repair

from groke.process_results import resolve_grid_next_place
from groke.data_loader import get_data_by_instruction


def load_navigation_instructions(file_path: str) -> Dict[str, Dict]:
    """Load navigation instructions (sub_goals and landmarks) from predictions file."""
    instructions = {}
    with open(file_path) as f:
        for record in map(json.loads, f):
            try:
                for candidate in record['response']['candidates']:
                    for part in candidate['content']['parts']:
                        if 'thought' not in part:
                            parsed = json_repair.repair_json(part['text'], return_objects=True)
                            instructions[str(record['key'])] = {
                                "key": str(record['key']),
                                "sub_goals": parsed.get('sub_goals', []),
                                "landmarks": parsed.get('landmarks', [])
                            }
            except Exception as e:
                print(f"Error parsing instruction for key {record.get('key')}: {e}")
                continue
    return instructions


def load_method_results(file_path: str, method_name: str) -> Dict[str, Dict]:
    """Load results from a method's JSONL file."""
    results = {}
    with open(file_path) as f:
        for line in f:
            record = json.loads(line)
            try:
                for candidate in record['response']['candidates']:
                    for part in candidate['content']['parts']:
                        if 'thought' not in part:
                            parsed = json_repair.repair_json(part['text'], return_objects=True)
                            results[str(record['key'])] = {
                                "key": str(record['key']),
                                "subplan_status": parsed.get('subplan_status', 'IN_PROGRESS'),
                                "next_place": parsed.get('next_place'),
                                "method": method_name
                            }
            except Exception as e:
                continue
    return results


def _get_heading_to_node(area_links, area_nodes, start_node_id, end_node_id):
    # 1. Build the Graph (Adjacency List) from area_links
    graph = {}
    for link in area_links:
        src = link['source']
        tgt = link['target']
        if src not in graph:
            graph[src] = []
        graph[src].append(tgt)

    # 2. Find the Path (Breadth-First Search)
    # This finds the shortest sequence of nodes connecting Start to End
    queue = deque([[start_node_id]])
    visited = {start_node_id}
    found_path = None

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == end_node_id:
            found_path = path
            break

        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

    if not found_path:
        raise ValueError("Error: No path found connecting these nodes.")

    # 3. Get the Immediate Previous Node
    # The node just before the destination is at index -2
    if len(found_path) < 2:
        raise ValueError("Error: Start and End nodes are the same.")

    prev_node_id = found_path[-2]

    # 4. Calculate Heading (Bearing)
    nodes = area_nodes
    lat1 = math.radians(nodes[prev_node_id]['lat'])
    lon1 = math.radians(nodes[prev_node_id]['lng'])
    lat2 = math.radians(nodes[end_node_id]['lat'])
    lon2 = math.radians(nodes[end_node_id]['lng'])

    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    bearing = math.degrees(math.atan2(y, x))
    compass_bearing = (bearing + 360) % 360

    return found_path, compass_bearing



def process_results_for_method(method_name, results_file, source_file_results, split_file: str = "test_seen_200.json"):
    print(f"\n=== Processing method: {method_name} ===")

    # Load results for this method
    results = load_method_results(results_file, method_name)
    print(f"Loaded {len(results)} results")

    for key, result in results.items():
        nav_inst = source_file_results[key]
        sub_goals = nav_inst.get('sub_instructions', [])

        if not sub_goals:
            print(f"Warning: No sub_goals for key {key}")
            continue

        # Load area data
        try:
            area_data = get_data_by_instruction(
                int(key),
                split_file,
                base_path='./data/map2seq/',
                neighbor_degrees=20
            )
        except Exception as e:
            print(f"Error loading area data for key {key}: {e}")
            continue

        instruction_data = area_data.get("instruction_data", {})
        route = instruction_data.get("route", {})
        osm_path = route.get("osm_path", [])

        if not osm_path:
            print(f"Warning: No osm_path for key {key}")
            continue

        # Initial state
        predicated_path = nav_inst.get(method_name, {}).get('predicated_path', [])
        if not predicated_path:
            print(f"Warning: No predicated_path for key {key}")
            continue

        current_node_id = predicated_path[-1]
        current_heading = nav_inst.get(method_name, {}).get('current_heading', float(0))
        current_instruction_index = nav_inst.get(method_name, {}).get('current_instruction_index', 0)
        num_retry_current_instruction = nav_inst.get(method_name, {}).get('num_retry_current_instruction', 0)

        # Get result from first step
        subplan_status = result.get('subplan_status', 'IN_PROGRESS')
        next_place = result.get('next_place')

        # Resolve next_place to node_id if needed (for grid method)
        if method_name == 'grid' and isinstance(next_place, list):
            _, next_place = resolve_grid_next_place(
                next_place,
                area_data,
                current_node_id,
                current_heading,
                [] # TODO: NEED TO FIX
            )

        # Determine current sub-instruction index
        if subplan_status == 'COMPLETED':
            current_instruction_index += 1  # Move to next sub-goal
            num_retry_current_instruction = 0
        else:
            current_instruction_index += 0  # Stay on same sub-goal
            num_retry_current_instruction += 1 # Increase the retry count

        # Calculate new position and heading
        new_heading = 0
        if next_place and next_place in area_data['area_nodes']:
            try:
                _, new_heading = _get_heading_to_node(
                    area_links=area_data['area_links'],
                    area_nodes=area_data['area_nodes'],
                    start_node_id=current_node_id,
                    end_node_id=next_place
                )
            except ValueError as e:
                print(f"Error getting heading for key {key}: {e}")
                source_file_results[key][method_name] = {
                    "status": "DONE",
                    "predicated_path": predicated_path + [next_place],
                    "current_instruction_index": current_instruction_index,
                    "num_retry_current_instruction": num_retry_current_instruction,
                    "current_instruction_status": subplan_status,
                    "current_heading": 0,
                    "error": str(e)
                }
                continue

        if current_instruction_index >= len(sub_goals):
            print(f"Key {key}: Navigation completed (all sub-goals done)")
            source_file_results[key][method_name] = {
                "status": "DONE",
                "predicated_path": predicated_path + [next_place],
                "current_instruction_index": current_instruction_index,
                "num_retry_current_instruction": num_retry_current_instruction,
                "current_instruction_status": subplan_status,
                "current_heading": new_heading
            }
            continue

        if not next_place:
            source_file_results[key][method_name] = {
                "status": "DONE",
                "predicated_path": predicated_path + [next_place],
                "current_instruction_index": current_instruction_index,
                "num_retry_current_instruction": num_retry_current_instruction,
                "current_instruction_status": subplan_status,
                "current_heading": 0,
                "error": "Current node is None or not exists in the area_nodes"
            }
            continue

        source_file_results[key][method_name] = {
            "status": "PROGRESS",
            "predicated_path": predicated_path + [next_place],
            "current_instruction_index": current_instruction_index,
            "num_retry_current_instruction": num_retry_current_instruction,
            "current_instruction_status": subplan_status,
            "current_heading": new_heading
        }


def main(ttype):
    """Main entry point."""
    file_name = f'main_test_{ttype}.json'

    load_instructions = True
    load_first_step = True
    step_number = 12

    methods = {
        'json': 'main_results_{ttype}/step{step_number}_json.jsonl',
        # 'textual': 'ablation_study/step{step_number}_textual.jsonl',
        # 'graph_vis': 'ablation_study/step{step_number}_graph_vis.jsonl',
        # 'grid': 'ablation_study/step{step_number}_grid.jsonl'
    }

    # Load navigation instructions (sub_goals and landmarks)
    if load_instructions:
        nav_instructions_file = f'main_results/main_test_{ttype}.jsonl'
        print(f"Loading navigation instructions from {nav_instructions_file}...")
        navigation_instructions = load_navigation_instructions(nav_instructions_file)
        print(f"Loaded {len(navigation_instructions)} navigation instructions")

        map2seq = {}
        with open(f'data/map2seq/splits/test_{ttype}.json', 'r') as f:
            test_set = json.loads(f.read())
            for ts in test_set:
                map2seq[str(ts['instructions_id'])] = {
                    "osm_path": ts['route']['osm_path'],
                    "initial_headings": ts['route']['initial_heading'],
                    "full_instructions": ts['instructions']
                }

        results = {}
        for nav_ins_key, nav_ins in navigation_instructions.items():

            # if thinking_budget and thinking_budget in ['high', 'low']:
            #     sub_instructions = nav_ins['sub_goals']
            # else:
            # sub_instructions = [{
            #     "description": map2seq.get(nav_ins_key, {}).get('full_instructions'),
            #     "status": "TODO"
            # }]

            results[nav_ins_key] = {
                    "instruction_id": nav_ins_key,
                    "complexity":{
                        'cognitive': 0,
                        'spatial':0,
                        'execution': 0
                    },
                    "human_annotation":"",
                    "landmarks": nav_ins['landmarks'],
                    "sub_instructions": nav_ins['sub_goals'],
                    # "sub_instructions": [{
                    #     "description": f"{divided_ins.strip()}.",
                    #     "status": "TODO"
                    # } for divided_ins in map2seq.get(nav_ins_key, {}).get('full_instructions').split(".") if divided_ins.strip()],
                    # "sub_instructions": sub_instructions,
                    **map2seq.get(nav_ins_key, {}),
                    **{method: {
                        "status": "PROGRESS",
                        "predicated_path": [map2seq.get(nav_ins_key, {}).get('osm_path', ["NONE"])[0]],
                        "current_instruction_index": 0,
                        "num_retry_current_instruction": 0,
                        "current_instruction_status": "TODO",
                        "current_heading": float(map2seq.get(nav_ins_key, {}).get('initial_headings', 0))
                    } for method in methods.keys()}
            }


        with open(file_name, 'w') as f:
            f.write(json.dumps(results, indent=4))

    if load_first_step:
        with open(file_name, 'r') as f:
            source_file_results = json.loads(f.read())

        # # TODO: REMOVE THIS -- DO SHIT
        # for sf_id, sf in source_file_results.items():
        #     sf['json'] = {
        #         "status": "PROGRESS",
        #         "predicated_path": sf['json']['predicated_path'][0],
        #         "current_instruction_index": 0,
        #         "num_retry_current_instruction": 0,
        #         "current_instruction_status": "TODO",
        #         "current_heading": 0
        #     }

        for sn in range(1, step_number):
            # Process each method's results
            for method_name, results_file in methods.items():
                results_file = results_file.format(step_number=sn, ttype=ttype)

                # Check whether the file exists or not if exists or not
                if not os.path.exists(results_file):
                    print(f"File {results_file} not exists, continue...")
                    continue

                try:
                    process_results_for_method(
                        method_name=method_name,
                        results_file=results_file,
                        source_file_results=source_file_results,
                        split_file=f'test_{ttype}.json',
                    )
                except Exception as e:
                    print(f"Error processing method {method_name}: {e}")
                    import traceback
                    traceback.print_exc()

        with open(file_name, 'w') as f:
            f.write(json.dumps(source_file_results, indent=4))


if __name__ == "__main__":
    for i in ['seen', 'unseen']:
        main(ttype=i)
