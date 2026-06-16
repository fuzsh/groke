import json
import math
from collections import deque

from groke.scorer.graph_context import navigate
from groke.scorer.grid_representation import convert2grid
from groke.data_loader import get_data_by_instruction
from rapidfuzz import fuzz

from groke.visualize import visualize_area

instruction_id = 4742
split_file = "paper_results/test_seen.json"

# Get the data, expanding the neighborhood by 20 degrees
area_data = get_data_by_instruction(
    instruction_id,
    split_file,
    base_path='/Users/fuzzy/Projects/sisu/data/map2seq/',
    neighbor_degrees=20
)

print(area_data['instruction_data']['instructions'])

# print(area_data)

visualize_area(area_data, focused_node_id=[
    "7227e68fd7b64cd4ad4232b38550cc19",
                "7362d8cb9a3e421c93294dd231111ba3",
                "c2a549516468432bb0fe9f8b7d3b9f2b",
                "42432194",
                "53ebded83d984acfa33e19d32ab3102e",
                "f57523382da84d69999b370afd0687ab",
                "4c20c8074b35482484fbf1351265b341"
])

# with open('test_seen_json_dump.json', 'r') as f:
#     data = json.load(f)
#
# new_data = {}
# for d, d_d in data.items():
#     new_data[d] = {
#         "complexity": d_d["complexity"],
#         "human_annotation": d_d["human_annotation"],
#     }
#
# with open("correctness_hardness.json", "w") as f:
#     f.write(json.dumps(new_data, indent=4))

# with open('test_seen.json', 'r') as f:
#     results = json.loads(f.read())
# results_2 = {}
# for r, rd in results.items():
#     print(r)
#     if int(r) in [11476
#         , 4077
#         , 1624
#         , 4130
#         , 4117
#         , 11574
#         , 5647
#         , 7331
#         , 4755
#         , 11197
#         , 11247
#         , 11844
#         , 2627
#         , 6448
#         , 6165
#         , 11790
#         , 1993
#         , 2478
#         , 10697
#         , 4224
#         , 6415
#         , 2329
#         , 11868
#         , 6220
#         , 6199
#         , 5261
#         , 7364
#         , 10347
#         , 4291
#         , 12751
#         , 10426
#         , 11070
#         , 1994
#         , 3450
#         , 11604
#         , 3782
#         , 3578
#                   ]:
#         print('hi')
#         results_2[r] = {
#             "instruction_id": rd['instruction_id'],
#             "landmarks":rd['landmarks'],
#             "full_instructions":rd['full_instructions'],
#             "sub_instructions":rd['sub_instructions']
#         }
#
# with open("error_analysis.json", 'w') as f:
#     f.write(json.dumps(results_2, indent=4))

#
#
landmarks = [
    {
        "name": "traffic light",
        "category": "Traffic Control",
        "relative_position": "ahead"
    },
    {
        "name": "Luzzo",
        "category": "Amenities",
        "relative_position": "on the corner"
    }
]
#
# available_pois = {}
#
# for node_id, info in area_data['area_pois'].items():
#     tags = json.loads(info["tags"])
#     print(tags)
#     name = (
#         f"{tags.get('name', '')} "
#         f"{tags.get('amenity', '').replace('_', ' ')} "
#         f"{tags.get('cuisine', '').replace('_', ' ')} "
#         f"{tags.get('leisure', '').replace('_', ' ')} "
#         f"{tags.get('tourism', '').replace('_', ' ')} "
#         f"{tags.get('shop', '').replace('_', ' ')} "
#         f"{tags.get('highway', '').replace('_', ' ')} "
#     ).strip()
#
#     for landmark in landmarks:
#         score = fuzz.partial_ratio(landmark['name'], name)
#         if score > 70:
#             available_pois.setdefault(landmark['name'], []).append(node_id)
#
# print(available_pois)
#
# # {
# #   "bike rentals": [
# #     "3708656264",
# #     "3708656265",
# #     "3708656226"
# #   ],
# #   "Cafe au bon gout": [
# #     "584120700"
# #   ],
# #   "Chipotle": [
# #     "3966939351"
# #   ]
# # }
#
# current_sub_instructions_landmarks = {}
#
# for lia_name, lia_node_id in available_pois.items():
#     if lia_name in 'Chipotle on the corner':
#         current_sub_instructions_landmarks[lia_name] = lia_node_id
#
# # {'Chipotle': ['3966939351']}
#
#
# used = set()
# mapping = {}
#
# # Characters that are disallowed
# forbidden = {"S", "P"}
#
# # Fallback alphabet
# alphabet = [chr(i) for i in range(ord("A"), ord("Z") + 1) if chr(i) not in forbidden]
#
# for name in current_sub_instructions_landmarks:
#     assigned = None
#
#     # Try characters from the name first
#     for ch in name:
#         upper = ch.upper()
#         if ch.isalpha() and upper not in forbidden and upper not in used:
#             assigned = upper
#             used.add(upper)
#             break
#
#     # Fallback if needed
#     if assigned is None:
#         for c in alphabet:
#             if c not in used:
#                 assigned = c
#                 used.add(c)
#                 break
#
#     mapping[name] = assigned
#
# # Format outputs
# landmark_keys = ", ".join(mapping.keys())  # Chipotle
# assignments = ", ".join(f"{char} = {name} (landmark)" for name, char in mapping.items())  # C = Chipotle (landmark)
#
#
# start_node = "813f2409328d4b95937bd35ba8cc6860"
# next_node = "5e7d79f2f1b04e3a95d55090c32831bb"
#
#
# prev_path_data = navigate(
#     map_json=area_data,
#     starting_point=start_node,
#     heading=299,
#     pois=current_sub_instructions_landmarks,
#     poi_mapping=mapping,
#     units=1,
#     last_instruction=False
# )
#
# path_data = navigate(
#     map_json=area_data,
#     starting_point=next_node,
#     heading=299,
#     pois=current_sub_instructions_landmarks,
#     poi_mapping=mapping,
#     units=1,
#     last_instruction=False
# )
#
#
# def _get_heading_to_node(area_links, area_nodes, start_node_id, end_node_id):
#     # 1. Build the Graph (Adjacency List) from area_links
#     graph = {}
#     for link in area_links:
#         src = link['source']
#         tgt = link['target']
#         if src not in graph:
#             graph[src] = []
#         graph[src].append(tgt)
#
#     # 2. Find the Path (Breadth-First Search)
#     # This finds the shortest sequence of nodes connecting Start to End
#     queue = deque([[start_node_id]])
#     visited = {start_node_id}
#     found_path = None
#
#     while queue:
#         path = queue.popleft()
#         current = path[-1]
#
#         if current == end_node_id:
#             found_path = path
#             break
#
#         if current in graph:
#             for neighbor in graph[current]:
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     new_path = list(path)
#                     new_path.append(neighbor)
#                     queue.append(new_path)
#
#     if not found_path:
#         return "Error: No path found connecting these nodes."
#
#     # 3. Get the Immediate Previous Node
#     # The node just before the destination is at index -2
#     if len(found_path) < 2:
#         return "Error: Start and End nodes are the same."
#
#     prev_node_id = found_path[-2]
#
#     # 4. Calculate Heading (Bearing)
#     nodes = area_nodes
#     lat1 = math.radians(nodes[prev_node_id]['lat'])
#     lon1 = math.radians(nodes[prev_node_id]['lng'])
#     lat2 = math.radians(nodes[end_node_id]['lat'])
#     lon2 = math.radians(nodes[end_node_id]['lng'])
#
#     dLon = lon2 - lon1
#     y = math.sin(dLon) * math.cos(lat2)
#     x = math.cos(lat1) * math.sin(lat2) - \
#         math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
#
#     bearing = math.degrees(math.atan2(y, x))
#     compass_bearing = (bearing + 360) % 360
#
#     print(found_path)
#
#     return found_path, compass_bearing
#
# def _get_previous_nodes(path_data, previous_visited):
#     new_path_data = []
#
#     for pd in path_data:
#         if pd['node_id'] not in previous_visited:
#             continue
#
#         new_connectivity = []
#         connectivity = pd.get('connectivity', {})
#         if connectivity:
#             for con in connectivity:
#                 if con['node_id'] in previous_visited:
#                     new_connectivity.append(con)
#         pd['connectivity'] = new_connectivity
#         new_path_data.append(pd)
#
#     return new_path_data
#
# visited_route, new_heading = _get_heading_to_node(area_data['area_links'], area_data['area_nodes'], start_node, next_node)
# previous_visited_path = []
#
# if visited_route:
#     previous_visited_path = _get_previous_nodes(prev_path_data, visited_route)
#
# print("\n".join(map(str, convert2grid(
#     path_data,
#     previous_visited_path,
#     area_nodes=area_data['area_nodes'],
#     area_pois=area_data['area_pois'],
#     pois=current_sub_instructions_landmarks,
#     poi_mapping=mapping
# ).tolist())))

# path_data = navigate(area_data, '06b04bf7308147cd8eaf533f27d11fae', 6)
# print(path_data)
# # path_data = [
# #     {'node_id': '12f93e32fb3c43498de62fa1985809a3', 'direction': 138, 'is_intersection': False, 'connectivity': [{'node_id': 'e97131e4d0c04571b7e525a5ba2d22b6', 'heading': 118.67988075413018, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': 'e97131e4d0c04571b7e525a5ba2d22b6', 'direction': 118.67988075413018, 'is_intersection': False, 'connectivity': [{'node_id': '9f77815d02fa4de397e087883ab2e51b', 'heading': 118.67994715087048, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': '9f77815d02fa4de397e087883ab2e51b', 'direction': 118.67994715087048, 'is_intersection': False, 'connectivity': [{'node_id': 'e437e1132c4943c9b7a1d1594fae73fc', 'heading': 118.68001353375695, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': 'e437e1132c4943c9b7a1d1594fae73fc', 'direction': 118.68001353375695, 'is_intersection': False, 'connectivity': [{'node_id': 'd0e80d8a90f04683989f3e5766654289', 'heading': 118.68007992391279, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': 'd0e80d8a90f04683989f3e5766654289', 'direction': 118.68007992391279, 'is_intersection': False, 'connectivity': [{'node_id': 'fe066372c2854e4791cc7b90790ac9ab', 'heading': 118.68014630865451, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': 'fe066372c2854e4791cc7b90790ac9ab', 'direction': 118.68014630865451, 'is_intersection': False, 'connectivity': [{'node_id': '46f53823fe2f4a3b98530d14c6d60d7f', 'heading': 118.68021270667316, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': '46f53823fe2f4a3b98530d14c6d60d7f', 'direction': 118.68021270667316, 'is_intersection': False, 'connectivity': [{'node_id': '62a0509e817a46df950e724d568c2eaa', 'heading': 118.68027908120655, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': '62a0509e817a46df950e724d568c2eaa', 'direction': 118.68027908120655, 'is_intersection': False, 'connectivity': [{'node_id': '9f80f354ba024c8c974daf591e0df4ec', 'heading': 118.68034547564469, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': '9f80f354ba024c8c974daf591e0df4ec', 'direction': 118.68034547564469, 'is_intersection': False, 'connectivity': [{'node_id': '07001decc9664b1d86980daf3310d816', 'heading': 118.68041186772052, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': '07001decc9664b1d86980daf3310d816', 'direction': 118.68041186772052, 'is_intersection': False, 'connectivity': [{'node_id': '42434800', 'heading': 118.68047825075843, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': '42434800', 'direction': 118.68047825075843, 'is_intersection': True, 'connectivity': [{'node_id': '7e751156672141ce865170eae308d1c8', 'heading': 119.1695009838295, 'direction': 'Forward'}, {'node_id': 'bd1c4c30a2b64fa68669c445886a9fa1', 'heading': 28.85276439327589, 'direction': 'Left'}, {'node_id': 'bda6d7f243b14fa4b73d39ff978e06d0', 'heading': 208.8958370857973, 'direction': 'Right'}], 'poi': 'traffic_signals', 'side_pois': []},
# #     {'node_id': '7e751156672141ce865170eae308d1c8', 'direction': 119.1695009838295, 'is_intersection': False, 'connectivity': [{'node_id': 'f4338a5260de4f0fb5b5f9730793d7c5', 'heading': 119.16957024057763, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': 'f4338a5260de4f0fb5b5f9730793d7c5', 'direction': 119.16957024057763, 'is_intersection': False, 'connectivity': [{'node_id': 'e2bc1259b7d040b28ef62e4d60f43a21', 'heading': 119.16963952487049, 'direction': 'Forward'}], 'poi': None, 'side_pois': []}
# # ]
# #
# # previous_visited = [
# #     {'node_id': '06b04bf7308147cd8eaf533f27d11fae', 'direction': 6, 'is_intersection': False, 'connectivity': [{'node_id': '3de44976bb314166829cebf925095a71', 'heading': 8.302398692648637, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': '3de44976bb314166829cebf925095a71', 'direction': 8.302398692648637, 'is_intersection': False, 'connectivity': [{'node_id': '42428220', 'heading': 8.30241154767657, 'direction': 'Forward'}], 'poi': None, 'side_pois': []},
# #     {'node_id': '42428220', 'direction': 8.30241154767657, 'is_intersection': True, 'connectivity': [{'node_id': '12f93e32fb3c43498de62fa1985809a3', 'heading': 118.67981435922349, 'direction': 'Right'}, {'node_id': 'f8c70d70b9a84755bfebf238c165ecd5', 'heading': 8.01785448296448, 'direction': 'Forward'}, {'node_id': '1a397f3b1fb94a5d9fdaa861c485e21e', 'heading': 299.1773000863904, 'direction': 'Left'}], 'poi': 'traffic_signals', 'side_pois': []},
# # ]
# grid_matrix = convert2grid(path_data, [])
#
# print("\n".join(map(str, convert2grid(path_data, []).tolist())))
# print(grid_matrix)
#
# print(get_node_id_from_position(path_data, [], [4, 3]))
#
#
# def _get_heading_to_node(area_links, area_nodes, start_node_id, end_node_id):
#     # 1. Build the Graph (Adjacency List) from area_links
#     graph = {}
#     for link in area_links:
#         src = link['source']
#         tgt = link['target']
#         if src not in graph:
#             graph[src] = []
#         graph[src].append(tgt)
#
#     # 2. Find the Path (Breadth-First Search)
#     # This finds the shortest sequence of nodes connecting Start to End
#     queue = deque([[start_node_id]])
#     visited = {start_node_id}
#     found_path = None
#
#     while queue:
#         path = queue.popleft()
#         current = path[-1]
#
#         if current == end_node_id:
#             found_path = path
#             break
#
#         if current in graph:
#             for neighbor in graph[current]:
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     new_path = list(path)
#                     new_path.append(neighbor)
#                     queue.append(new_path)
#
#     if not found_path:
#         return "Error: No path found connecting these nodes."
#
#     # 3. Get the Immediate Previous Node
#     # The node just before the destination is at index -2
#     if len(found_path) < 2:
#         return "Error: Start and End nodes are the same."
#
#     prev_node_id = found_path[-2]
#
#     # 4. Calculate Heading (Bearing)
#     nodes = area_nodes
#     lat1 = math.radians(nodes[prev_node_id]['lat'])
#     lon1 = math.radians(nodes[prev_node_id]['lng'])
#     lat2 = math.radians(nodes[end_node_id]['lat'])
#     lon2 = math.radians(nodes[end_node_id]['lng'])
#
#     dLon = lon2 - lon1
#     y = math.sin(dLon) * math.cos(lat2)
#     x = math.cos(lat1) * math.sin(lat2) - \
#         math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
#
#     bearing = math.degrees(math.atan2(y, x))
#     compass_bearing = (bearing + 360) % 360
#
#     print(found_path)
#
#     return {
#         "calculated_route": found_path,
#         "previous_node": prev_node_id,
#     }, compass_bearing
#
#
# print(_get_heading_to_node(area_data['area_links'], area_data['area_nodes'], '06b04bf7308147cd8eaf533f27d11fae',
#                            '12f93e32fb3c43498de62fa1985809a3'))
#
# #
# instruction_id = 6220
# split_file = "test_seen.json"
#
# # Get the data, expanding the neighborhood by 20 degrees
# area_data = get_data_by_instruction(
#     instruction_id,
#     split_file,
#     base_path='/Users/fuzzy/Projects/sisu/data/map2seq/',
#     neighbor_degrees=20
# )
#
# # Visualize the data
# if area_data:
#     print(f"Data for instruction {instruction_id}:")
#     print(f"  Route length: {len(area_data['instruction_data']['route']['osm_path'])}")
#     print(f"  Number of nodes in area: {len(area_data['area_nodes'])}")
#     print(f"  Number of links in area: {len(area_data['area_links'])}")
#     print(f"  Number of POIs in area: {len(area_data['area_pois'])}")
#     visualize_area(area_data)
