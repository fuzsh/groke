import json
import random

import json_repair

from src.data_loader import get_data_by_instruction
from templates import navigation_divider_batch

# NUM_ITEMS = 1000
# SPLIT_FILE = "test_unseen.json"
# # SELECTED_FILE = "test_seen_100.json"
# OUTPUT_FILE = "main_test_unseen.jsonl"
#
# # Load, shuffle, and select 100 items
# with open(f"data/map2seq/splits/{SPLIT_FILE}", 'r') as f:
#     data = json.load(f)
#
# # Random shuffle the data
# random.shuffle(data)
#
# # Select 100 items
# selected_data = data[:NUM_ITEMS]
#
# # # Save selected items to separate file
# # with open(f"data/map2seq/splits/{SELECTED_FILE}", 'w') as f:
# #     json.dump(selected_data, f, indent=2)
#
# # Create navigation instructions from selected data
# requests = []
# list_instructions = {}
#
# for d in selected_data:
#     requests.append(navigation_divider_batch(d['instructions_id'], d['instructions']))
#
# with open(OUTPUT_FILE, "w") as f:
#     for req in requests:
#         f.write(json.dumps(req) + "\n")

rows = []
with open(f"main_results/main_test_seen.jsonl", 'r') as f:
    for l in f.readlines():
        rows.append(json.loads(l))

navigation_instruction_data = []

x = set()
f = set()
for r in rows:
    x.add(r['key'])
    for candidate in r['response']['candidates']:
        for part in candidate['content']['parts']:
            if 'thought' not in part:
                navigation_instruction_data.append(
                    {"key": r['key'], **json_repair.repair_json(part['text'], return_objects=True)}
                )
                f.add(r['key'])

diff = x.difference(f)
print(diff)
print(len(navigation_instruction_data))
#
# with open(f'main_test_seen.jsonl', 'w') as f:
#     for req in navigation_instruction_data:
#         f.write(json.dumps(req) + "\n")
#
# # instruction_data = area_data.get("instruction_data", {})
# # navigation_instruction = instruction_data.get("instructions", "")
#
#
# # TODO:
# #  1. Based on the parsed instruction create next steps -- complete logic to generate the code
# #  2. Complete the steps before hand (stored file) ...
# #  3. Create parser and generate the next step files ...
