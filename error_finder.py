import json
from typing import Dict

import json_repair


def load_navigation_instructions(file_path: str) -> list:
    """Load navigation instructions (sub_goals and landmarks) from predictions file."""
    list_ids = []
    with open(file_path) as f:
        for record in map(json.loads, f):
            try:
                list_ids.append(record['key'])
            except Exception as e:
                print(f"Error parsing instruction for key {record.get('key')}: {e}")
                continue
    return list_ids

mode='json'
list_ids = load_navigation_instructions(f'step2_{mode}.jsonl')

rows = []
with open(f'{mode}.jsonl', 'r') as f:
    for l in f.readlines():
        l = json.loads(l)
        if l['key'] not in list_ids:
            rows.append(l)

print(len(rows))
#
# with open(f'{mode}.jsonl', "w") as f:
#     for req in rows:
#         f.write(json.dumps(req) + "\n")