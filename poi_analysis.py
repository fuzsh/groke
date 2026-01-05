# import json
#
# from gliner import GLiNER
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForTokenClassification
#
# model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
#
# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
# bert_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
#
# bert_ner_model = pipeline("ner", model=bert_ner, tokenizer=tokenizer)
#
#
# with open("ablation_study/test_seen_100.json", 'r') as f:
#     instructions = json.loads(f.read())
#
# # Label mapping based on your description
# label_map = {
#     "ORG": "Organization",
#     "LOC": "Location",
#     "PER": "Person",
#     "MIS": "Miscellaneous",
#     "O": "Outside"
# }
#
# landmarks_gliner = {}
# landmarks_bert = {}
# for instruction in instructions:
#     instruction_text = instruction['instructions']
#
#     labels = ["amenity", "cuisine", "leisure", "tourism", "shop", "highway", "transportation"]
#     entities = model.predict_entities(instruction_text, labels)
#     landmarks_gliner[instruction['instructions_id']] = [{
#         'name': entity['text'], 'category': entity['label'], 'score': entity['score']
#     } for entity in entities]
#
#     ner_results = bert_ner_model(instruction_text)
#
#     bert_entities = []
#     current_entity = None
#
#     for token in ner_results:
#         entity_tag = token['entity']
#         word = token['word']
#         score = float(token['score'])  # Convert numpy float to standard float
#
#         # Extract type (e.g., ORG from B-ORG)
#         if '-' in entity_tag:
#             prefix, tag_type = entity_tag.split('-')
#         else:
#             prefix, tag_type = "O", "O"
#
#         # Logic to combine tokens
#         if prefix == 'B':
#             # Save the previous entity if it exists
#             if current_entity:
#                 bert_entities.append(current_entity)
#
#             # Start a new entity
#             current_entity = {
#                 'name': word,
#                 'category': label_map.get(tag_type, tag_type),
#                 'score': score
#             }
#
#         elif prefix == 'I' and current_entity:
#             # Check if this token belongs to the current entity type
#             if label_map.get(tag_type, tag_type) == current_entity['category']:
#                 # Handle subwords (tokens starting with ##)
#                 if word.startswith('##'):
#                     current_entity['name'] += word[2:]  # Remove ## and join
#                 else:
#                     current_entity['name'] += " " + word  # Add space for new word
#
#                 # Update score (optional: take the average or keep the highest)
#                 current_entity['score'] = (current_entity['score'] + score) / 2
#
#     # Don't forget to append the last entity found after the loop finishes
#     if current_entity:
#         bert_entities.append(current_entity)
#
#     # Store in landmarks_bert instead of overwriting gliner
#     landmarks_bert[instruction['instructions_id']] = bert_entities
#
#
# with open("poi_analysis_gliner.json", 'w') as f:
#     f.write(json.dumps(landmarks_gliner, indent=4))
#
# with open("poi_analysis_bert.json", 'w') as f:
#     f.write(json.dumps(landmarks_bert, indent=4))
import json

import json_repair

landmarks_gliner = {}
landmarks_bert = {}
landmarks_gemmeni3 = {}

with open("poi_analysis_gliner.json", 'r') as f:
    landmarks_gliner = json.loads(f.read())

with open("poi_analysis_bert.json", 'r') as f:
    landmarks_bert = json.loads(f.read())

def load_navigation_instructions(file_path: str):
    """Load navigation instructions (sub_goals and landmarks) from predictions file."""
    instructions = {}
    with open(file_path) as f:
        for record in map(json.loads, f):
            try:
                for candidate in record['response']['candidates']:
                    for part in candidate['content']['parts']:
                        if 'thought' not in part:
                            parsed = json_repair.repair_json(part['text'], return_objects=True)
                            instructions[str(record['key'])] = parsed.get('landmarks', [])
            except Exception as e:
                print(f"Error parsing instruction for key {record.get('key')}: {e}")
                continue
    return instructions

landmarks_gemmeni3 = load_navigation_instructions("ablation_study/test_seen_200_predictions.jsonl")

with open("ablation_study/test_seen_100.json", 'r') as f:
    all_instruction = json.loads(f.read())

kossher = {}
for ai in all_instruction:
    kossher[ai['instructions_id']] = ai['instructions']


# Helper function to normalize names for comparison (case-insensitive)
def get_poi_names(poi_list):
    names = set()
    for item in poi_list:
        # Handle different potential keys (e.g. 'name', 'landmark', 'text')
        name = item.get('name') or item.get('landmark') or item.get('text')
        if name:
            names.add(name.strip().lower())
    return names


# 1. Find the Intersection of keys (keys present in ALL three)
common_keys = set(landmarks_gliner.keys()) & set(landmarks_bert.keys()) & set(landmarks_gemmeni3.keys())

print(f"Found {len(common_keys)} common instructions across all three models.\n")

# 2. Iterate through common keys and compare
for key in sorted(common_keys):
    print(f"--- Instruction ID: {key}: {kossher.get(int(key))}")

    # Extract sets of names found by each model
    gliner_pois = get_poi_names(landmarks_gliner[key])
    bert_pois = get_poi_names(landmarks_bert[key])
    gemini_pois = get_poi_names(landmarks_gemmeni3[key])

    # Create the "Master Set" (Union of all unique POIs found by any method)
    # This represents everything 'detected' for this instruction
    all_unique_pois = gliner_pois | bert_pois | gemini_pois

    # Calculate what is missing for each (Master Set - Model Set)
    missing_in_gliner = all_unique_pois - gliner_pois
    missing_in_bert = all_unique_pois - bert_pois
    missing_in_gemini = all_unique_pois - gemini_pois

    print(f"GLiN:\t {list(gliner_pois)}")

    print(f"BERT:\t {list(bert_pois)}")

    print(f"Gemi:\t {list(gemini_pois)}")
    print(f"GLiN:\t\t BERT:\t\t Gemi:")

