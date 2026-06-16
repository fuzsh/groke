import json
from collections import Counter



with open('ablation_study/test_seen_100.json', 'r')as f:
    navigation_instructions = json.load(f)

all_instructions_word_list = []
for navigation_instruction in navigation_instructions:
    instruction = navigation_instruction['instructions']
    word_list = instruction.split()
    all_instructions_word_list.extend(word_list)

word_count = len(all_instructions_word_list)
word_frequencies = Counter(all_instructions_word_list)
print(word_count)
print(word_frequencies)