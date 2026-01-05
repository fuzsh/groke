#!/usr/bin/env python3
"""
Merge corrections from ablation_study/sup/ files into main ablation_study/ files.

This script reads corrected entries from the sup/ directory and replaces
the corresponding entries in the main files based on matching "key" values.
"""

import json
import os
from pathlib import Path
from typing import Dict, List


def load_jsonl(file_path: Path) -> Dict[str, dict]:
    """Load JSONL file and return a dictionary indexed by 'key' field."""
    entries = {}
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist, skipping...")
        return entries
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = str(entry.get('key'))
                if key:
                    entries[key] = entry
                else:
                    print(f"Warning: Entry at line {line_num} in {file_path} has no 'key' field, skipping...")
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse JSON at line {line_num} in {file_path}: {e}")
                continue
    return entries


def load_jsonl_list(file_path: Path) -> List[dict]:
    """Load JSONL file and return a list of all entries."""
    entries = []
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist, returning empty list...")
        return entries
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse JSON at line {line_num} in {file_path}: {e}")
                continue
    return entries


def save_jsonl(file_path: Path, entries: List[dict]) -> None:
    """Save entries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def merge_files(sup_dir: Path, main_dir: Path, filename: str) -> None:
    """Merge a single file from sup/ into main/."""
    sup_file = sup_dir / filename
    main_file = main_dir / filename
    
    if not sup_file.exists():
        print(f"Skipping {filename}: not found in sup/ directory")
        return
    
    if not main_file.exists():
        print(f"Warning: {main_file} does not exist, creating from sup/ file...")
        # If main file doesn't exist, just copy from sup
        sup_entries = load_jsonl_list(sup_file)
        save_jsonl(main_file, sup_entries)
        print(f"Created {main_file} with {len(sup_entries)} entries from sup/")
        return
    
    # Load corrected entries from sup/
    sup_entries_dict = load_jsonl(sup_file)
    print(f"Loaded {len(sup_entries_dict)} corrected entries from {sup_file.name}")
    
    if not sup_entries_dict:
        print(f"No entries found in {sup_file.name}, skipping merge...")
        return
    
    # Load all entries from main file
    main_entries = load_jsonl_list(main_file)
    print(f"Loaded {len(main_entries)} entries from {main_file.name}")
    
    # Create a set of keys to replace
    keys_to_replace = set(sup_entries_dict.keys())
    
    # Replace entries in main file
    replaced_count = 0
    for i, entry in enumerate(main_entries):
        key = str(entry.get('key', ''))
        if key in keys_to_replace:
            main_entries[i] = sup_entries_dict[key]
            replaced_count += 1
            keys_to_replace.remove(key)
    
    # Add any entries from sup/ that weren't in main file
    added_count = 0
    for key in keys_to_replace:
        main_entries.append(sup_entries_dict[key])
        added_count += 1
    
    # Save merged entries back to main file
    save_jsonl(main_file, main_entries)
    
    print(f"Merged {filename}: {replaced_count} replaced, {added_count} added")
    print(f"Total entries in {main_file.name}: {len(main_entries)}")


def main():
    """Main function to merge all files."""
    # Set up paths
    base_dir = Path(__file__).parent
    sup_dir = base_dir / "ablation_study" / "sup"
    main_dir = base_dir / "ablation_study"
    
    # Ensure directories exist
    if not sup_dir.exists():
        print(f"Error: {sup_dir} directory does not exist!")
        return
    
    if not main_dir.exists():
        print(f"Error: {main_dir} directory does not exist!")
        return
    
    # List of files to merge
    files_to_merge = [
        "graph_vis.jsonl",
        "grid.jsonl",
        "json.jsonl",
        "textual.jsonl"
    ]
    
    print(f"Merging corrections from {sup_dir} into {main_dir}")
    print("=" * 60)
    
    for filename in files_to_merge:
        print(f"\nProcessing {filename}...")
        merge_files(sup_dir, main_dir, filename)
    
    print("\n" + "=" * 60)
    print("Merge completed!")


if __name__ == "__main__":
    main()

