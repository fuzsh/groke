#!/usr/bin/env python3
"""
Usage Metadata Analysis Script

Analyzes usageMetadata statistics from JSONL files containing instruction steps.
Each instruction can have 1-10 steps, and this script computes per-instruction
aggregates and overall statistics.

Usage:
    python usage_metadata_anal.py <folder_path>

Example:
    python usage_metadata_anal.py ablation_study_json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics


def load_jsonl_files(folder_path: str) -> Dict[str, List[Dict]]:
    """
    Load all JSONL files from the folder and group records by instruction key.

    Returns:
        Dictionary mapping instruction key to list of records (one per step)
    """
    instructions = defaultdict(list)
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    jsonl_files = sorted(folder.glob("*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {folder_path}")

    for jsonl_file in jsonl_files:
        step_name = jsonl_file.stem  # e.g., "step1_json"
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record.get("key", f"unknown_{line_num}")
                    record["_step_file"] = step_name
                    instructions[key].append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {jsonl_file}: {e}")

    return dict(instructions)


def extract_usage_metadata(record: Dict) -> Optional[Dict]:
    """Extract usageMetadata from a record."""
    try:
        return record.get("response", {}).get("usageMetadata", None)
    except (KeyError, TypeError):
        return None


def aggregate_instruction_usage(records: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate usageMetadata across all steps for a single instruction.

    Returns a dictionary with:
        - Total counts for each metric
        - Number of steps
        - List of per-step values
    """
    metrics = {
        "promptTokenCount": [],
        "candidatesTokenCount": [],
        "thoughtsTokenCount": [],
        "totalTokenCount": [],
        "billablePromptTextCount": [],
    }

    for record in records:
        usage = extract_usage_metadata(record)
        if usage is None:
            continue

        if "promptTokenCount" in usage:
            metrics["promptTokenCount"].append(usage["promptTokenCount"])
        if "candidatesTokenCount" in usage:
            metrics["candidatesTokenCount"].append(usage["candidatesTokenCount"])
        if "thoughtsTokenCount" in usage:
            metrics["thoughtsTokenCount"].append(usage["thoughtsTokenCount"])
        if "totalTokenCount" in usage:
            metrics["totalTokenCount"].append(usage["totalTokenCount"])

        # billablePromptUsage.textCount
        billable = usage.get("billablePromptUsage", {})
        if billable and "textCount" in billable:
            metrics["billablePromptTextCount"].append(billable["textCount"])

    result = {
        "num_steps": len(records),
        "num_valid_steps": len(metrics["totalTokenCount"]),
    }

    for metric_name, values in metrics.items():
        if values:
            result[f"{metric_name}_total"] = sum(values)
            result[f"{metric_name}_avg_per_step"] = statistics.mean(values)
            result[f"{metric_name}_values"] = values
        else:
            result[f"{metric_name}_total"] = 0
            result[f"{metric_name}_avg_per_step"] = 0
            result[f"{metric_name}_values"] = []

    return result


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistical measures for a list of values."""
    if not values:
        return {
            "count": 0,
            "sum": 0,
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "stdev": 0,
            "variance": 0,
        }

    stats = {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }

    if len(values) > 1:
        stats["stdev"] = statistics.stdev(values)
        stats["variance"] = statistics.variance(values)
    else:
        stats["stdev"] = 0
        stats["variance"] = 0

    # Add percentiles
    sorted_values = sorted(values)
    n = len(sorted_values)
    stats["p25"] = sorted_values[int(n * 0.25)] if n > 0 else 0
    stats["p75"] = sorted_values[int(n * 0.75)] if n > 0 else 0
    stats["p90"] = sorted_values[int(n * 0.90)] if n > 0 else 0
    stats["p95"] = sorted_values[int(n * 0.95)] if n > 0 else 0

    return stats


def analyze_folder(folder_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Main analysis function.

    Returns a comprehensive analysis dictionary with:
        - Per-instruction aggregates
        - Overall statistics across all instructions
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {folder_path}")
    print(f"{'='*60}\n")

    # Load data
    instructions = load_jsonl_files(folder_path)
    print(f"Loaded {len(instructions)} unique instructions")

    # Aggregate per instruction
    instruction_aggregates = {}
    for key, records in instructions.items():
        instruction_aggregates[key] = aggregate_instruction_usage(records)

    # Compute overall statistics
    metrics_to_analyze = [
        "promptTokenCount_total",
        "candidatesTokenCount_total",
        "thoughtsTokenCount_total",
        "totalTokenCount_total",
        "billablePromptTextCount_total",
        "num_steps",
    ]

    overall_stats = {}
    for metric in metrics_to_analyze:
        values = [agg[metric] for agg in instruction_aggregates.values()]
        overall_stats[metric] = compute_statistics(values)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS (per instruction totals)")
    print(f"{'='*60}\n")

    print(f"Number of instructions: {len(instructions)}")
    print()

    # Steps per instruction
    steps_stats = overall_stats["num_steps"]
    print("Steps per instruction:")
    print(f"  Mean:   {steps_stats['mean']:.2f}")
    print(f"  Median: {steps_stats['median']:.0f}")
    print(f"  Min:    {steps_stats['min']:.0f}")
    print(f"  Max:    {steps_stats['max']:.0f}")
    print(f"  Stdev:  {steps_stats['stdev']:.2f}")
    print()

    # Token statistics
    token_metrics = [
        ("Prompt Tokens", "promptTokenCount_total"),
        ("Candidates Tokens", "candidatesTokenCount_total"),
        ("Thoughts Tokens", "thoughtsTokenCount_total"),
        ("Total Tokens", "totalTokenCount_total"),
        ("Billable Prompt Text", "billablePromptTextCount_total"),
    ]

    for label, metric in token_metrics:
        stats = overall_stats[metric]
        print(f"{label} (per instruction):")
        print(f"  Sum:    {stats['sum']:,.0f}")
        print(f"  Mean:   {stats['mean']:,.2f}")
        print(f"  Median: {stats['median']:,.0f}")
        print(f"  Min:    {stats['min']:,.0f}")
        print(f"  Max:    {stats['max']:,.0f}")
        print(f"  Stdev:  {stats['stdev']:,.2f}")
        print(f"  P25:    {stats['p25']:,.0f}")
        print(f"  P75:    {stats['p75']:,.0f}")
        print(f"  P90:    {stats['p90']:,.0f}")
        print(f"  P95:    {stats['p95']:,.0f}")
        print()

    # Per-step analysis
    print(f"\n{'='*60}")
    print("PER-STEP BREAKDOWN")
    print(f"{'='*60}\n")

    # Collect per-step data
    step_data = defaultdict(lambda: defaultdict(list))
    for key, records in instructions.items():
        for i, record in enumerate(records, 1):
            usage = extract_usage_metadata(record)
            if usage:
                step_data[i]["promptTokenCount"].append(usage.get("promptTokenCount", 0))
                step_data[i]["candidatesTokenCount"].append(usage.get("candidatesTokenCount", 0))
                step_data[i]["thoughtsTokenCount"].append(usage.get("thoughtsTokenCount", 0))
                step_data[i]["totalTokenCount"].append(usage.get("totalTokenCount", 0))

    for step_num in sorted(step_data.keys()):
        data = step_data[step_num]
        print(f"Step {step_num} (n={len(data['totalTokenCount'])} instructions):")
        for metric in ["promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount"]:
            values = data[metric]
            if values:
                print(f"  {metric}: mean={statistics.mean(values):,.0f}, median={statistics.median(values):,.0f}, sum={sum(values):,}")
        print()

    # Verbose: per-instruction details
    if verbose:
        print(f"\n{'='*60}")
        print("PER-INSTRUCTION DETAILS")
        print(f"{'='*60}\n")

        for key in sorted(instruction_aggregates.keys()):
            agg = instruction_aggregates[key]
            print(f"Instruction {key}:")
            print(f"  Steps: {agg['num_steps']}")
            print(f"  Total Tokens: {agg['totalTokenCount_total']:,}")
            print(f"  Prompt Tokens: {agg['promptTokenCount_total']:,}")
            print(f"  Candidates Tokens: {agg['candidatesTokenCount_total']:,}")
            print(f"  Thoughts Tokens: {agg['thoughtsTokenCount_total']:,}")
            print()

    return {
        "folder": folder_path,
        "num_instructions": len(instructions),
        "instruction_aggregates": instruction_aggregates,
        "overall_stats": overall_stats,
        "step_data": dict(step_data),
    }


def compare_folders(folder_paths: List[str]) -> None:
    """Compare statistics across multiple folders."""
    results = []
    for folder in folder_paths:
        try:
            result = analyze_folder(folder, verbose=False)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {folder}: {e}")

    if len(results) < 2:
        return

    print(f"\n{'='*60}")
    print("COMPARISON ACROSS FOLDERS")
    print(f"{'='*60}\n")

    # Create comparison table
    headers = ["Folder", "Instructions", "Avg Steps", "Avg Total Tokens", "Avg Thoughts", "Avg Prompt"]

    print(f"{'Folder':<30} {'Instr':>8} {'AvgSteps':>10} {'AvgTotalTok':>14} {'AvgThoughts':>14} {'AvgPrompt':>14}")
    print("-" * 100)

    for result in results:
        folder_name = os.path.basename(result["folder"])
        n_instr = result["num_instructions"]
        avg_steps = result["overall_stats"]["num_steps"]["mean"]
        avg_total = result["overall_stats"]["totalTokenCount_total"]["mean"]
        avg_thoughts = result["overall_stats"]["thoughtsTokenCount_total"]["mean"]
        avg_prompt = result["overall_stats"]["promptTokenCount_total"]["mean"]

        print(f"{folder_name:<30} {n_instr:>8} {avg_steps:>10.2f} {avg_total:>14,.0f} {avg_thoughts:>14,.0f} {avg_prompt:>14,.0f}")


def export_to_json(result: Dict, output_path: str) -> None:
    """Export analysis results to JSON file."""
    # Convert defaultdicts and make JSON serializable
    export_data = {
        "folder": result["folder"],
        "num_instructions": result["num_instructions"],
        "overall_stats": result["overall_stats"],
        "instruction_aggregates": {
            k: {key: val for key, val in v.items() if not key.endswith("_values")}
            for k, v in result["instruction_aggregates"].items()
        },
    }

    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"\nExported results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze usageMetadata statistics from JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "folder",
        nargs="+",
        help="Path(s) to folder(s) containing JSONL files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-instruction details"
    )
    parser.add_argument(
        "-o", "--output",
        help="Export results to JSON file"
    )
    parser.add_argument(
        "-c", "--compare",
        action="store_true",
        help="Compare statistics across multiple folders"
    )

    args = parser.parse_args()

    if len(args.folder) == 1:
        result = analyze_folder(args.folder[0], verbose=args.verbose)
        if args.output:
            export_to_json(result, args.output)
    else:
        if args.compare:
            compare_folders(args.folder)
        else:
            for folder in args.folder:
                result = analyze_folder(folder, verbose=args.verbose)
                if args.output:
                    base_name = os.path.basename(folder)
                    output_path = args.output.replace(".json", f"_{base_name}.json")
                    export_to_json(result, output_path)


if __name__ == "__main__":
    main()