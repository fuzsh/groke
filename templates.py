"""
Templates for generating batch requests and prompts.

This module provides template functions for creating navigation agent requests
with various representation formats.
"""

import json
from typing import Dict, List, Any, Optional
def navigation_divider_batch(key: str, navigation_instruction: str):
    return {
        "key": key,
        "request": {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"Navigation Instruction: {navigation_instruction}"
                        }
                    ],
                    "role": "user"
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "description": "Final structured response for the navigation instruction.",
                    "properties": {
                        "landmarks": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "description": "Represents a physical map entity (POI) found in the navigation instruction.",
                                "properties": {
                                    "relative_position": {
                                        "type": "STRING",
                                        "enum": [
                                            "left",
                                            "right",
                                            "ahead",
                                            "on the corner",
                                            "across",
                                            "next to",
                                            "unknown"
                                        ]
                                    },
                                    "name": {
                                        "type": "STRING"
                                    },
                                    "category": {
                                        "type": "STRING",
                                        "nullable": True
                                    }
                                },
                                "required": [
                                    "name",
                                    "relative_position"
                                ],
                                "propertyOrdering": [
                                    "name",
                                    "category",
                                    "relative_position"
                                ]
                            }
                        },
                        "sub_goals": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "description": "Represents a single actionable navigation step.",
                                "properties": {
                                    "description": {
                                        "type": "STRING"
                                    },
                                    "status": {
                                        "type": "STRING",
                                        "enum": [
                                            "TODO",
                                            "IN_PROGRESS",
                                            "COMPLETED"
                                        ]
                                    }
                                },
                                "required": [
                                    "description"
                                ],
                                "propertyOrdering": [
                                    "description",
                                    "status"
                                ]
                            }
                        }
                    },
                    "required": [
                        "landmarks",
                        "sub_goals"
                    ],
                    "propertyOrdering": [
                        "landmarks",
                        "sub_goals"
                    ]
                },
                "thinking_config": {"include_thoughts": True}
            },
            "systemInstruction": {
                "parts": [
                    {
                        "text": "[SYSTEM ROLE]\nYou are a Navigation Instruction Parser. Your goal is to translate natural language navigation instructions into structured, machine-readable sub-goals compatible with OpenStreetMap (OSM) data.\n\n[DEFINITIONS]\n1. LANDMARKS (OSM POIs): Identify physical entities visible on a map.\n   - Traffic Control: Traffic lights, stop signs.\n   - Amenities: Banks, shops, restaurants, pharmacy, gas stations, bicycle rental, cinema.\n   - Natural: Parks, etc.\n\n2. ACTIONS: Use only these verbs:\n   - MOVE_FORWARD (continue straight)\n   - TURN_LEFT\n   - TURN_RIGHT\n\n3. RELATIONS: Define the spatial relationship between the Action and the Landmark (e.g., \"turn left AT the lights\", \"walk PAST the bank\").\n\n[TASK]\nDecompose the Full Instruction into a JSON object containing:\n1. A list of all unique 'landmarks' mentioned.\n2. A sequential list of 'sub_goals'.\n3. For each sub-goal, assign a 'status' (TODO, IN_PROGRESS, COMPLETED). *Note: Unless live telemetry is provided, default all future steps to TODO.*\n\n[OUTPUT FORMAT - STRICT JSON].\n\n"
                    }
                ],
                "role": "user"
            }
        }
    }

def navigator_batch(
        key:str,
        prompt:str,
        next_node_type
):
    return {
        "key": key,
        "request": {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ],
                    "role": "user"
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "description": "Final structured reasoning output for navigation.",
                    "properties": {
                        "subplan_status": {
                            "type": "STRING",
                            "enum": [
                                "IN_PROGRESS",
                                "COMPLETED"
                            ]
                        },
                        "next_place": next_node_type
                    },
                    "required": [
                        "subplan_status",
                        "next_place"
                    ],
                    "propertyOrdering": [
                        "subplan_status",
                        "next_place"
                    ]
                },
                "thinking_config": {"include_thoughts": True}
            },
            "systemInstruction": {
                "parts": [
                    {
                        "text": """[TASK DESCRIPTION]
You are an embodied agent navigating using a top-down semantic map.
Your goal is to determine the final target location (cell) for the current Sub-Goal, rather than performing step-by-step navigation.
"""}
                ],
                "role": "user"
            }
        }
    }


def navigator_batch_multi_format(
    key: str,
    navigation_instruction: str,
    navigator_message: str,
    current_heading: str,
    location_legend: str,
    landmarks: str,
    landmark_legend: str,
    planning_state: str,
    your_position: str,
    representations: Dict[str, str],
    primary_format: str = "matrix"
) -> Dict[str, Any]:
    """
    Generate a batch request with multiple representation formats.

    Args:
        key: Unique identifier for this request
        navigation_instruction: Full navigation instruction text
        navigator_message: Current sub-instruction/message for navigator
        current_heading: Current heading as string
        location_legend: Legend for position markers
        landmarks: Landmarks mentioned in instruction
        landmark_legend: Legend mapping letters to landmarks
        planning_state: Current state of planning
        your_position: Description of current position
        representations: Dict of format_name -> representation_string
        primary_format: Which format to emphasize as primary

    Returns:
        Dict formatted as a batch request
    """
    return navigator_batch(
        key=key,
        navigation_instruction=navigation_instruction,
        navigator_message=navigator_message,
        current_heading=current_heading,
        location_legend=location_legend,
        landmarks=landmarks,
        landmark_legend=landmark_legend,
        planning_state=planning_state,
        your_position=your_position,
        matrix_representation=representations.get("matrix", ""),
        textual_representation=representations.get("textual"),
        json_representation=representations.get("json"),
        graphviz_representation=representations.get("graphviz"),
        include_formats=list(representations.keys()) if representations else [primary_format]
    )


def step_prompt_template(
    step_number: int,
    sub_instruction: str,
    heading: float,
    representation: str,
    format_type: str = "matrix"
) -> str:
    """
    Generate a prompt template for a single navigation step.

    Args:
        step_number: Current step number in the sequence
        sub_instruction: Current sub-instruction
        heading: Current heading in degrees
        representation: The representation string (any format)
        format_type: Type of representation provided

    Returns:
        Formatted prompt string
    """
    heading_compass = _heading_to_compass(heading)

    prompt = f"""Step {step_number}:
Sub-Instruction: {sub_instruction}
Current Heading: {heading:.1f}° ({heading_compass})

"""

    if format_type == "textual":
        prompt += f"Environment Description:\n{representation}\n"
    elif format_type == "json":
        prompt += f"Structured Context:\n{representation}\n"
    elif format_type == "graphviz":
        prompt += f"Graph Structure:\n{representation}\n"
    else:  # matrix
        prompt += f"Map Matrix:\n{representation}\n"

    prompt += """
Based on this context, determine:
1. The target location for this sub-instruction
2. Whether the sub-goal is COMPLETED or IN_PROGRESS
3. The next heading direction

Respond with JSON: {"next_position": [row, col], "status": "...", "heading": ...}
"""

    return prompt


def multi_step_batch(
    key: str,
    steps: List[Dict[str, Any]],
    navigation_instruction: str
) -> List[Dict[str, Any]]:
    """
    Generate batch requests for a multi-step navigation sequence.

    Args:
        key: Base identifier for requests
        steps: List of step dicts with sub_instruction, heading, representations
        navigation_instruction: Full navigation instruction

    Returns:
        List of batch request dicts
    """
    requests = []

    for i, step in enumerate(steps):
        sub_instruction = step.get("sub_instruction", "")
        heading = step.get("heading", 0)
        representations = step.get("representations", {})

        # Determine primary format
        primary_format = step.get("primary_format", "matrix")

        request = {
            "custom_id": f"request-{key}-step-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are navigating step {i + 1} of a multi-step journey. "
                                   f"Full instruction: {navigation_instruction}"
                    },
                    {
                        "role": "user",
                        "content": step_prompt_template(
                            step_number=i + 1,
                            sub_instruction=sub_instruction,
                            heading=heading,
                            representation=representations.get(primary_format, ""),
                            format_type=primary_format
                        )
                    }
                ],
                "max_tokens": 500
            },
            "metadata": {
                "step": i,
                "key": key,
                "format": primary_format
            }
        }

        requests.append(request)

    return requests


def _heading_to_compass(heading: float) -> str:
    """Convert heading degrees to compass direction."""
    heading = heading % 360
    if 337.5 <= heading or heading < 22.5:
        return "North"
    elif 22.5 <= heading < 67.5:
        return "Northeast"
    elif 67.5 <= heading < 112.5:
        return "East"
    elif 112.5 <= heading < 157.5:
        return "Southeast"
    elif 157.5 <= heading < 202.5:
        return "South"
    elif 202.5 <= heading < 247.5:
        return "Southwest"
    elif 247.5 <= heading < 292.5:
        return "West"
    else:
        return "Northwest"
