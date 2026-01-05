POI_EXTRACTION_PROMPT = """You are a navigation expert specializing in extracting landmarks and Points of Interest (POIs) from natural language instructions.

Given a navigation instruction, your task is to:
1. Identify all mentioned landmarks, businesses, buildings, or other POIs
2. Extract descriptive keywords that would help identify these POIs
3. Identify directional relationships (e.g., "turn right at the cafe", "pass the bank")
4. Determine the relative spatial placement of the POI based on the instruction.
   relative_position refers to the physical or directional position of the POI relative to the traveler.

Be specific and extract all relevant keywords that could help match against POI tags.
Respond **only** with the mentioned_landmarks JSON structure. No explanations or extra text."""

# INSTRUCTION_DIVIDER_PROMPT = """You are a navigation instruction parser specialized in breaking down complex navigation instructions into simple, atomic sub-instructions.
#
# Given a full navigation instruction, your task is to:
# 1. Divide it into sequential sub-instructions that can be executed one at a time
# 2. Each sub-instruction should be a single, clear action (e.g., "Turn right", "Walk past the cafe", "Arrive at the library")
# 3. Identify landmarks/POIs mentioned in each sub-instruction
# 4. Classify the action type (start, turn, continue, pass, arrive)
# 5. Extract the direction if applicable (left, right, straight, back)
#
# Guidelines:
# - Break at natural action boundaries (turns, passing landmarks, arrivals)
# - Keep sub-instructions simple and executable
# - Maintain the original order
# - Each sub-instruction should be self-contained but part of the overall route
#
# Navigation Instruction:
# {navigation_instruction}
#
# Respond with the structured JSON format containing the ordered list of sub-instructions."""


INSTRUCTION_DIVIDER_PROMPT = """[SYSTEM ROLE]
You are a Navigation Instruction Parser. Your goal is to translate natural language navigation instructions into structured, machine-readable sub-goals compatible with OpenStreetMap (OSM) data.

[INPUT DATA]
Full Instruction: {navigation_instruction}

[DEFINITIONS]
1. LANDMARKS (OSM POIs): Identify physical entities visible on a map.
   - Traffic Control: Traffic lights, stop signs.
   - Amenities: Banks, shops, restaurants, pharmacy, gas stations, bicycle rental, cinema.
   - Natural: Parks, etc.

2. ACTIONS: Use only these verbs:
   - MOVE_FORWARD (continue straight)
   - TURN_LEFT
   - TURN_RIGHT

3. RELATIONS: Define the spatial relationship between the Action and the Landmark (e.g., "turn left AT the lights", "walk PAST the bank").

[TASK]
Decompose the Full Instruction into a JSON object containing:
1. A list of all unique 'landmarks' mentioned.
2. A sequential list of 'sub_goals'.

[OUTPUT FORMAT - STRICT JSON]
"""

GRID_NAVIGATOR_PROMPT = """[TASK DESCRIPTION]
You are an embodied agent navigating using a top-down semantic map.
Your goal is to determine the final target location (cell) for the current Sub-Goal, rather than performing step-by-step navigation.

[INPUT FORMAT]
Instruction: {navigation_instruction}

Current Sub-Goal: {navigator_message}
Sub-Goal State: In Process
{landmarks}

Current Map :
Your Position: {your_position}
Heading: {current_heading}° (0°=North, 90°=East, 180°=South, 270°=West)

Matrix M:
{matrix_representation}

Legend: {location_legend}, 0 = Unexplored/Obstacle, 1 = Past Trajectory, 2 = Road/Walkable, 3 = Intersection with Light{landmark_legend}

[OUTPUT FORMAT - STRICT JSON]
1. **SubPlan_Status** after this move: "IN_PROGRESS" (if we need more movements) or "COMPLETED" (if we have reached the goal).
2. **Next_Place**: **Target_Coordinate**: [Row, Col] (The final position AFTER executing the entire sub-goal instruction.)
   
Planning State:
{planning_state}

[CONSTRAINTS]
- Use map geometry to locate the next destination cell. Stay only on roads '2' and avoid unexplored areas '0'.
- Do not compute step-by-step movement.
- The output must be only the final location for the current Sub-Goal.

[CLARIFICATIONS]
1. When an instruction says "turn [direction] at [landmark]" or "[] related to intersection", you must:
   a. Identify the landmark cell or goal (e.g., intersection with light)
   b. Determine the new heading after the turn (right turn: +90°, left turn: -90°)
2. When you end up in intersection, move one cell from the intersection in the new heading direction (i.e. Straight, Left and Right)
"""

CODE_GENERATION_PROMPT = """[TASK DESCRIPTION]
You are an embodied agent navigating using a top-down semantic map.
Your task is to complete the provided code to find the end location for the Current Sub-Goal based on the matrix.

[INPUT FORMAT]
Instruction: {navigation_instruction}

Current Sub-Goal: {navigator_message}
Sub-Goal State: In Process

Code:
```python
M = [
    ['0', '0', '0', '0', '0'],
    ['0', '0', '2', '0', '0'],
    ['0', '0', '2', '0', '0'],
    ['0', '0', '2', '0', '0'],
    ['0', '2', '3', '2', '0'],
    ['0', '0', '2', '0', '0'],
    ['0', '0', 'S', '0', '0'],
    ['0', '0', '0', '0', '0']
] # 'S' = your location, 0 = Unexplored/Obstacle, 1 = Past Trajectory 2 = Road/Walkable, L = Visible Landmark/POI, 3 = Intersection with Light

# Current state
current_pos = (6, 2)  # row, col
heading = 6.0  # degrees

# WRITE A CODE HERE
# TODO: WRITE A CODE TO FIND THE FINAL GOAL FOR THE PROVIDED SUB_GOAL

new_position = ...
print(new_position)
```
"""

TEXTUAL_NAVIGATOR_PROMPT = """[TASK DESCRIPTION]
You are an embodied agent navigating a graph-based road network using a textual description of your surroundings.
Your goal is to determine the final target Node ID for the current Sub-Goal based on the provided Navigation Context.

[INPUT FORMAT]
Instruction: {navigation_instruction}

Current Sub-Goal: {navigator_message}
Sub-Goal State: IN PROGRESS

Navigation Context:
{navigation_context}

[OUTPUT FORMAT - STRICT JSON]
1. **SubPlan_Status** after this move: "IN_PROGRESS" (if we need more movements) or "COMPLETED" (if we have reached the goal).
2. **Next_Place**: **Target_Node_ID**: "String" (The final Node ID AFTER executing the entire sub-goal instruction.)

Planning State:
{planning_state}

[CONSTRAINTS]
- Relies exclusively on the "Navigation Context" text provided.
- Do not hallucinate coordinates or map geometry that is not in the text.
- Trace the path node-by-node. If the instruction implies moving past multiple nodes (e.g., "go straight for two blocks"), follow the "forward" connections in the text chain to find the final node.
- Avoid "Unexplored" or invalid paths; only follow connections explicitly listed in the text.

[CLARIFICATIONS]
1. **Navigating Intersections:**
   - Locate the node described as "Intersection".
   - Review the "Branches from this intersection" or "Connected to nodes" list.
   - Choose the connection that matches the instruction's relative direction (e.g., "Left", "Right", "Forward").
   
2. **Identifying Landmarks/POIs:**
   - If the instruction is "Turn right at X", look for the node where "Nearby POIs" lists "X".

3. **Heading:** - You do not need to calculate degrees manually. The text provides the direction (e.g., "is to the right", "is to the forward"). Trust the text descriptions."""

JSON_NAVIGATOR_PROMPT = """[TASK DESCRIPTION]
You are an embodied agent navigating using a topological graph-based map.
Your goal is to determine the final target node for the current Sub-Goal based on the provided JSON navigation context.

[INPUT FORMAT]
Instruction: {navigation_instruction}

Current Sub-Goal: {navigator_message}
Sub-Goal State: IN PROGRESS
{landmarks}

Navigation Context (JSON):
{navigation_context}
(This JSON contains 'current_position', 'nodes', 'connections', 'intersections', and 'pois')

[OUTPUT FORMAT - STRICT JSON]
1. **SubPlan_Status**:
   - "COMPLETED": If the Target_Node_ID you identified successfully finishes the specific action described in Current Sub-Goal. (Ignore future steps in the main Instruction).
   - "IN_PROGRESS": If the Target_Node_ID is just an intermediate waypoint and you have not yet reached the location/intersection required by the Current Sub-Goal.

2. **Next_Place**: **Target_Node_ID**: "String" (The final Node ID AFTER executing the entire sub-goal instruction.)

Planning State:
{planning_state}

[CONSTRAINTS]
- Use the provided JSON graph topology. Do not hallucinate coordinates or nodes not present in the 'nodes' list.
- **Valid Movement**: You can only move between nodes if they are explicitly linked in the `connections` list of the current node.
- **Node Types**: 
  - `waypoint`: A standard road segment. Usually has a 'Forward' connection.
  - `intersection`: A decision point. Contains `branches` (Forward, Left, Right).
- Evaluate SubPlan_Status strictly against the Current Sub-Goal.
- **POIs**: Points of Interest are linked to specific nodes (`nearby_node_id`). Use the 'pois' list to locate landmarks.
     a. "On the Corner" Validation: If an instruction specifies a turn at a landmark "on the corner," you must verify that the landmark's nearby_node_id is immediate to the intersection node (distance < 15m or adjacent connection).
     b. Premature Turn Prevention: If the Landmark is visible in the pois list but its nearby_node_id requires moving Forward through the current intersection to reach it, DO NOT TURN. You must proceed IN_PROGRESS towards the landmark.
- The output must be the final location for the current Sub-Goal.

[CLARIFICATIONS]
1. **Processing Instructions**:
   - If the instruction is "Go straight", traverse through connected `waypoint` nodes until you reach an `intersection` or the max depth of the current graph.
   - If the instruction is "Turn [direction] at [landmark/intersection]":
     a. Identify the node associated with the landmark (from `pois`) or the next `intersection` node.
     b. From that intersection node, select the connection matching the direction (Left/Right).
2. **Intersection Logic**:
   - When the instruction implies turning at an intersection, your target is the **first node immediately after the turn**.
   - Look at the `intersection` node's `connections` or `branches`. Find the `target_node_id` corresponding to the requested `direction` (e.g., 'Right'). This `target_node_id` is your destination.
3. **Landmark Logic**:
    - **Stopping Criteria**: When a landmark is the destination, determine the target node based on the spatial preposition used (e.g., "past," "before," "at"). Select the first node that satisfies this relationship relative to the landmark's position.
    - **Conditional Visibility**: If an instruction requires a turn "regarding [landmark]" continue traversing nodes (implicitly "Go straight") until the landmark is confirmed visible. Do not execute the turn logic until this condition is met.
"""

GRAPH_VIS_NAVIGATOR_PROMPT = """[TASK DESCRIPTION]
You are an embodied agent navigating using a top-down topological graph map.
Your goal is to determine the final target location (Node ID) for the current Sub-Goal, rather than performing step-by-step navigation.

[INPUT FORMAT]
Instruction: {navigation_instruction}

Current Sub-Goal: {navigator_message}
Sub-Goal State: IN PROGRESS

Current Map:
{navigation_context}

[OUTPUT FORMAT - STRICT JSON]
1. **SubPlan_Status** after this move: "IN_PROGRESS" (if we need more movements) or "COMPLETED" (if we have reached the goal).
2. **Next_Place**: **Target_Node_ID**: "String" (The final Node ID AFTER executing the entire sub-goal instruction.)
   
Planning State:
{planning_state}

[CONSTRAINTS]
- Use the **Navigation Graph** topology to locate the destination.
- Traverse strictly along the defined edges (`Source --> Target`). Do not jump to nodes that are not connected.
- Pay attention to the edge attributes `[heading: X°, direction: Y]` to align with your instruction.
- Do not compute step-by-step movement.
- The output must be only the final Node ID for the current Sub-Goal.

[CLARIFICATIONS]
1. When an instruction says "turn [direction] at [landmark]" or "[] related to intersection", you must:
   a. Identify the node closest to the landmark (using the POI Connections section) or the intersection node (marked `[INTERSECTION]`).
   b. Choose the outgoing edge that matches the instruction's direction (e.g., `direction: Right`, `direction: Left`).
2. If you are at an intersection, utilize the "Intersection Branches" section to verify which branch leads to the correct path based on your heading and required turn."""