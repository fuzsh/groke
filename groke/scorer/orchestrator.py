import json
import math
from collections import deque

from google.adk.agents import BaseAgent, LlmAgent, InvocationContext
from typing import AsyncGenerator, Dict, override, Optional, Tuple
from google.adk.events import Event
from rapidfuzz import fuzz

from .graph_context import navigate
from .grid_representation import convert2grid, get_node_id_from_position


class NavigationOrchestrator(BaseAgent):
    instruction_divider: LlmAgent
    node_navigator: LlmAgent

    # Allow arbitrary types for Pydantic
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
            self,
            name: str,
            instruction_divider: LlmAgent,
            node_navigator: LlmAgent
    ):
        # Create sub-agents list
        sub_agents_list = [
            instruction_divider,
            node_navigator,
        ]

        # Register with parent BaseAgent
        super().__init__(
            name=name,
            description="Navigate on the city graph using natural language instructions with multi-agent coordination.",
            instruction_divider=instruction_divider,
            node_navigator=node_navigator,
            sub_agents=sub_agents_list
        )

    def _get_heading_to_node(self, area_links, area_nodes, start_node_id, end_node_id):
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
            return "Error: No path found connecting these nodes."

        # 3. Get the Immediate Previous Node
        # The node just before the destination is at index -2
        if len(found_path) < 2:
            return "Error: Start and End nodes are the same."

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

        print(found_path)

        return found_path, compass_bearing

    def _build_navigator_message(
            self,
            area_data: Dict,
            current_node_id: str,
            current_heading: float,
            pois: dict = None,
            poi_mapping: dict = None,
            target_poi: str = None,
            previous_visited=None
    ):
        """
        Build the navigation context message for the node navigator using DSL format.
        Uses navigate() to get the full path ahead and encodes it with route DSL format.
        """
        # Use navigate() to get the list of nodes in front
        path_data = navigate(
            map_json=area_data,
            starting_point=current_node_id,
            heading=int(current_heading),
            pois=pois,
            poi_mapping=poi_mapping,
            units=1,
            last_instruction=False
        )

        if previous_visited is None:
            previous_visited = []

        return path_data, convert2grid(
            path_data,
            previous_visited,
            area_nodes=area_data['area_nodes'],  # REQUIRED for positioning
            area_pois=area_data['area_pois'],  # REQUIRED for positioning
        ).tolist()

    def _get_previous_nodes(self, path_data, previous_visited):
        new_path_data = []

        for pd in path_data:
            if pd['node_id'] not in previous_visited:
                continue

            new_connectivity = []
            connectivity = pd.get('connectivity', {})
            if connectivity:
                for con in connectivity:
                    if con['node_id'] in previous_visited:
                        new_connectivity.append(con)
            pd['connectivity'] = new_connectivity
            new_path_data.append(pd)

        return new_path_data

    def _find_available_pois(self, area_pois, landmarks):
        available_pois = {}

        for node_id, info in area_pois.items():
            tags = json.loads(info["tags"])

            name = (
                f"{tags.get('name', '')} "
                f"{tags.get('amenity', '').replace('_', ' ')} "
                f"{tags.get('cuisine', '').replace('_', ' ')} "
                f"{tags.get('leisure', '').replace('_', ' ')} "
                f"{tags.get('tourism', '').replace('_', ' ')} "
                f"{tags.get('shop', '').replace('_', ' ')}"
            ).strip()

            for landmark in landmarks:
                score = fuzz.partial_ratio(landmark['name'], name)
                if score > 70:
                    available_pois.setdefault(landmark['name'], []).append(node_id)

        return available_pois

    def _get_unique_identifiers(self, landmark_in_area):
        used = set()
        mapping = {}

        # Characters that are disallowed
        forbidden = {"S", "P"}

        # Fallback alphabet
        alphabet = [chr(i) for i in range(ord("A"), ord("Z") + 1) if chr(i) not in forbidden]

        for name in landmark_in_area:
            assigned = None

            # Try characters from the name first
            for ch in name:
                upper = ch.upper()
                if ch.isalpha() and upper not in forbidden and upper not in used:
                    assigned = upper
                    used.add(upper)
                    break

            # Fallback if needed
            if assigned is None:
                for c in alphabet:
                    if c not in used:
                        assigned = c
                        used.add(c)
                        break

            mapping[name] = assigned

        return mapping

    def _get_current_instruction_poi(self, sub_instruction_text, landmarks_in_area):
        current_sub_instructions_landmarks = {}

        for lia_name, lia_node_id in landmarks_in_area.items():
            if lia_name in sub_instruction_text:
                current_sub_instructions_landmarks[lia_name] = lia_node_id

        return current_sub_instructions_landmarks

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Main orchestration logic for talk-to-nav navigation.

        Process:
        1. Extract POIs from instruction
        2. Divide instruction into sub-instructions
        3. For each sub-instruction:
           a. Get current position and neighbors
           b. Predict next node using LLM agent
           c. Check if sub-instruction is satisfied
           d. Move to next node or next sub-instruction
        4. Stop when all sub-instructions are completed or stuck
        """

        print("Stage 1: Instruction Division")

        # ==== Stage 1: Instruction Division ====
        async for event in self.instruction_divider.run_async(ctx):
            yield event

        print("Stage 2: Instruction Division")

        instruction_division = ctx.session.state.get("instruction_division", {})
        sub_instructions = instruction_division.get("sub_goals", [])
        landmarks = instruction_division.get("landmarks", [])

        if not sub_instructions:
            ctx.session.state["error"] = "Failed to divide instruction into sub-instructions"
            return

        # Get area_data from session state (must be set by caller)
        area_data = ctx.session.state.get("area_data")
        if not area_data:
            ctx.session.state["error"] = "area_data not found in session state"
            return

        instruction_data = area_data.get("instruction_data", {})
        navigation_instruction = instruction_data.get("instructions", "")

        if not navigation_instruction:
            ctx.session.state["error"] = "No navigation instruction found"
            return

        # Extract graph data for calculate headings in the future
        area_nodes = area_data.get("area_nodes", {})
        area_links = area_data.get("area_links", [])
        area_pois = area_data.get("area_pois", [])

        landmarks_in_area = self._find_available_pois(area_pois, landmarks)

        # Get route information for initialization
        route = instruction_data.get("route", {})
        osm_path = route.get("osm_path", [])
        if not osm_path:
            ctx.session.state["error"] = "No route path found"
            return

        # Initialize at start node
        current_node_id = osm_path[0]
        current_heading = float(route.get("initial_heading", 0))
        previous_visited_path = []

        # ==== Stage 3: Navigation Loop ====
        navigation_trajectory = []
        predicted_path = [current_node_id]  # Track the predicted path
        current_sub_idx = 0
        max_steps_per_sub = 15  # Prevent infinite loops
        steps_in_current_sub = 0
        total_steps = 0
        max_total_steps = 100  # Global safety limit

        while current_sub_idx < len(sub_instructions) and total_steps < max_total_steps:
            current_sub = sub_instructions[current_sub_idx]
            sub_instruction_text = current_sub.get("description", "")

            ctx.session.state["navigator_message"] = sub_instruction_text
            ctx.session.state["current_heading"] = f"{current_heading:.2f}"

            ctx.session.state[
                "location_legend"
            ] = "S = Current Position" if current_sub_idx == 0 else "S = Start Position, P = Current Position"

            # Find the landmark related to instruction and add it to context + add it as legend ...
            unique_identifiers_mapping = None
            sub_instruction_landmarks = self._get_current_instruction_poi(sub_instruction_text, landmarks_in_area)
            if sub_instruction_landmarks:
                unique_identifiers_mapping = self._get_unique_identifiers(sub_instruction_landmarks)
                ctx.session.state["landmarks"] = "Landmark: " + ", ".join(unique_identifiers_mapping.keys())
                ctx.session.state["landmark_legend"] = ", " + ", ".join(
                    f"{char} = {name} (landmark)" for name, char in unique_identifiers_mapping.items()
                )
            else:
                ctx.session.state["landmarks"] = ""
                ctx.session.state["landmark_legend"] = ""

            print(landmarks, unique_identifiers_mapping, sub_instruction_text)

            # Put all the sub-instructions as the planning_state
            ctx.session.state["planning_state"] = "\n".join(
                f"{i + 1}. {sub.get('description', '')} "
                f"({'IN_PROGRESS' if i == current_sub_idx else 'COMPLETED' if i < current_sub_idx else 'TODO'})"
                for i, sub in enumerate(sub_instructions)
            )

            # Store the navigation context message in session state for the navigator agent
            # TODO: Put the POI in the correct place
            path_data, matrix_representation = self._build_navigator_message(
                area_data=area_data,
                current_node_id=current_node_id,
                current_heading=current_heading,
                previous_visited=previous_visited_path,
                pois=sub_instruction_landmarks,
                poi_mapping=unique_identifiers_mapping
            )

            ctx.session.state["matrix_representation"] = "\n".join(map(str, matrix_representation))

            locations: Dict[str, Optional[Tuple[str, str]]] = {"S": None, "P": None}

            for r, row in enumerate(matrix_representation):
                for c, val in enumerate(row):
                    if val in locations:
                        locations[val] = (str(r), str(c))

            ctx.session.state[
                "your_position"
            ] = f"[{locations['S'][0]}, {locations['S'][1]}] (marked as S)" if current_sub_idx == 0 \
                else f"[{locations['P'][0]}, {locations['P'][1]}] (marked as P) and you start position [{locations['S'][0]}, {locations['S'][1]}] (marked as S)"

            # ==== Call Node Navigator Agent ====
            async for event in self.node_navigator.run_async(ctx):
                yield event

            next_node_prediction = ctx.session.state.get("next_node_prediction", {})

            if not next_node_prediction:
                ctx.session.state["navigation_status"] = "failed"
                ctx.session.state["failure_reason"] = "Node navigator failed to predict next node"
                break

            sub_instruction_status = next_node_prediction.get("subplan_status")

            next_place = next_node_prediction.get("next_place")
            next_node_id = get_node_id_from_position(path_data, previous_visited_path, next_place)

            # TODO: Check if the next node is not in the route --> if not match return the path and stop !!!!

            # Get new heading for the move
            visited_route, new_heading = self._get_heading_to_node(
                area_links, area_nodes, current_node_id, next_node_id
            )

            if visited_route:
                # TODO: Sum the previous nodes with it's previous nodes
                previous_visited_path = self._get_previous_nodes(path_data, visited_route)

            if new_heading is None:
                new_heading = current_heading

            # Record step
            navigation_trajectory.append({
                "step": len(navigation_trajectory),
                "from_node": current_node_id,
                "to_node": next_node_id,
                "from_heading": current_heading,
                "to_heading": new_heading,
                "sub_instruction_index": current_sub_idx,
                "sub_instruction": sub_instruction_text,
            })

            # Move to next node
            previous_node_id = current_node_id
            current_node_id = next_node_id
            current_heading = new_heading
            predicted_path.append(current_node_id)
            steps_in_current_sub += 1
            total_steps += 1

            # Check if sub-instruction is satisfied
            if sub_instruction_status == "COMPLETED":
                # Move to next sub-instruction
                current_sub_idx += 1
                steps_in_current_sub = 0

                # Check if this was the last sub-instruction
                if current_sub_idx >= len(sub_instructions):
                    ctx.session.state["navigation_status"] = "completed"
                    break
            else:
                # Check if we've taken too many steps on this sub-instruction
                if steps_in_current_sub >= max_steps_per_sub:
                    ctx.session.state["navigation_status"] = "stuck"
                    ctx.session.state[
                        "stuck_reason"
                    ] = f"Too many steps ({max_steps_per_sub}) for sub-instruction {current_sub_idx}"
                    # Move to next sub-instruction anyway
                    current_sub_idx += 1
                    steps_in_current_sub = 0

        # ==== Store Results ====
        ctx.session.state["navigation_trajectory"] = navigation_trajectory
        ctx.session.state["predicted_path"] = predicted_path
        ctx.session.state["final_node"] = current_node_id
        ctx.session.state["completed_sub_instructions"] = current_sub_idx
        ctx.session.state["total_sub_instructions"] = len(sub_instructions)
        ctx.session.state["total_navigation_steps"] = total_steps
        ctx.session.state["navigation_log"] = navigation_trajectory
