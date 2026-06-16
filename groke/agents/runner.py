import asyncio

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langfuse import Langfuse
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from groke.scorer.agent import root_agent
from groke.data_loader import get_data_by_instruction

# Langfuse Configuration
# langfuse = Langfuse(
#     secret_key="sk-lf-1163bf21-9d2e-4aca-aced-ff7235591541",
#     public_key="pk-lf-71ae0d9a-5491-4061-aa2c-fc43a6915da5",
#     host="http://localhost:3000"
# )
#
# # Verify connection
# if langfuse.auth_check():
#     print("Langfuse client is authenticated and ready!")
# else:
#     print("Authentication failed. Please check your credentials and host.")
#
# GoogleADKInstrumentor().instrument()


async def call_agent_async(query: str, runner, user_id, session_id):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        print(f"[Event] {event}")
        # print(f"[Event] Author: {event.author},\t Answer: {event.content.parts[0].text}")

    return final_response_text


async def run_navigation():
    """
    Run a complete navigation example.
    """
    # Step 2: Create the navigation orchestrator
    print("\n[2/5] Creating navigation orchestrator...")
    navigation_agent = root_agent

    # Step 3: Load navigation data
    print("\n[3/5] Loading navigation data...")
    instruction_id = 6220
    split_file = "paper_results/test_seen.json"

    area_data = get_data_by_instruction(
        instruction_id,
        split_file,
        base_path='./data/map2seq/',
        neighbor_degrees=20
    )

    instruction_data = area_data.get("instruction_data", {})
    navigation_instruction = instruction_data.get("instructions", "")

    if not navigation_instruction:
        return

    print(f"✓ Data loaded for instruction {instruction_id}")
    print(f"  Instruction: \"{navigation_instruction}\"")
    print(f"  Ground truth path length: {len(area_data['instruction_data']['route']['osm_path'])} nodes")
    print(f"  Area nodes: {len(area_data['area_nodes'])}")
    print(f"  Area POIs: {len(area_data['area_pois'])}")

    # Step 4: Initialize and run the agent
    print("\n[4/5] Running navigation agent...")
    print("-" * 70)

    # Create session service and runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=navigation_agent,
        app_name="navigation_app",
        session_service=session_service
    )

    # Create session and inject area_data
    user_id = "test_user"
    session_id = "nav_session_1"

    # Initialize session state with area_data
    await session_service.create_session(
        app_name="navigation_app",
        user_id=user_id,
        session_id=session_id,
        state={"area_data": area_data, 'navigation_instruction': navigation_instruction}
    )

    # Run the agent
    try:
        response = await asyncio.wait_for(
            call_agent_async(
                query=navigation_instruction,
                runner=runner, user_id=user_id, session_id=session_id,
            ),
            timeout=1800  # seconds
        )

        # Step 5: Display results
        print("\n[5/5] Navigation Results:")
        print("=" * 70)

        # Get final session state
        final_session = await session_service.get_session(
            app_name="navigation_app",
            user_id=user_id,
            session_id=session_id
        )
        state = final_session.state

        # Display path information
        predicted_path = state.get("predicted_path", [])
        ground_truth_path = state.get("ground_truth_path", [])

        print(f"\nPredicted Path Length: {len(predicted_path)} nodes")
        print(f"Ground Truth Path Length: {len(ground_truth_path)} nodes")
        print(f"Total Navigation Steps: {state.get('total_navigation_steps', 0)}")
        print(f"Path Distance: {state.get('navigator_path_length', 0):.1f} meters")

        # Display evaluation metrics
        print("\n" + state.get("evaluation_report", "No evaluation available"))

        # Display navigation log summary
        nav_log = state.get("navigation_log", [])
        if nav_log:
            print("\nNavigation Log (last 5 steps):")
            print("-" * 70)
            for log_entry in nav_log[-5:]:
                print(f"Step {log_entry.get('step', '?')}:")
                print(f"  From: {log_entry.get('from_node', 'N/A')}")
                print(f"  To: {log_entry.get('to_node', 'N/A')}")
                print(f"  Confidence: {log_entry.get('confidence', 0):.2f}")
                print(f"  Reasoning: {log_entry.get('reasoning', 'N/A')[:60]}...")
                print()

    except Exception as e:
        print(f"\n✗ Error during navigation: {str(e)}")
        import traceback
        traceback.print_exc()

    print("=" * 70)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(run_navigation())
