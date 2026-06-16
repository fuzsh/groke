from typing import Optional

import litellm
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from groke.prompts import INSTRUCTION_DIVIDER_PROMPT, NODE_NAVIGATOR_PROMPT, CODE_GENERATION_PROMPT
from .orchestrator import NavigationOrchestrator
from .schema import InstructionDivision, NavigationStepOutput

# litellm.api_key = "G4aNJjsWhnxYOmJRmZ9iB61zokKJ2eMVqsqncytZoXfETlYJQITGJQQJ99BIACfhMk5XJ3w3AAAAACOG5yaV"
# litellm.api_base = "https://feri.cognitiveservices.azure.com/"
# litellm.api_version = "2024-12-01-preview"
# MODEL_NAME = "gpt-4o-mini"

# model = LiteLlm(model=f"azure/{MODEL_NAME}")
# model = LiteLlm(model="xai/grok-4-0709")
model = 'gemini-3-pro-preview'


# Create Instruction Divider Agent
instruction_divider = LlmAgent(
    name="InstructionDivider",
    model=model,
    instruction=INSTRUCTION_DIVIDER_PROMPT,
    output_schema=InstructionDivision,
    output_key="instruction_division"
)

# Create Node Navigator Agent
node_navigator = LlmAgent(
    name="NodeNavigator",
    model=model,
    generate_content_config=types.GenerateContentConfig(
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(initial_delay=1, attempts=2),
        ),
    ),
    # instruction=CODE_GENERATION_PROMPT,
    instruction=NODE_NAVIGATOR_PROMPT,
    include_contents='none',
    output_schema=NavigationStepOutput,
    output_key="next_node_prediction"
)

# Create the Navigation Orchestrator with all agents
pipeline_agent = NavigationOrchestrator(
    name="NavigationOrchestrator",
    instruction_divider=instruction_divider,
    node_navigator=node_navigator
)

root_agent = pipeline_agent