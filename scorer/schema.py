from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict


class SubInstruction(BaseModel):
    """A single sub-instruction extracted from the full navigation instruction"""
    instruction_text: str = Field(description="The text of this sub-instruction")
    # mentioned_landmarks: List[str] = Field(
    #     default_factory=list,
    #     description="List of landmarks/POIs mentioned in this sub-instruction"
    # )
    # action_type: Literal["start", "turn", "continue", "pass", "arrive"] = Field(
    #     description="The primary action type of this sub-instruction"
    # )
    # direction: Optional[Literal["left", "right", "straight", "back", "unknown"]] = Field(
    #     default="unknown",
    #     description="Direction of movement if applicable"
    # )


class NextNodePrediction(BaseModel):
    """Prediction of the next node to navigate to"""
    next_node_id: str = Field(description="The OSM ID of the next node to navigate to")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this prediction (0.0 to 1.0)"
    )
    reasoning: str = Field(description="Brief explanation of why this node was chosen")
    matched_pois: List[str] = Field(
        default_factory=list,
        description="POI IDs that were matched at or near this node"
    )
    sub_instruction_satisfied: bool = Field(
        description="Whether the current sub-instruction is considered satisfied after reaching this node"
    )


from typing import List, Literal, Optional
from pydantic import BaseModel

# --- ENUMS ---

ActionType = Literal[
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT"
]

StatusType = Literal[
    "TODO",
    "IN_PROGRESS",
    "COMPLETED"
]


# --- CORE SCHEMAS ---

class Landmark(BaseModel):
    """
    Represents a physical map entity (POI) found in the navigation instruction.
    """
    name: str
    category: Optional[str] = None  # e.g., 'traffic_light', 'bank', 'restaurant', etc.
    # relative_position: Literal[
    #     "left",
    #     "right",
    #     "ahead",
    #     "on the corner",
    #     "across",
    #     "next to",
    #     "unknown"
    # ]


# action: ActionType  # one of MOVE_FORWARD / TURN_LEFT / TURN_RIGHT
class SubGoal(BaseModel):
    """
    Represents a single actionable navigation step.
    """
    description: str  # natural-language paraphrase
    status: StatusType = "TODO"


class InstructionDivision(BaseModel):
    """
    Final structured response for the navigation instruction.
    """
    landmarks: List[Landmark]
    sub_goals: List[SubGoal]


SubPlanStatus = Literal[
    "IN_PROGRESS",
    "COMPLETED"
]


class Thought(BaseModel):
    """
    Internal reasoning steps the agent uses to select the next move.
    """
    orientation_analysis: str  # description of current heading alignment
    landmark_visible: bool  # whether expected 'L' is observed in matrix
    decision_summary: str  # explanation of chosen next cell


class NavigationStepOutput(BaseModel):
    """
    Final structured reasoning output for matrix-based navigation.
    """
    # thought: Thought
    subplan_status: SubPlanStatus
    next_place: List[int]  # Movement target for the next step. [row, col]
