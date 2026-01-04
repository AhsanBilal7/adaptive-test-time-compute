from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional, Dict, List, Any


Tool = Literal[
    "reactive_actor",
    "cot",
    "heuristic_script",
    "numeric_verifier",
    "verifier",
    "summarizer",
    "reframe",
]

class ToolsPayload(BaseModel):
    tools: List[Tool] = Field(min_length=1)  # v2 uses min_length
    model_config = ConfigDict(extra="forbid")



class DecisionPayload(BaseModel):
    strategy: Literal["best_of_n", "beam_search", "lookahead"]
    param: int = Field(..., description="Best-of-N/beam: 2–8; lookahead: 1–3")
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_ranges(self):
        if self.strategy in ("best_of_n", "beam_search") and not (2 <= self.param <= 8):
            raise ValueError("param must be in [2,8] for best_of_n/beam_search")
        if self.strategy == "lookahead" and not (1 <= self.param <= 3):
            raise ValueError("param must be in [1,3] for lookahead")
        return self
    



# Pydantic model for structured final answer output
class FinalAnswer(BaseModel):
    """Structured output for final answer."""
    answer: int = Field(description="The final numerical answer or simplified expression")
    confidence: Optional[float] = Field(default=None, description="Confidence score between 0 and 1")
    model_config = ConfigDict(extra="forbid")