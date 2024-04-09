from typing import TypedDict, Annotated
from pydantic import BaseModel, Field

class Agent(TypedDict):
    name: str # the name of the agent
    objective: str # what the agent should do e.g. "Debate the chicken or the egg with the other AI"
    model: str # optional, model to use for this specific agent, if not provided use default

AgentDetails = tuple[Agent, Agent]

class Actor(BaseModel):
    thought: str = Field(..., description="scratchpad for step-by-step reasoning")
    action: str = Field(..., description="actions to be exectued to accomplish the task")

class Critic(BaseModel):
    evaluation: str = Field(..., description="evaluation of the actor's actions")
    feedback: str = Field(..., description="constructive feedback to revise actions")