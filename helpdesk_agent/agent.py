import logging
import os

from google.adk.agents import Agent
from google.adk.models import Gemini

from .callbacks import audit_result, enforce_role
from .policy_tools import search_policy
from .prompts import POLICY_INSTRUCTION, PROMPT_VERSION, ROUTER_INSTRUCTION, TICKET_INSTRUCTION
from .ticket_tools import create_incident, get_incident, list_open_incidents, update_incident

logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL = os.getenv("HELPDESK_MODEL", "gemini-3.5-flash")
# Some newer Gemini models are only served from the global endpoint. Agent Engine
# overrides GOOGLE_CLOUD_LOCATION with the deploy region, so pin the model's own
# location here instead of relying on that variable.
MODEL_LOCATION = os.getenv("HELPDESK_MODEL_LOCATION", "global")


def build_model():
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
    if not use_vertex:
        return MODEL  # API key mode: plain model name
    return Gemini(model=MODEL, client_kwargs={"vertexai": True, "location": MODEL_LOCATION})


LLM = build_model()
GUARDS = {"before_tool_callback": enforce_role, "after_tool_callback": audit_result}

policy_agent = Agent(
    name="policy_agent",
    model=LLM,
    description="Answers university policy questions from the Stanford Administrative Guide, with sources.",
    instruction=POLICY_INSTRUCTION,
    tools=[search_policy],
    **GUARDS,
)

ticket_agent = Agent(
    name="ticket_agent",
    model=LLM,
    description="Opens, looks up and triages ServiceNow incidents, with user approval before any write.",
    instruction=TICKET_INSTRUCTION,
    tools=[create_incident, list_open_incidents, get_incident, update_incident],
    **GUARDS,
)

root_agent = Agent(
    name="helpdesk_router",
    model=LLM,
    description=f"Campus help desk copilot (prompt {PROMPT_VERSION}).",
    instruction=ROUTER_INSTRUCTION,
    sub_agents=[policy_agent, ticket_agent],
)
