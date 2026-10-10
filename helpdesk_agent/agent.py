import logging
import os

from google.adk.agents import Agent

from .callbacks import audit_result, enforce_role
from .policy_tools import search_policy
from .prompts import POLICY_INSTRUCTION, PROMPT_VERSION, ROUTER_INSTRUCTION, TICKET_INSTRUCTION
from .ticket_tools import create_incident, get_incident, list_open_incidents, update_incident

logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL = os.getenv("HELPDESK_MODEL", "gemini-flash-latest")
GUARDS = {"before_tool_callback": enforce_role, "after_tool_callback": audit_result}

policy_agent = Agent(
    name="policy_agent",
    model=MODEL,
    description="Answers university policy questions from the Stanford Administrative Guide, with sources.",
    instruction=POLICY_INSTRUCTION,
    tools=[search_policy],
    **GUARDS,
)

ticket_agent = Agent(
    name="ticket_agent",
    model=MODEL,
    description="Opens, looks up and triages ServiceNow incidents, with user approval before any write.",
    instruction=TICKET_INSTRUCTION,
    tools=[create_incident, list_open_incidents, get_incident, update_incident],
    **GUARDS,
)

root_agent = Agent(
    name="helpdesk_router",
    model=MODEL,
    description=f"Campus help desk copilot (prompt {PROMPT_VERSION}).",
    instruction=ROUTER_INSTRUCTION,
    sub_agents=[policy_agent, ticket_agent],
)
