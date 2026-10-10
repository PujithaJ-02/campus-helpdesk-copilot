"""Runs the real ADK runner with a scripted fake model, to prove the callbacks,
session-state roles and tools work together inside ADK (not just in isolation)."""
import asyncio
from typing import AsyncGenerator
from unittest.mock import MagicMock

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from helpdesk_agent import ticket_tools
from helpdesk_agent.callbacks import audit_result, enforce_role
from helpdesk_agent.prompts import TICKET_INSTRUCTION


class ScriptedLlm(BaseLlm):
    """First turn: call the requested tool. Second turn: reply with text."""
    tool_call: dict

    async def generate_content_async(self, llm_request, stream=False) -> AsyncGenerator[LlmResponse, None]:
        already_called = any(
            p.function_response for c in llm_request.contents for p in (c.parts or [])
        )
        if already_called:
            part = types.Part(text="done")
        else:
            part = types.Part(function_call=types.FunctionCall(**self.tool_call))
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def run(role, tool_call, monkeypatch):
    sn = MagicMock()
    sn.get_incident.return_value = {"sys_id": "s1", "number": "INC0010001", "description": "x"}
    monkeypatch.setattr(ticket_tools, "_client", sn)
    agent = Agent(
        name="ticket_agent", model=ScriptedLlm(model="scripted", tool_call=tool_call),
        instruction=TICKET_INSTRUCTION,
        tools=[ticket_tools.get_incident, ticket_tools.update_incident],
        before_tool_callback=enforce_role, after_tool_callback=audit_result,
    )

    async def go():
        runner = InMemoryRunner(agent=agent, app_name="t")
        session = await runner.session_service.create_session(
            app_name="t", user_id="u", state={"user:role": role} if role else {})
        responses = []
        msg = types.Content(role="user", parts=[types.Part(text="hi")])
        async for event in runner.run_async(user_id="u", session_id=session.id, new_message=msg):
            responses.extend(fr.response for fr in event.get_function_responses() or [])
        return responses

    return asyncio.run(go()), sn


def test_requester_is_blocked_inside_adk(monkeypatch):
    call = {"name": "get_incident", "args": {"number": "INC0010001"}}
    responses, sn = run("requester", call, monkeypatch)
    assert responses[0]["status"] == "denied"
    sn.get_incident.assert_not_called()


def test_staff_tool_runs_inside_adk(monkeypatch):
    call = {"name": "get_incident", "args": {"number": "INC0010001"}}
    responses, sn = run("it_staff", call, monkeypatch)
    assert responses[0]["status"] == "success"
    sn.get_incident.assert_called_once()
