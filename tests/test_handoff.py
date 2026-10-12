"""Runs the real agent graph with a scripted fake model (no network, no Gemini).

Live testing showed a requester asking for a ticket after a policy question got
"unable to transfer due to a system restriction": the role callback was denying
ADK's transfer_to_agent tool. This replays that conversation end to end.
"""
import asyncio
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from helpdesk_agent import agent as A


def _who(req):
    si = str(req.config.system_instruction or "")
    if "You are ticket_agent" in si:
        return "ticket_agent"
    if "front desk" in si:
        return "helpdesk_router"
    return "policy_agent"


def _text(t):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=t)]))


def _transfer(name):
    call = types.FunctionCall(name="transfer_to_agent", args={"agent_name": name})
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))


class ScriptedLlm(BaseLlm):
    """Router sends vehicle questions to policy_agent, everything else to ticket_agent.
    policy_agent answers vehicle questions and hands anything else to ticket_agent."""
    model: str = "scripted"

    async def generate_content_async(self, req, stream=False) -> AsyncGenerator[LlmResponse, None]:
        me = _who(req)
        user = next(p.text for c in reversed(req.contents) if c.role == "user"
                    for p in (c.parts or []) if p.text and not p.text.startswith("For context"))
        if me == "ticket_agent":
            yield _text("ticket_agent here: Should I submit this?")
        elif "vehicle" in user:
            yield _text("policy answer") if me == "policy_agent" else _transfer("policy_agent")
        else:
            yield _transfer("ticket_agent")


def test_requester_reaches_ticket_agent_after_a_policy_question(monkeypatch):
    fake = ScriptedLlm()
    for a in (A.root_agent, A.policy_agent, A.ticket_agent):
        monkeypatch.setattr(a, "model", fake)

    async def run():
        runner = InMemoryRunner(agent=A.root_agent, app_name="t")
        s = await runner.session_service.create_session(
            app_name="t", user_id="u", state={"user:role": "requester"})
        events = []
        for q in ["Can I use a university vehicle for errands?",
                  "My wifi keeps disconnecting in the library, can you open a ticket?"]:
            msg = types.Content(role="user", parts=[types.Part(text=q)])
            async for e in runner.run_async(user_id="u", session_id=s.id, new_message=msg):
                events.append(e)
        return events

    events = asyncio.run(run())
    denied = [fr for e in events for fr in (e.get_function_responses() or [])
              if (fr.response or {}).get("status") == "denied"]
    assert not denied, denied
    assert any(e.author == "ticket_agent" and e.content and e.content.parts
               and "Should I submit" in (e.content.parts[0].text or "") for e in events)
