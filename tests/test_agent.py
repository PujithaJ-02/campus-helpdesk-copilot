"""Checks the agent graph and the tool schemas ADK will send to Gemini."""
from google.adk.tools import FunctionTool

from helpdesk_agent.agent import policy_agent, root_agent, ticket_agent


def test_router_has_both_specialists():
    assert {a.name for a in root_agent.sub_agents} == {"policy_agent", "ticket_agent"}


def test_specialists_have_guard_callbacks():
    for agent in (policy_agent, ticket_agent):
        assert agent.before_tool_callback is not None
        assert agent.after_tool_callback is not None


def test_tool_schemas_keep_real_parameters_through_decorator():
    tool = FunctionTool([t for t in ticket_agent.tools if t.__name__ == "create_incident"][0])
    decl = tool._get_declaration()
    # ADK 2.x puts the schema in parameters_json_schema; older versions used parameters.
    schema = decl.parameters_json_schema or decl.parameters.model_dump()
    params = set(schema["properties"])
    assert params == {"short_description", "description", "category", "urgency", "impact"}
    assert "Opens a new ServiceNow incident" in decl.description


def test_prompts_forbid_collecting_personal_details_and_self_transfer():
    from helpdesk_agent import prompts
    for text in (prompts.ROUTER_INSTRUCTION, prompts.POLICY_INSTRUCTION, prompts.TICKET_INSTRUCTION):
        assert "Never ask for the user's name, email, phone number, SUNet ID" in text
    assert "never call transfer_to_agent to reach ticket_agent" in prompts.TICKET_INSTRUCTION
