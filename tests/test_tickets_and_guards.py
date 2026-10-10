"""ServiceNow is mocked; these run offline with no credentials."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from helpdesk_agent import ticket_tools
from helpdesk_agent.callbacks import audit_result, enforce_role
from helpdesk_agent.guardrails import check_fields, redact, tool_allowed

INCIDENT = {
    "sys_id": "abc123", "number": "INC0010001",
    "short_description": "VPN drops every 10 min",
    "description": "Reach me at jane.doe@stanford.edu or 650-555-0100",
    "category": "Network", "urgency": "2 - Medium", "state": "New",
}


@pytest.fixture
def sn(monkeypatch):
    fake = MagicMock()
    fake.get_incident.return_value = dict(INCIDENT)
    fake.list_incidents.return_value = [dict(INCIDENT)]
    fake.update_incident.return_value = {"number": "INC0010001", "priority": "3 - Moderate"}
    fake.create_incident.return_value = {"number": "INC0010099"}
    fake.find_group_id.return_value = "grp789"
    monkeypatch.setattr(ticket_tools, "_client", fake)
    return fake


def ctx(role=None):
    return SimpleNamespace(state={"user:role": role} if role else {})


def test_redact_masks_pii_but_not_ticket_numbers():
    text, counts = redact("a.b@x.edu, (650) 555-0100, 123-45-6789, see INC0010001")
    assert counts == {"EMAIL": 1, "SSN": 1, "PHONE": 1}
    assert "INC0010001" in text


def test_policy_rejects_bad_fields():
    assert len(check_fields({"state": "7", "category": "payroll", "urgency": "9"})) == 3


def test_roles():
    assert tool_allowed("requester", "create_incident")
    assert not tool_allowed("requester", "update_incident")
    assert tool_allowed("it_staff", "update_incident")
    assert not tool_allowed("unknown_role", "search_policy")


def test_callback_denies_requester_triage():
    out = enforce_role(SimpleNamespace(name="update_incident"), {"number": "INC0010001"}, ctx("requester"))
    assert out["status"] == "denied"


def test_callback_defaults_to_least_privilege():
    out = enforce_role(SimpleNamespace(name="get_incident"), {}, ctx())
    assert out["status"] == "denied"


def test_callback_allows_staff_and_audit_passes_through():
    tool = SimpleNamespace(name="update_incident")
    assert enforce_role(tool, {}, ctx("it_staff")) is None
    assert audit_result(tool, {}, ctx("it_staff"), {"status": "success"}) is None


def test_get_incident_redacts_and_hides_sys_id(sn):
    out = ticket_tools.get_incident("inc0010001")
    assert out["status"] == "success" and "sys_id" not in out["incident"]
    assert "jane.doe" not in out["incident"]["description"]


def test_bad_incident_number_never_hits_servicenow(sn):
    assert ticket_tools.get_incident("12345")["status"] == "error"
    sn.get_incident.assert_not_called()


def test_update_is_dry_run_by_default(sn, monkeypatch):
    monkeypatch.delenv("DRY_RUN", raising=False)
    assert ticket_tools.update_incident("INC0010001", "network", "2", "3", "VPN")["status"] == "dry_run"
    sn.update_incident.assert_not_called()


def test_update_resolves_group_when_live(sn, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "false")
    out = ticket_tools.update_incident("INC0010001", "network", "2", "3", "VPN", "Network")
    assert out["status"] == "success"
    sys_id, fields = sn.update_incident.call_args.args
    assert sys_id == "abc123" and fields["assignment_group"] == "grp789"


def test_create_incident_redacts_pii_before_writing(sn, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "false")
    out = ticket_tools.create_incident("Laptop stolen", "Call me at 650-555-0100", "hardware", "1", "3")
    assert out["status"] == "success" and out["number"] == "INC0010099"
    written = sn.create_incident.call_args.args[0]
    assert "650-555-0100" not in written["description"]


def test_create_incident_rejects_bad_category(sn, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "false")
    assert ticket_tools.create_incident("x", "y", "payroll", "1", "3")["status"] == "rejected"
    sn.create_incident.assert_not_called()


def test_servicenow_failure_becomes_tool_error(sn):
    sn.list_incidents.side_effect = RuntimeError("HTTP 401")
    out = ticket_tools.list_open_incidents(3)
    assert out["status"] == "error" and "401" in out["message"]
