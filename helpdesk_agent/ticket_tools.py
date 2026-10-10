"""ServiceNow tools. Docstrings are what the model reads; every tool returns a dict with "status"."""
import os
import re

from .guardrails import check_fields, redact
from .servicenow import ServiceNowClient
from .telemetry import instrumented

_client = None


def _sn():
    global _client
    if _client is None:
        _client = ServiceNowClient()
    return _client


def _dry_run():
    return os.getenv("DRY_RUN", "true").lower() != "false"


def _clean(row, pii_total):
    out = dict(row)
    for key in ("short_description", "description"):
        if isinstance(out.get(key), str):
            out[key], counts = redact(out[key])
            for label, n in counts.items():
                pii_total[label] = pii_total.get(label, 0) + n
    return out


def _valid_number(number):
    number = number.strip().upper()
    return number if re.fullmatch(r"INC\d{7}", number) else None


@instrumented
def list_open_incidents(limit: int = 5) -> dict:
    """Lists the most recent open incidents in ServiceNow.

    Args:
        limit: How many incidents to return (1 to 20).

    Returns:
        dict with status and incidents (number, short_description, category, urgency, state).
    """
    limit = max(1, min(int(limit), 20))
    pii = {}
    rows = [_clean(r, pii) for r in _sn().list_incidents(limit=limit)]
    keep = ("number", "short_description", "category", "urgency", "state")
    return {"status": "success", "incidents": [{k: r.get(k) for k in keep} for r in rows], "pii_redacted": pii}


@instrumented
def get_incident(number: str) -> dict:
    """Gets one incident by number, for example INC0010001. PII in free text is masked.

    Args:
        number: The incident number.

    Returns:
        dict with status and the incident fields.
    """
    number = _valid_number(number)
    if not number:
        return {"status": "error", "message": "number must look like INC0010001"}
    row = _sn().get_incident(number)
    if not row:
        return {"status": "not_found", "message": f"{number} does not exist"}
    pii = {}
    row = _clean(row, pii)
    row.pop("sys_id", None)
    return {"status": "success", "incident": row, "pii_redacted": pii}


@instrumented
def update_incident(
    number: str,
    category: str,
    urgency: str,
    impact: str,
    work_note: str,
    assignment_group: str = "",
) -> dict:
    """Applies a triage decision to an incident. Only call after the user explicitly approves.

    Args:
        number: The incident number, for example INC0010001.
        category: One of inquiry, software, hardware, network, database.
        urgency: "1" (high), "2" (medium) or "3" (low).
        impact: "1" (high), "2" (medium) or "3" (low).
        work_note: One or two sentences explaining the decision.
        assignment_group: Optional group name, for example "Service Desk".

    Returns:
        dict with status. In dry-run mode nothing is written.
    """
    fields = {
        "category": category.strip().lower(),
        "urgency": str(urgency).strip(),
        "impact": str(impact).strip(),
        "work_notes": f"[AI triage] {work_note.strip()}",
    }
    if assignment_group.strip():
        fields["assignment_group"] = assignment_group.strip()
    violations = check_fields(fields)
    if violations:
        return {"status": "rejected", "violations": violations}

    number = _valid_number(number)
    row = _sn().get_incident(number) if number else None
    if not row:
        return {"status": "not_found", "message": "incident does not exist"}
    if _dry_run():
        return {"status": "dry_run", "number": row["number"], "would_set": fields}

    payload = dict(fields)
    if "assignment_group" in payload:
        group_id = _sn().find_group_id(payload["assignment_group"])
        if not group_id:
            return {"status": "rejected", "violations": [f"group '{payload['assignment_group']}' not found"]}
        payload["assignment_group"] = group_id
    updated = _sn().update_incident(row["sys_id"], payload)
    return {"status": "success", "number": updated.get("number"), "priority": updated.get("priority")}


@instrumented
def create_incident(short_description: str, description: str, category: str, urgency: str, impact: str) -> dict:
    """Opens a new ServiceNow incident. Only call after the user explicitly approves the summary.

    Args:
        short_description: One-line summary, under 100 characters.
        description: What happened, what the user tried, and any policy passage already checked.
        category: One of inquiry, software, hardware, network, database.
        urgency: "1" (high), "2" (medium) or "3" (low).
        impact: "1" (high), "2" (medium) or "3" (low).

    Returns:
        dict with status and the new incident number. In dry-run mode nothing is written.
    """
    short, _ = redact(short_description.strip()[:100])
    desc, pii = redact(description.strip())
    fields = {
        "short_description": short,
        "description": desc,
        "category": category.strip().lower(),
        "urgency": str(urgency).strip(),
        "impact": str(impact).strip(),
    }
    violations = check_fields(fields, require_writable=False)
    if not short:
        violations.append("short_description is empty")
    if violations:
        return {"status": "rejected", "violations": violations}
    if _dry_run():
        return {"status": "dry_run", "would_create": fields, "pii_redacted": pii}
    created = _sn().create_incident(fields)
    return {"status": "success", "number": created.get("number"), "pii_redacted": pii}
