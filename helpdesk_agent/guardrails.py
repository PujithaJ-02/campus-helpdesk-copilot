"""Guardrails: PII redaction, write policy, and role-based tool access.

Policy is plain data (POLICY below) so it can be reviewed like config,
and every tool call is checked against it before it runs.
"""
import os
import re

_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")),
]

POLICY = {
    # Which tools each role may call. Anything not listed is denied.
    "roles": {
        "requester": {"search_policy", "create_incident"},
        "it_staff": {
            "search_policy", "create_incident", "list_open_incidents",
            "get_incident", "update_incident",
        },
    },
    "writable_fields": {"category", "urgency", "impact", "assignment_group", "work_notes"},
    "categories": {"inquiry", "software", "hardware", "network", "database"},
    "levels": {"1", "2", "3"},  # ServiceNow impact/urgency: 1=High, 2=Medium, 3=Low
    "max_note_chars": 2000,
    "max_description_chars": 4000,
}
# Least privilege unless the deployment says otherwise.
DEFAULT_ROLE = os.getenv("DEFAULT_ROLE", "requester")


def redact(text):
    """Replace PII with typed placeholders. Returns (clean_text, counts)."""
    if not text:
        return text, {}
    counts = {}
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts


def tool_allowed(role, tool_name):
    return tool_name in POLICY["roles"].get(role, set())


def check_fields(fields, require_writable=True):
    """Return a list of policy violations. Empty list means the write is allowed."""
    errors = []
    for key, value in fields.items():
        if require_writable and key not in POLICY["writable_fields"]:
            errors.append(f"field '{key}' is not writable by the agent")
        elif key == "category" and value not in POLICY["categories"]:
            errors.append(f"category '{value}' not in {sorted(POLICY['categories'])}")
        elif key in ("urgency", "impact") and str(value) not in POLICY["levels"]:
            errors.append(f"{key} must be 1, 2 or 3")
    if len(fields.get("work_notes", "")) > POLICY["max_note_chars"]:
        errors.append("work_notes too long")
    if len(fields.get("description", "")) > POLICY["max_description_chars"]:
        errors.append("description too long")
    return errors
