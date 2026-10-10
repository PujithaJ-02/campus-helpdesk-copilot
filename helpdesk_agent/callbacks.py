"""ADK callbacks: role-based access control and an audit trail for every tool call.

The role lives in session state under "user:role" (set by the UI or API caller).
Missing role falls back to DEFAULT_ROLE, the least-privileged one.
"""
from .guardrails import DEFAULT_ROLE, redact, tool_allowed
from .telemetry import emit


def _role(tool_context):
    try:
        return tool_context.state.get("user:role") or DEFAULT_ROLE
    except Exception:
        return DEFAULT_ROLE


def enforce_role(tool, args, tool_context):
    """Runs before every tool. Returning a dict skips the tool and hands that dict to the model."""
    role = _role(tool_context)
    allowed = tool_allowed(role, tool.name)
    safe_args = {k: redact(v)[0] if isinstance(v, str) else v for k, v in args.items()}
    emit("tool_request", tool=tool.name, role=role, allowed=allowed, args=safe_args)
    if not allowed:
        return {
            "status": "denied",
            "message": f"The '{role}' role cannot use {tool.name}. Offer to open a ticket instead.",
        }
    return None


def audit_result(tool, args, tool_context, tool_response):
    """Runs after every tool. Logs the outcome without changing it."""
    status = tool_response.get("status") if isinstance(tool_response, dict) else "unknown"
    emit("tool_result", tool=tool.name, role=_role(tool_context), status=status)
    return None
