"""Structured logs: one JSON line per tool call and per policy decision.

On Agent Engine, stdout goes to Cloud Logging, where these lines can be
filtered (jsonPayload.event="tool_call") and charted for latency and errors.
"""
import json
import logging
import time
from functools import wraps

log = logging.getLogger("helpdesk_agent")


def emit(event, **fields):
    log.info(json.dumps({"event": event, "ts": round(time.time(), 3), **fields}))


def instrumented(fn):
    """Time a tool and log its outcome. Converts unexpected errors into a tool error dict."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        status = "error"
        try:
            result = fn(*args, **kwargs)
            status = result.get("status", "unknown")
            return result
        except Exception as exc:  # the model gets a readable error instead of a crash
            return {"status": "error", "message": f"{type(exc).__name__}: {exc}"[:300]}
        finally:
            emit("tool_call", tool=fn.__name__, status=status,
                 latency_ms=round((time.perf_counter() - start) * 1000, 1))
    return wrapper
