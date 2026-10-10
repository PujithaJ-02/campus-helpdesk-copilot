"""Offline eval: does the model pick the same category a human did?

Pulls already-categorized incidents from ServiceNow, hides the human label,
asks a tool-less classifier (same triage rules as the agent) to label each one,
and reports accuracy plus latency. Results go to eval/results/triage.json.

Run from the project root:
    python -m scripts.eval_triage --limit 50
"""
import argparse
import asyncio
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "helpdesk_agent" / ".env")

from google.adk.agents import Agent  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from helpdesk_agent.agent import MODEL  # noqa: E402
from helpdesk_agent.guardrails import POLICY, redact  # noqa: E402
from helpdesk_agent.prompts import CLASSIFY_INSTRUCTION, PROMPT_VERSION  # noqa: E402
from helpdesk_agent.servicenow import ServiceNowClient  # noqa: E402

classifier = Agent(name="triage_classifier", model=MODEL, instruction=CLASSIFY_INSTRUCTION, tools=[])


async def classify(runner, text):
    session = await runner.session_service.create_session(app_name="eval", user_id="eval")
    msg = types.Content(role="user", parts=[types.Part(text=text)])
    reply = ""
    async for event in runner.run_async(user_id="eval", session_id=session.id, new_message=msg):
        if event.is_final_response() and event.content and event.content.parts:
            reply = event.content.parts[0].text or ""
    match = re.search(r"\{.*\}", reply, re.S)
    return json.loads(match.group(0)) if match else {}


async def main(limit):
    sn = ServiceNowClient()
    cats = "^OR".join(f"category={c}" for c in sorted(POLICY["categories"]))
    rows = sn.list_incidents(query=f"{cats}^short_descriptionISNOTEMPTY", limit=limit)
    # display values come back capitalized ("Software"), so normalize
    runner = InMemoryRunner(agent=classifier, app_name="eval")

    results, latencies = [], []
    for row in rows:
        gold = (row.get("category") or "").strip().lower()
        if gold not in POLICY["categories"]:
            continue
        text, _ = redact(f"{row.get('short_description', '')}\n{row.get('description', '')}")
        start = time.perf_counter()
        try:
            pred = await classify(runner, text)
        except Exception as exc:  # keep going, count as a miss
            pred = {"error": str(exc)}
        latencies.append((time.perf_counter() - start) * 1000)
        results.append({
            "number": row["number"],
            "gold": gold,
            "pred": str(pred.get("category", "")).lower(),
        })

    if not results:
        print("No categorized incidents found. Check SN_* settings.")
        return

    correct = sum(r["gold"] == r["pred"] for r in results)
    confusion = Counter((r["gold"], r["pred"]) for r in results if r["gold"] != r["pred"])
    summary = {
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "n": len(results),
        "category_accuracy": round(correct / len(results), 3),
        "latency_ms_p50": round(statistics.median(latencies)),
        "latency_ms_max": round(max(latencies)),
        "top_confusions": [f"{g} -> {p}: {n}" for (g, p), n in confusion.most_common(5)],
    }
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/triage.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    asyncio.run(main(parser.parse_args().limit))
