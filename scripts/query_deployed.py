"""Talk to the agent after it is deployed to Vertex AI Agent Engine.

    python scripts/query_deployed.py RESOURCE_NAME "How do I report a stolen laptop?" --role requester

RESOURCE_NAME is printed at the end of `adk deploy agent_engine`
(projects/.../locations/.../reasoningEngines/...). If SDK method names have
changed, check https://docs.cloud.google.com/agent-builder/agent-engine/use/adk
"""
import argparse

import vertexai
from vertexai import agent_engines

parser = argparse.ArgumentParser()
parser.add_argument("resource")
parser.add_argument("message")
parser.add_argument("--role", default="requester", choices=["requester", "it_staff"])
args = parser.parse_args()

parts = args.resource.split("/")
vertexai.init(project=parts[1], location=parts[3])
app = agent_engines.get(args.resource)

session = app.create_session(user_id="demo-user", state={"user:role": args.role})
for event in app.stream_query(user_id="demo-user", session_id=session["id"], message=args.message):
    for part in event.get("content", {}).get("parts", []):
        if "text" in part:
            print(part["text"])
        elif "function_call" in part:
            print(f"[tool call] {part['function_call'].get('name')}")
