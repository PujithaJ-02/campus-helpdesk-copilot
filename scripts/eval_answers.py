"""End-to-end answer eval with an LLM judge.

    python -m scripts.eval_answers

Runs each golden question through policy_agent, captures the passages its
search_policy call returned, then asks a judge model whether the answer is
faithful to those passages, cites a source, and answers the question.
"""
import asyncio
import json
import os
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "helpdesk_agent" / ".env")

from google import genai  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from helpdesk_agent.agent import MODEL, policy_agent  # noqa: E402
from helpdesk_agent.prompts import JUDGE_INSTRUCTION, PROMPT_VERSION  # noqa: E402

JUDGE_MODEL = os.getenv("JUDGE_MODEL", MODEL)


async def ask(runner, question):
    session = await runner.session_service.create_session(
        app_name="eval", user_id="eval", state={"user:role": "requester"})
    msg = types.Content(role="user", parts=[types.Part(text=question)])
    answer, passages = "", []
    async for event in runner.run_async(user_id="eval", session_id=session.id, new_message=msg):
        for fr in event.get_function_responses() or []:
            if fr.name == "search_policy":
                passages.extend((fr.response or {}).get("passages", []))
        if event.is_final_response() and event.content and event.content.parts:
            answer = "".join(p.text or "" for p in event.content.parts)
    return answer, passages


def judge(client, question, answer, passages):
    prompt = (
        f"{JUDGE_INSTRUCTION}\n\nQUESTION:\n{question}\n\nPASSAGES:\n"
        + json.dumps(passages, indent=1)[:12000]
        + f"\n\nANSWER:\n{answer}"
    )
    resp = client.models.generate_content(
        model=JUDGE_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    match = re.search(r"\{.*\}", resp.text or "", re.S)
    return json.loads(match.group(0)) if match else {}


async def main(golden_path="eval/policy_golden.jsonl"):
    golden = [json.loads(line) for line in Path(golden_path).read_text().splitlines() if line.strip()]
    runner = InMemoryRunner(agent=policy_agent, app_name="eval")
    client = genai.Client()

    rows, latencies = [], []
    for item in golden:
        start = time.perf_counter()
        answer, passages = await ask(runner, item["question"])
        latencies.append((time.perf_counter() - start) * 1000)
        verdict = judge(client, item["question"], answer, passages)
        rows.append({**item, "answer": answer, "verdict": verdict,
                     "retrieved_urls": [p["url"] for p in passages]})
        print(f"{item['question'][:60]:60}  {verdict}")

    def rate(key):
        return round(sum(bool(r["verdict"].get(key)) for r in rows) / len(rows), 3)

    summary = {
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "judge_model": JUDGE_MODEL,
        "n": len(rows),
        "faithful_rate": rate("faithful"),
        "citation_rate": rate("cites_source"),
        "answered_rate": rate("answers_question"),
        "latency_ms_p50": round(statistics.median(latencies)),
    }
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/answers.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
