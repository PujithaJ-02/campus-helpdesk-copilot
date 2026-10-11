"""End-to-end answer eval with an LLM judge.

    python -m scripts.eval_answers

Runs each golden question through policy_agent, captures the passages its
search_policy call returned, then asks a judge model whether the answer is
faithful to those passages, cites a source, and answers the question.
"""
import argparse
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


def is_rate_limit(exc):
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


async def with_backoff(make_call, label, tries=5):
    """Retry on 429 with growing waits (10s, 20s, 40s, 80s). New projects have low quotas."""
    for attempt in range(tries):
        try:
            return await make_call()
        except Exception as exc:
            if not is_rate_limit(exc) or attempt == tries - 1:
                raise
            wait = 10 * 2 ** attempt
            print(f"  rate limited on {label}, waiting {wait}s")
            await asyncio.sleep(wait)


async def ask(runner, question):
    """Returns (answer, passages, number_of_search_calls)."""
    session = await runner.session_service.create_session(
        app_name="eval", user_id="eval", state={"user:role": "requester"})
    msg = types.Content(role="user", parts=[types.Part(text=question)])
    answer, passages, searches = "", [], 0
    async for event in runner.run_async(user_id="eval", session_id=session.id, new_message=msg):
        for fr in event.get_function_responses() or []:
            if fr.name == "search_policy":
                searches += 1
                passages.extend((fr.response or {}).get("passages", []))
        if event.is_final_response() and event.content and event.content.parts:
            answer = "".join(p.text or "" for p in event.content.parts)
    return answer, passages, searches


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


async def main(golden_path, limit, pause):
    golden = [json.loads(line) for line in Path(golden_path).read_text().splitlines() if line.strip()]
    golden = golden[:limit] if limit else golden
    runner = InMemoryRunner(agent=policy_agent, app_name="eval")
    client = genai.Client()

    rows, latencies, failed = [], [], []
    for n, item in enumerate(golden, 1):
        try:
            start = time.perf_counter()
            answer, passages, searches = await with_backoff(lambda: ask(runner, item["question"]), "agent")
            latencies.append((time.perf_counter() - start) * 1000)
            verdict = await with_backoff(
                lambda: asyncio.to_thread(judge, client, item["question"], answer, passages), "judge")
        except Exception as exc:  # one bad question shouldn't lose the whole run
            failed.append({"question": item["question"], "error": str(exc)[:200]})
            print(f"[{n}/{len(golden)}] FAILED: {str(exc)[:100]}")
            continue
        rows.append({**item, "answer": answer, "verdict": verdict, "searches": searches,
                     "retrieved_urls": [p["url"] for p in passages]})
        flags = "".join("Y" if verdict.get(k) else "n" for k in ("faithful", "cites_source", "answers_question"))
        print(f"[{n}/{len(golden)}] {flags}  searches={searches}  {item['question'][:60]}")
        await asyncio.sleep(pause)  # stay under the per-minute quota

    if not rows:
        print("No questions completed.")
        return

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
        "avg_searches": round(sum(r["searches"] for r in rows) / len(rows), 2),
        "failed": len(failed),
    }
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/answers.json").write_text(json.dumps({"summary": summary, "rows": rows, "failed": failed}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/policy_golden.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N questions")
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between questions")
    args = parser.parse_args()
    asyncio.run(main(args.golden, args.limit, args.pause))
