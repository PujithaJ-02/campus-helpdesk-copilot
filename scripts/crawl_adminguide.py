"""Crawl public Stanford Administrative Guide pages into data/pages.jsonl.

    python -m scripts.crawl_adminguide --chapters computing purchasing-and-payments --max-pages 200

Stays inside the chapters you list, waits between requests, and stores each
page as (title, [(section heading, text), ...]) so chunking can keep sections intact.
"""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin, urldefrag

import requests
from bs4 import BeautifulSoup

BASE = "https://adminguide.stanford.edu"
DEFAULT_CHAPTERS = [
    "computing",
    "purchasing-and-payments",
    "human-resources",
    "financial-administration",
    "health-and-safety",
    "services",
]
HEADERS = {"User-Agent": "campus-helpdesk-copilot/0.1 (student portfolio project)"}


def extract(html, url, chapter):
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.find("article") or soup.body
    for tag in main.find_all(["nav", "header", "footer", "script", "style", "aside", "form"]):
        tag.decompose()
    h1 = main.find("h1") or soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else url.rsplit("/", 1)[-1]

    sections, heading = [], title
    for el in main.find_all(["h2", "h3", "h4", "p", "li", "td"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h2", "h3", "h4"):
            heading = text
        elif el.find_parent(["li", "td"]) is None or el.name in ("li", "td"):
            sections.append((heading, text))
    return {"url": url, "chapter": chapter, "title": title, "sections": sections}


def links_in(html, url, prefix):
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select("a[href]"):
        link = urldefrag(urljoin(url, a["href"]))[0].split("?")[0].rstrip("/")
        if link.startswith(prefix):
            yield link


def crawl(chapters, max_pages, delay):
    pages, seen = [], set()
    session = requests.Session()
    session.headers.update(HEADERS)
    for chapter in chapters:
        prefix = f"{BASE}/chapters/{chapter}"
        queue = [prefix]
        while queue and len(pages) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            try:
                resp = session.get(url, timeout=20)
            except requests.RequestException as exc:
                print(f"skip {url}: {exc}")
                continue
            if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            page = extract(resp.text, url, chapter)
            if page["sections"]:
                pages.append(page)
                print(f"[{len(pages)}] {page['title'][:70]}")
            queue.extend(link for link in links_in(resp.text, url, prefix) if link not in seen)
            time.sleep(delay)
    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", nargs="+", default=DEFAULT_CHAPTERS)
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--out", default="data/pages.jsonl")
    args = parser.parse_args()

    pages = crawl(args.chapters, args.max_pages, args.delay)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for page in pages:
            f.write(json.dumps(page) + "\n")
    print(f"wrote {len(pages)} pages to {out}")
