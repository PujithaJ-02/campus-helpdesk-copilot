from helpdesk_agent import policy_tools
from helpdesk_agent.rag.chunking import chunk_page
from scripts.crawl_adminguide import extract, links_in


def test_chunks_stay_inside_sections_and_carry_title():
    page = {
        "url": "u", "chapter": "computing", "title": "Info Security",
        "sections": [("Scope", "a " * 300), ("Reporting", "report it now")],
    }
    chunks = chunk_page(page, max_words=200, overlap=50)
    sections = [c["section"] for c in chunks]
    assert sections == ["Scope", "Scope", "Reporting"]
    assert chunks[-1]["text"].startswith("Info Security > Reporting\n")
    assert len(chunks[0]["text"].split("\n", 1)[1].split()) == 200


def test_overlap_repeats_words_between_chunks():
    words = " ".join(f"w{i}" for i in range(250))
    page = {"url": "u", "chapter": "c", "title": "T", "sections": [("S", words)]}
    first, second = chunk_page(page, max_words=200, overlap=40)
    assert second["text"].split("\n", 1)[1].split()[0] == "w160"


def test_hybrid_finds_the_right_policy(retriever):
    hits = retriever.search("I lost my laptop, who do I report it to?", k=2)
    assert hits[0]["id"] == "a"


def test_chapter_filter_never_leaks_other_chapters(retriever):
    hits = retriever.search("password laptop travel", k=3, chapter="purchasing-and-payments")
    assert hits and all(h["chapter"] == "purchasing-and-payments" for h in hits)


def test_ablation_modes_run(retriever):
    for mode in ("bm25", "dense", "hybrid"):
        assert retriever.search("travel card personal charges", k=1, mode=mode)[0]["id"] == "c"


def test_search_policy_redacts_and_ignores_unknown_chapter(retriever, monkeypatch):
    monkeypatch.setattr(policy_tools, "_retriever", retriever)
    out = policy_tools.search_policy("personal charges on travel card", chapter="not-a-chapter")
    assert out["status"] == "success" and out["chapter_filter"] is None
    travel = next(p for p in out["passages"] if p["title"] == "Travel Cards")
    assert "jane@stanford.edu" not in travel["text"] and "[EMAIL]" in travel["text"]


def test_crawler_extracts_sections_and_stays_in_chapter():
    html = """<html><body><nav><a href="/x">Menu</a></nav><main>
      <h1>Travel Cards</h1><h2>Use</h2><p>Business only.</p><ul><li>No personal charges.</li></ul>
      <a href="/chapters/purchasing-and-payments/travel-cards/limits#top">Limits</a>
      <a href="/chapters/human-resources">HR</a></main></body></html>"""
    url = "https://adminguide.stanford.edu/chapters/purchasing-and-payments/travel-cards"
    page = extract(html, url, "purchasing-and-payments")
    assert page["title"] == "Travel Cards"
    assert page["sections"] == [("Use", "Business only."), ("Use", "No personal charges.")]
    links = list(links_in(html, url, "https://adminguide.stanford.edu/chapters/purchasing-and-payments"))
    assert links == ["https://adminguide.stanford.edu/chapters/purchasing-and-payments/travel-cards/limits"]


def test_crawler_drops_print_button_and_dedupes_alias_urls(monkeypatch):
    import scripts.crawl_adminguide as crawler

    html = "<main><h1>Travel Cards</h1><p>Print</p><p>Business only.</p></main>"
    alias = "https://adminguide.stanford.edu/chapters/p/travel-and-business-expenses/travel-cards"
    index = ('<main><h1>Chapter</h1><p>Intro</p>'
             '<a href="/chapters/p/business-and-travel-expenses/travel-cards">a</a>'
             f'<a href="{alias}">b</a></main>')

    class Resp:
        status_code = 200
        headers = {"Content-Type": "text/html"}

        def __init__(self, text):
            self.text = text

    def fake_get(self, url, timeout):
        return Resp(index if url.endswith("/chapters/p") else html)

    monkeypatch.setattr(crawler.requests.Session, "get", fake_get)
    pages = crawler.crawl(["p"], max_pages=10, delay=0)
    titles = [p["title"] for p in pages]
    assert titles == ["Chapter", "Travel Cards"]
    assert pages[1]["sections"] == [("Travel Cards", "Business only.")]


def test_eval_matches_alias_and_subpages():
    from scripts.eval_retrieval import is_match
    base = "https://adminguide.stanford.edu/chapters/p/business-and-travel-expenses/travel-cards"
    assert is_match(base + "/limits", base)
    assert is_match("https://adminguide.stanford.edu/chapters/p/travel-and-business-expenses/travel-cards", base)
    assert not is_match("https://adminguide.stanford.edu/chapters/p/business-and-travel-expenses/expense-advances", base)


def test_lower_bm25_weight_lets_dense_win(retriever):
    # "stolen laptop" vs a keyword-heavy distractor: with equal votes BM25 can drag
    # a wrong chunk up; with a small BM25 weight the dense ranking decides.
    full = retriever.search("report stolen laptop restricted data", k=3, bm25_weight=1.0)
    light = retriever.search("report stolen laptop restricted data", k=3, bm25_weight=0.1)
    dense = retriever.search("report stolen laptop restricted data", k=3, mode="dense")
    assert [h["id"] for h in light] == [h["id"] for h in dense]
    assert full  # equal-weight run still returns results
