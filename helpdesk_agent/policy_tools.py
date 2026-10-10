"""Policy search tool backed by the hybrid Admin Guide index."""
from .guardrails import redact
from .telemetry import instrumented

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from .rag.retriever import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever


@instrumented
def search_policy(question: str, chapter: str = "") -> dict:
    """Searches the Stanford Administrative Guide for policy text relevant to a question.

    Args:
        question: The user's question in plain words.
        chapter: Optional filter. One of computing, purchasing-and-payments, human-resources,
            financial-administration, health-and-safety, services. Leave empty to search all.

    Returns:
        dict with status and up to 4 passages, each with title, section, url and text.
    """
    retriever = get_retriever()
    chapter = chapter.strip().lower()
    if chapter and chapter not in retriever.chapters():
        chapter = ""  # unknown chapter: fall back to searching everything
    hits = retriever.search(question.strip()[:500], k=4, chapter=chapter or None)
    passages = []
    for h in hits:
        text, _ = redact(h["text"])
        passages.append({
            "title": h["title"],
            "section": h["section"],
            "url": h["url"],
            "text": text[:1200],
        })
    if not passages:
        return {"status": "no_results", "passages": []}
    return {"status": "success", "chapter_filter": chapter or None, "passages": passages}
