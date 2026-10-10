"""Heading-aware chunking.

A page is a list of (section heading, paragraph text) blocks. Chunks never
cross a section boundary, so each chunk can cite one specific section.
Each chunk's text starts with "<page title> > <section>" so the embedding
knows which policy it came from even when the paragraph alone is vague.
"""
import hashlib


def _words(text):
    return text.split()


def chunk_page(page, max_words=220, overlap=40):
    chunks = []
    by_section = {}
    order = []
    for heading, text in page["sections"]:
        heading = heading or page["title"]
        if heading not in by_section:
            by_section[heading] = []
            order.append(heading)
        by_section[heading].append(text)

    for heading in order:
        words = _words(" ".join(by_section[heading]))
        if not words:
            continue
        start = 0
        while start < len(words):
            piece = " ".join(words[start:start + max_words])
            prefix = page["title"] if heading == page["title"] else f"{page['title']} > {heading}"
            uid = hashlib.sha1(f"{page['url']}|{heading}|{start}".encode()).hexdigest()[:12]
            chunks.append({
                "id": uid,
                "url": page["url"],
                "chapter": page["chapter"],
                "title": page["title"],
                "section": heading,
                "text": f"{prefix}\n{piece}",
            })
            if start + max_words >= len(words):
                break
            start += max_words - overlap
    return chunks
