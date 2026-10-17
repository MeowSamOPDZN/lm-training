"""
Parse the Simple English Wikipedia XML dump into clean plain text.

Reads data/raw/simplewiki.xml.bz2 (streaming, never fully in RAM),
extracts article text from <page> elements (main namespace only,
redirects skipped), strips wiki markup with regexes, and writes
data/raw/simplewiki.txt with <|endoftext|> separators between articles.

Usage:
    ~/workspace/lm-training/venv/bin/python scripts/parse_wiki.py
"""

import bz2
import os
import re
import sys
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(BASE, "data", "raw", "simplewiki.xml.bz2")
OUT_PATH = os.path.join(BASE, "data", "raw", "simplewiki.txt")
EOS_SEP = "\n<|endoftext|>\n"

# --- markup stripping -------------------------------------------------------
RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_REF = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
RE_REF_SELF = re.compile(r"<ref[^>]*/>", re.IGNORECASE)
RE_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_LINK = re.compile(r"\[\[([^|\]]*\|)?([^\]]+)\]\]")
RE_EXTLINK = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")
RE_HEADING = re.compile(r"^=+\s*(.*?)\s*=+$", re.M)
RE_BOLDITALIC = re.compile(r"'''?([^']+)'''?")
RE_LIST = re.compile(r"^[*#;:]+", re.M)


def clean_wikitext(text: str) -> str:
    text = RE_COMMENT.sub("", text)
    text = RE_REF.sub("", text)
    text = RE_REF_SELF.sub("", text)
    # templates can nest; loop a few times
    for _ in range(4):
        new = RE_TEMPLATE.sub("", text)
        if new == text:
            break
        text = new
    text = RE_LINK.sub(r"\2", text)
    text = RE_EXTLINK.sub(r"\1", text)
    text = RE_HEADING.sub(r"\1", text)
    text = RE_BOLDITALIC.sub(r"\1", text)
    text = RE_LIST.sub("", text)
    text = RE_TAG.sub("", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def iter_pages(path):
    """Yield (title, text) for main-namespace non-redirect pages."""
    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        context = ET.iterparse(f, events=("end",))
        title, ns, text, is_redirect = "", "", "", False
        for event, elem in context:
            name = localname(elem.tag)
            if name == "title":
                title = elem.text or ""
            elif name == "ns":
                ns = elem.text or ""
            elif name == "redirect":
                is_redirect = True
            elif name == "text":
                text = elem.text or ""
            elif name == "page":
                if ns == "0" and not is_redirect and text.strip():
                    yield title, text
                title, ns, text, is_redirect = "", "", "", False
                elem.clear()


def main() -> None:
    if not os.path.exists(IN_PATH):
        print(f"Missing {IN_PATH} - run the wiki download first.")
        sys.exit(1)
    n_pages, n_chars, first = 0, 0, True
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for title, raw in iter_pages(IN_PATH):
            cleaned = clean_wikitext(raw)
            if len(cleaned) < 200:  # stubs / disambiguation-ish
                continue
            if not first:
                out.write(EOS_SEP)
            out.write(f"{title}\n\n{cleaned}")
            first = False
            n_pages += 1
            n_chars += len(cleaned)
            if n_pages % 2000 == 0:
                print(f"  ... {n_pages} pages, {n_chars/1e6:.0f}M chars", flush=True)
    print(f"Done: {n_pages} articles, {n_chars/1e6:.1f}M chars -> {OUT_PATH}")


if __name__ == "__main__":
    main()
