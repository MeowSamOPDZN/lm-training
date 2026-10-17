"""
Download public-domain books from Project Gutenberg.

Curated list of famous book IDs. Each book is fetched with curl
(resume + retries, since the network here is flaky), the Gutenberg
header/footer boilerplate is stripped, and everything is concatenated
into data/raw/books.txt with <|endoftext|> separators between books.

Failed / tiny downloads are skipped and reported, never fatal.

Usage:
    ~/workspace/lm-training/venv/bin/python scripts/download_books.py
"""

import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "books.txt")
TMP_DIR = os.path.join(RAW_DIR, "_books_tmp")

# Famous public-domain books (Project Gutenberg ebook IDs).
# Uncertain IDs are harmless: failures are skipped and logged.
BOOK_IDS = [
    11, 12, 16, 23, 33, 35, 36, 43, 55, 74, 76, 84, 98, 100, 103, 120,
    135, 140, 141, 145, 158, 161, 163, 164, 174, 196, 205, 208, 209, 219,
    228, 244, 254, 260, 281, 285, 345, 394, 408, 415, 423, 430, 431, 436,
    437, 512, 513, 514, 519, 520, 521, 522, 523, 580, 613, 614, 615, 643,
    644, 645, 730, 731, 766, 768, 769, 863, 880, 881, 882, 883, 884, 885,
    886, 887, 888, 996, 1023, 1080, 1184, 1232, 1257, 1259, 1260, 1261,
    1262, 1298, 1342, 1350, 1399, 1400, 1479, 1496, 1497, 1584, 1597, 1600,
    1611, 1635, 1661, 1787, 1952, 2000, 2009, 2010, 2050, 2097, 2148, 2199,
    2253, 2445, 2542, 2554, 2600, 26184, 2641, 28054, 2814, 2852, 2895,
    3296, 4300, 4419, 5135, 5200, 5212, 64317, 6593, 67979, 69087,
]

MIN_BYTES = 50_000  # skip anything smaller (probably an error page)
EOS_SEP = "\n<|endoftext|>\n"

HEADER_RE = re.compile(
    r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.DOTALL
)
FOOTER_RE = re.compile(
    r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.DOTALL
)


def fetch(book_id: int) -> str | None:
    """Return local path of downloaded book, or None on failure."""
    os.makedirs(TMP_DIR, exist_ok=True)
    dest = os.path.join(TMP_DIR, f"{book_id}.txt")
    if os.path.exists(dest) and os.path.getsize(dest) > MIN_BYTES:
        return dest
    for name in (f"{book_id}-0.txt", f"{book_id}.txt"):
        url = f"https://www.gutenberg.org/files/{book_id}/{name}"
        r = subprocess.run(
            ["curl", "-sS", "-C", "-", "--retry", "5", "--retry-delay", "3",
             "--max-time", "120", "-o", dest, "-w", "%{http_code}", url],
            capture_output=True, text=True,
        )
        code = (r.stdout or "").strip()[-3:]
        if code == "200" and os.path.exists(dest) and os.path.getsize(dest) > MIN_BYTES:
            return dest
    if os.path.exists(dest):
        os.remove(dest)
    return None


def clean(text: str) -> str:
    text = HEADER_RE.sub("", text)
    text = FOOTER_RE.sub("", text)
    # collapse 3+ newlines, strip trailing spaces
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-books", type=int, default=None,
                    help="only download the first N books (quick test)")
    args = ap.parse_args()

    book_ids = BOOK_IDS[:args.max_books] if args.max_books else BOOK_IDS
    os.makedirs(RAW_DIR, exist_ok=True)
    ok, failed = 0, []
    total_bytes = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        first = True
        for i, bid in enumerate(book_ids, 1):
            print(f"[{i}/{len(book_ids)}] fetching id={bid}...", flush=True)
            path = fetch(bid)
            if path is None:
                failed.append(bid)
                print(f"[{i}/{len(book_ids)}] id={bid}: SKIP", flush=True)
                continue
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = clean(f.read())
            if len(text) < 10_000:
                failed.append(bid)
                print(f"[{i}/{len(book_ids)}] id={bid}: too small, SKIP", flush=True)
                continue
            if not first:
                out.write(EOS_SEP)
            out.write(text)
            first = False
            ok += 1
            total_bytes += len(text.encode("utf-8"))
            print(f"[{i}/{len(book_ids)}] id={bid}: OK "
                  f"({len(text)//1000}k chars, total {total_bytes//1_000_000}MB)",
                  flush=True)
    print(f"\nDone: {ok} books -> {OUT_PATH} ({total_bytes/1e6:.1f} MB)")
    if failed:
        print(f"Skipped {len(failed)} ids: {failed}")


if __name__ == "__main__":
    sys.exit(main())
