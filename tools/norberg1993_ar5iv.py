"""Mine open-access papers for the numerical values of Norberg (1993) square-cylinder
benchmarks (Re = 200).  Fast path: ar5iv HTML (single GET per paper) instead of PDFs.

Usage: python3 tools/norberg1993_ar5iv.py --budget 90 --max 12 "query1" "query2"
State: runs/norberg1993_research/{html/, done.json, contexts.log}
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("runs/norberg1993_research")
HTML = OUT / "html"
HTML.mkdir(parents=True, exist_ok=True)
DONE = OUT / "done.json"
CTX = OUT / "contexts.log"


def fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "TensorLBM-ref-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search(query: str, n: int = 20):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "max_results": n, "sortBy": "relevance"}
    )
    xml = fetch(url).decode("utf-8", "replace")
    ids = re.findall(r"<id>https?://arxiv.org/abs/([^<]+)</id>", xml)
    return ids


def text_of(aid: str) -> str:
    p = HTML / (aid.replace("/", "_") + ".txt")
    if p.exists():
        return p.read_text(errors="replace")
    raw = fetch(f"https://ar5iv.labs.arxiv.org/html/{aid}", timeout=45).decode("utf-8", "replace")
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = re.sub(r"&nbsp;?", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    p.write_text(txt)
    return txt


def main() -> None:
    budget = 90.0
    args = sys.argv[1:]
    if "--budget" in args:
        i = args.index("--budget")
        budget = float(args[i + 1])
        del args[i : i + 2]
    maxp = 12
    if "--max" in args:
        i = args.index("--max")
        maxp = int(args[i + 1])
        del args[i : i + 2]
    queries = args or ['abs:"square cylinder" AND abs:"Strouhal"']

    done = json.loads(DONE.read_text()) if DONE.exists() else {}
    t0 = time.time()
    cand: list[str] = []
    for q in queries:
        try:
            for aid in search(q):
                if aid not in done and aid not in cand:
                    cand.append(aid)
        except Exception as e:
            print(f"[search-fail] {q}: {e}", flush=True)
    print(f"[candidates] {len(cand)} new", flush=True)

    n_papers = 0
    for aid in cand:
        if time.time() - t0 > budget or n_papers >= maxp:
            break
        try:
            txt = text_of(aid)
        except Exception as e:
            print(f"[fetch-fail] {aid}: {e}", flush=True)
            done[aid] = "fetch_fail"
            continue
        n_papers += 1
        hits = 0
        with CTX.open("a") as fh:
            for m in re.finditer(r"Norberg", txt):
                ctx = txt[max(0, m.start() - 350) : m.start() + 450]
                if re.search(r"square|rectangular|aspect ratio|Re\s*[=≈]?\s*200", ctx, re.I):
                    hits += 1
                    fh.write(f"\n=== {aid}\n{ctx}\n")
        done[aid] = hits
        print(f"[ok] {aid} hits={hits} (t={time.time()-t0:.0f}s)", flush=True)
    DONE.write_text(json.dumps(done, indent=1))
    print(f"[summary] papers_read={n_papers} total_cached={len(done)}")


if __name__ == "__main__":
    main()