"""Probe network reachability for a few candidate reference sources (read-only)."""
import urllib.request

URLS = [
    "https://html.duckduckgo.com/html/?q=strouhal+number+cylinder+Re%3D100+0.164+Williamson+Brown+table",
    "https://www.mdpi.com/2311-5521/7/1/1",
    "https://arxiv.org/abs/2010.00019",
    "https://ntrs.nasa.gov/api/citations/19930092271/downloads/19930092271.pdf",
]

for u in URLS:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (research-bot)"})
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read(400)
            print("OK  ", r.status, len(body), u)
    except Exception as e:  # noqa: BLE001
        print("FAIL", str(e)[:90], u)