import json, urllib.request

t = open("/tmp/tok.txt").read().strip()
req = urllib.request.Request("http://127.0.0.1:8010/api/v1/openapi.json")
req.add_header("Authorization", "Bearer " + t)
doc = json.loads(urllib.request.urlopen(req, timeout=60).read())
paths = [p for p in doc["paths"] if any(k in p for k in ("knowledge", "oss", "negative", "capabilit"))]
for p in sorted(paths):
    print(p, list(doc["paths"][p].keys()))