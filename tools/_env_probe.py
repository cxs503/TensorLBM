import os

for p in ("/root/tp-agent", "/root/tp-agent/work", "/root"):
    try:
        print(p, "->", sorted(os.listdir(p))[:25])
    except Exception as e:  # noqa: BLE001
        print(p, "ERR", e)

for k, v in sorted(os.environ.items()):
    if any(s in k.upper() for s in ("DATABASE", "POSTGRES", "PG", "DB_")):
        print(k, "=", v if "PASS" not in k else "***")