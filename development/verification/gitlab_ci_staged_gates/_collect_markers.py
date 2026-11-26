"""pytest plugin: dump every collected item with its markers to JSON (one pass)."""

import json


def pytest_collection_finish(session):
    items = []
    for it in session.items:
        items.append(
            {
                "nodeid": it.nodeid,
                "markers": sorted({m.name for m in it.iter_markers()}),
            }
        )
    with open("/tmp/corpus_markers.json", "w") as f:
        json.dump(items, f)
