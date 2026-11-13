#!/usr/bin/env python3
"""W26 anchor extraction: Ghia, Ghia & Shin (1982) Tables I & II, Re = 100/400/1000.

ZERO-hand-copying pipeline
--------------------------
Anchor values are never typed by hand.  Two independent machine witnesses are
combined for every tabulated number:

  W1 TEXT LAYER (PyMuPDF word boxes).  The circulating PDF of the 1982 paper is a
     scan carrying an OCR text layer.  The OCR is reliable for the *digits* but
     routinely mangles the printed minus sign ('a', 'AI', 'Al', 'xJ', 'X)', 'Jx',
     '4', 'I'), so the text layer alone cannot fix the sign.

  W2 PIXELS (page raster, 8 px/pt).  The value columns are right-aligned, so the
     ink width of a cell differs by exactly one minus advance (~4.5 pt) between
     '-0.xxxxx' (~29.5 pt) and '0.xxxxx' (~25 pt).  Per column the widths split
     into two well separated clusters; the larger one is the signed one.  The
     cluster geometry is reported so the split can be audited.

  Final value = sign(W2) * magnitude(W1).  Cells where either witness is unusable
  stay null and are flagged -- nothing is ever patched from memory.

Provenance per value: PDF sha256, page index, journal page, row y band, cell bbox,
verbatim OCR token, OCR digits, ink x-extent and the sign-cluster statistics.

Usage:
    GHIA_PDF=/tmp/ghia1982.pdf python3 scripts/w26_ghia/extract_ghia_tables.py \
        --out papers/artifacts/w26/d3q19-bgk-analytic-fourcase/cells_raw.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pypdf
import fitz  # PyMuPDF

# Table geometry (PDF points), calibrated on the word boxes of the two pages.
TABLES = {
    "I": {
        "page_index": 11,
        "journal_page": 398,
        "quantity": "u-velocity along the vertical centreline x/L=0.5 (normalised by lid speed)",
        "coord_name": "y",
        "coord_col": (85.0, 116.0),
        "value_cols": {100.0: (118.0, 160.0), 400.0: (163.0, 201.0), 1000.0: (205.0, 245.0)},
    },
    "II": {
        "page_index": 12,
        "journal_page": 399,
        "quantity": "v-velocity along the horizontal centreline y/L=0.5 (normalised by lid speed)",
        "coord_name": "x",
        "coord_col": (85.0, 116.0),
        "value_cols": {100.0: (118.0, 160.0), 400.0: (163.0, 201.0), 1000.0: (205.0, 245.0)},
    },
}

ROW_TOL = 2.5       # pt, clustering tolerance for row centres
MERGE_GAP = 4.2     # pt, merge word fragments of one cell
ZOOM = 8.0          # page raster zoom for the ink measurements (8 px/pt)
INK_THR = 200       # grey level below which a pixel counts as ink


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_meta(path: str) -> Dict[str, object]:
    with open(path, "rb") as fh:
        n_pages = len(pypdf.PdfReader(fh).pages)
    return {"path": path, "sha256": sha256_file(path), "n_pages": n_pages,
            "bytes": os.path.getsize(path)}


# --------------------------------------------------------------------------- #
def row_centres(words) -> List[float]:
    ys = sorted((w[1] + w[3]) / 2 for w in words)
    bands: List[List[float]] = []
    for y in ys:
        if bands and y - bands[-1][-1] < ROW_TOL:
            bands[-1].append(y)
        else:
            bands.append([y])
    return [float(np.mean(b)) for b in bands]


def cells_of_row(words, y0: float, y1: float, x0: float, x1: float) -> List[Tuple]:
    """The one cell inside a (row band, column band) window.

    A column band holds exactly one printed value, but the OCR layer sometimes cuts
    a value into fragments ('-0.2058' + '1').  The fragments are therefore joined in
    x-order unconditionally; the window itself guarantees that nothing from a
    neighbouring column can leak in (the bands do not overlap).
    """
    sel = [w for w in words if y0 <= (w[1] + w[3]) / 2 <= y1 and x0 <= (w[0] + w[2]) / 2 <= x1]
    sel.sort(key=lambda w: w[0])
    if not sel:
        return []
    return [(min(w[0] for w in sel), min(w[1] for w in sel),
             max(w[2] for w in sel), max(w[3] for w in sel),
             "".join(w[4] for w in sel), len(sel))]


# --------------------------------------------------------------------------- #
class PageRaster:
    """Full-page raster: PDF points map linearly to pixels (rect origin = 0)."""

    def __init__(self, page):
        self.rect = page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), colorspace=fitz.csGRAY)
        self.img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

    def ink_bbox(self, x0, y0, x1, y1):
        """Ink extent (PDF pt) inside a rectangle, or None if blank."""
        i0 = max(0, int(round((x0 - self.rect.x0) * ZOOM)))
        i1 = min(self.img.shape[1], int(round((x1 - self.rect.x0) * ZOOM)))
        j0 = max(0, int(round((y0 - self.rect.y0) * ZOOM)))
        j1 = min(self.img.shape[0], int(round((y1 - self.rect.y0) * ZOOM)))
        ink = self.img[j0:j1, i0:i1] < INK_THR
        if not ink.any():
            return None
        rows = np.where(ink.any(axis=1))[0]
        cols = np.where(ink.any(axis=0))[0]
        return {
            "x0": x0 + cols.min() / ZOOM, "x1": x0 + (cols.max() + 1) / ZOOM,
            "y0": y0 + rows.min() / ZOOM, "y1": y0 + (rows.max() + 1) / ZOOM,
        }


def two_cluster_split(widths: List[float]) -> Dict[str, object]:
    """Split ink widths into {unsigned, signed} clusters (1-D k-means, k=2)."""
    w = np.asarray(widths, dtype=float)
    lo, hi = np.percentile(w, 5), np.percentile(w, 95)
    c = np.array([lo, hi], dtype=float)
    for _ in range(200):
        lab = np.argmin(np.abs(w[:, None] - c[None, :]), axis=1)
        new = np.array([w[lab == k].mean() if (lab == k).any() else c[k] for k in (0, 1)])
        if np.allclose(new, c):
            break
        c = new
    order = np.argsort(c)
    c = c[order]
    labels = np.argmin(np.abs(w[:, None] - c[None, :]), axis=1)
    unsigned, signed = int(np.sum(labels == 0)), int(np.sum(labels == 1))
    return {
        "centres_pt": [round(float(c[0]), 2), round(float(c[1]), 2)],
        "gap_pt": round(float(c[1] - c[0]), 2),
        "threshold_pt": round(float(c.mean()), 2),
        "n_unsigned": unsigned,
        "n_signed": signed,
        "width_min_pt": round(float(w.min()), 2),
        "width_max_pt": round(float(w.max()), 2),
        "separation_clean": bool(c[1] - c[0] > 2.5 and
                                 np.min(w[labels == 1]) - np.max(w[labels == 0]) > 1.0),
    }


# --------------------------------------------------------------------------- #
def decode_digits(tok: str) -> Dict[str, object]:
    """Digit string of an OCR token -> printed magnitude (sign comes from pixels).

    All table entries are printed as 'X.XXXXX' (one integer digit, five decimals),
    so if the OCR swallowed the decimal point ('AI13641') the magnitude is
    int(digits) * 1e-5.  A magnitude above 1 is impossible for these normalised
    profiles (|u|,|v| <= lid speed), and there the first OCR character is in fact
    the misplaced minus -> magnitude is re-read as 0.<rest>; that repair is flagged.
    """
    digits = re.sub(r"[^0-9.]", "", tok).replace("o", "0").replace("O", "0")
    if digits.startswith("."):
        digits = "0" + digits
    if digits.count(".") > 1:
        head, tail = digits.split(".", 1)
        digits = head + "." + tail.replace(".", "")
    if not digits:
        return {"digits": "", "magnitude": None, "recovered_decimal": False, "magnitude_repaired": False}
    if "." not in digits:
        mag = int(digits) * 1e-5
        return {"digits": digits, "magnitude": mag, "recovered_decimal": True, "magnitude_repaired": False}
    try:
        mag = float(digits)
    except ValueError:
        return {"digits": digits, "magnitude": None, "recovered_decimal": False, "magnitude_repaired": False}
    if mag > 1.0 + 1e-12:  # first character is the minus, not a digit
        rest = digits.replace(".", "")[1:]
        mag = float("0." + rest)
        return {"digits": digits, "magnitude": mag, "recovered_decimal": True, "magnitude_repaired": True}
    return {"digits": digits, "magnitude": mag, "recovered_decimal": False, "magnitude_repaired": False}


def decode_coord(tok: str):
    s = re.sub(r"[^0-9.]", "", tok).replace("o", "0").replace("O", "0")
    if s.startswith("."):
        s = "0" + s
    if s.count(".") > 1:
        head, tail = s.split(".", 1)
        s = head + "." + tail.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
def extract(pdf_path: str) -> Dict[str, object]:
    doc = fitz.open(pdf_path)
    out: Dict[str, object] = {"pdf": pdf_meta(pdf_path), "tables": {}}
    for name, spec in TABLES.items():
        page = doc[spec["page_index"]]
        raster = PageRaster(page)
        words = page.get_text("words")
        body = [w for w in words if w[1] > 370.0 and w[3] < 600.0 and w[0] < 245.0]
        centres = row_centres(body)
        rows = []
        for bi, yc in enumerate(centres):
            b0, b1 = yc - ROW_TOL, yc + ROW_TOL
            coord_cells = cells_of_row(words, b0, b1, *spec["coord_col"])
            pt_cells = cells_of_row(words, b0, b1, 55.0, 80.0)
            pt_raw = pt_cells[0][4] if pt_cells else None
            rows.append({
                "row_index": bi,
                "y_band_pt": [round(b0, 2), round(b1, 2)],
                "pt_no_raw": pt_raw,
                "pt_no": int(re.sub(r"[^0-9]", "", pt_raw)) if pt_raw and re.sub(r"[^0-9]", "", pt_raw) else None,
                "coord_raw": [c[4] for c in coord_cells],
                "coord": decode_coord(coord_cells[0][4]) if coord_cells else None,
                "coord_bbox_pt": [[round(c[0], 2), round(c[1], 2), round(c[2], 2)] for c in coord_cells],
                "values": {},
            })
        # ---- pass 2: per-cell text + pixel ink extent -------------------------
        for re_key, (cx0, cx1) in spec["value_cols"].items():
            widths = []
            cells_by_row = []
            for r in rows:
                b0, b1 = r["y_band_pt"]
                cells = cells_of_row(words, b0, b1, cx0, cx1)
                if len(cells) != 1:
                    cells_by_row.append(None)
                    widths.append(np.nan)
                    continue
                c = cells[0]
                bbox = raster.ink_bbox(cx0 - 2.0, b0, cx1 + 2.0, b1)
                cells_by_row.append((c, bbox))
                widths.append(np.nan if bbox is None else (bbox["x1"] - bbox["x0"]))
            valid = [w for w in widths if w == w]
            split = two_cluster_split(valid) if len(valid) >= 4 else {"separation_clean": False}
            thr = split.get("threshold_pt")
            for r, cell_pair, wd in zip(rows, cells_by_row, widths):
                key = str(re_key)
                if cell_pair is None:
                    r["values"][key] = {"status": "cell_not_unique", "ink_width_pt": None}
                    continue
                c, bbox = cell_pair
                dec = decode_digits(c[4])
                sign = None
                if thr is not None and wd == wd:
                    sign = -1.0 if wd >= thr else 1.0
                r["values"][key] = {
                    "ocr_token": c[4],
                    "digits": dec["digits"],
                    "magnitude": dec["magnitude"],
                    "decimal_recovered": dec["recovered_decimal"],
                    "magnitude_repaired": dec["magnitude_repaired"],
                    "sign_from_ink": sign,
                    "ink_width_pt": None if wd != wd else round(wd, 2),
                    "ink_bbox_pt": None if bbox is None else [round(bbox[k], 2) for k in ("x0", "y0", "x1", "y1")],
                    "value": None if (dec["magnitude"] is None or sign is None) else sign * dec["magnitude"],
                }
            out.setdefault("sign_split", {})[f"{name}_{int(re_key)}"] = split
        # ---- structural check: are the value columns right-aligned? ----------
        right_edges = [v["ink_bbox_pt"][2] for r in rows for v in r["values"].values()
                       if v.get("ink_bbox_pt")]
        header_words = [{"text": w[4], "bbox_pt": [round(w[0], 2), round(w[1], 2),
                                                   round(w[2], 2), round(w[3], 2)]}
                        for w in words if 330.0 <= w[1] <= 372.0 and w[0] < 245.0]
        header_words.sort(key=lambda d: (round(d["bbox_pt"][1], 1), d["bbox_pt"][0]))
        out["tables"][name] = {
            "page_index": spec["page_index"],
            "journal_page": spec["journal_page"],
            "quantity": spec["quantity"],
            "coord_name": spec["coord_name"],
            "n_rows": len(rows),
            "rows": rows,
            "header_words": header_words,
            "right_edge_spread_pt_all_columns": round(float(np.ptp(right_edges)), 3) if right_edges else None,
        }
    return out


def summary(data) -> None:
    for name, tab in data["tables"].items():
        print(f"== Table {name} (p.{tab['journal_page']}, {tab['n_rows']} rows)")
        for r in tab["rows"]:
            vals = []
            for k in ("100.0", "400.0", "1000.0"):
                v = r["values"].get(k, {})
                vals.append(f"{v.get('value')!s:>9}<-{v.get('ocr_token')!s:>8} w={v.get('ink_width_pt')!s:>6}")
            print(f"  {str(r['coord_raw'][0] if r['coord_raw'] else '?'):>9} | " + " | ".join(vals))
        for k, sp in data.get("sign_split", {}).items():
            if k.startswith(name + "_"):
                print(f"  split {k}: {sp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=os.environ.get("GHIA_PDF", "/tmp/ghia1982.pdf"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = extract(args.pdf)
    data["extractor"] = {
        "script": "scripts/w26_ghia/extract_ghia_tables.py",
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "pymupdf": fitz.__doc__.strip().splitlines()[0],
        "pypdf": pypdf.__version__,
        "zoom_px_per_pt": ZOOM,
        "ink_threshold_gray": INK_THR,
        "row_tol_pt": ROW_TOL,
        "merge_gap_pt": MERGE_GAP,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    summary(data)
    print("wrote", args.out)


if __name__ == "__main__":
    main()