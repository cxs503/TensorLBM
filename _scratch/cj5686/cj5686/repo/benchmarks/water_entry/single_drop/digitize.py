"""Reproduce the 18 observed marker centres in the publisher's Figure 5 JPEG."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SOURCE_SHA256 = "8c251d994ff6442d96aa1b7d9093cc2acca8e72ec6116120df7a574fa99bad06"
# Extract the 3337 x 1832 JPEG from page 9 of the publisher's PDF (pdfimages -j).
# Extreme gridline centres in that image; pixel y grows down.
X0, X20 = 222.4, 3268.29
Y0, Y2 = 1626.54, 67.02


def digitize(source, dest):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("different image revision or resolution")
    r, g, b = np.asarray(Image.open(source).convert("RGB"), dtype=np.int16).transpose(2, 0, 1)
    mask = (r > g + 60) & (r > b + 60) & (r > 120)
    # Legend box in the published plot, including its example red marker.
    mask[1302:1615, 725:3255] = False
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    points = []
    for label, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        y, x = np.where(labels[region] == label)
        if len(x) >= 100:
            points.append((x.mean() + region[1].start, y.mean() + region[0].start))
    if len(points) != 18:
        raise ValueError("expected 18 experimental markers; inspect the source")
    with dest.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["x_px", "y_px", "time_ms", "depth_over_D"])
        for x, y in sorted(points):
            writer.writerow(
                [
                    f"{x:.3f}",
                    f"{y:.3f}",
                    f"{20 * (x - X0) / (X20 - X0):.5f}",
                    f"{2 * (Y0 - y) / (Y0 - Y2):.6f}",
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publisher_jpeg", type=Path)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    digitize(args.publisher_jpeg, args.csv)
