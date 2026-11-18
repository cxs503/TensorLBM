"""Build a GIF from the sampled contours of a completed or interrupted entry run."""

import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.run_directory.glob("contours/phase_*.png"))
    if len(files) < 2:
        parser.error("at least two contours are required")
    steps = [int(p.stem.split("_")[-1]) for p in files]
    # Play the sampled lattice history in eight seconds, with a pause at the end.
    duration = [
        max(20, round(8000 * (b - a) / (steps[-1] - steps[0]))) for a, b in zip(steps, steps[1:])
    ] + [800]
    frames = []
    for path in files:
        with Image.open(path) as frame:
            frame.thumbnail((760, 460), Image.Resampling.LANCZOS)
            frames.append(frame.convert("RGB").quantize(colors=128))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output, save_all=True, append_images=frames[1:], duration=duration, loop=0, disposal=2
    )


if __name__ == "__main__":
    main()
