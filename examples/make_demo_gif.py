"""Generate a lid-driven-cavity demo GIF.

Runs a 2D lid-driven cavity case for 2000 steps, saves a frame every 50 steps,
and assembles the generated ``flow_step_*.png`` files into a GIF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from tensorlbm import LidDrivenCavityConfig, run_lid_driven_cavity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=96, help="Grid size (nx x nx)")
    parser.add_argument("--re", type=float, default=400.0, help="Reynolds number")
    parser.add_argument("--u-lid", dest="u_lid", type=float, default=0.1, help="Lid speed")
    parser.add_argument("--n-steps", type=int, default=2000, help="Total simulation steps")
    parser.add_argument("--frame-every", type=int, default=50, help="Snapshot cadence (steps)")
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second")
    parser.add_argument("--output-root", default="outputs", help="Root output directory")
    parser.add_argument("--run-name", default="demo_gif", help="Run folder name")
    parser.add_argument("--gif-name", default="lid_driven_cavity_demo.gif", help="Output GIF filename")
    parser.add_argument("--device", choices=["cpu", "sdaa", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run folder")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = LidDrivenCavityConfig(
        nx=args.nx,
        re=args.re,
        u_lid=args.u_lid,
        n_steps=args.n_steps,
        output_interval=args.frame_every,
        output_root=Path(args.output_root),
        run_name=args.run_name,
        device=args.device,
        overwrite=args.overwrite,
    )
    run_dir = run_lid_driven_cavity(config)

    frame_paths = sorted(run_dir.glob("flow_step_*.png"))
    if not frame_paths:
        msg = f"No flow_step_*.png frames found in {run_dir}"
        raise FileNotFoundError(msg)

    frames = [Image.open(path).convert("P") for path in frame_paths]
    gif_path = run_dir / args.gif_name
    duration_ms = max(1, int(1000 / max(args.fps, 1)))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )

    print(f"Run directory: {run_dir}")
    print(f"Frames: {len(frame_paths)}")
    print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()
