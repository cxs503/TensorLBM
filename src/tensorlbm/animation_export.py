"""Flow-field animation export utilities.

Creates GIF or MP4 animations from a sequence of simulation snapshot PNG
images stored in a job output directory.  This matches the animation export
capability of PowerFlow and XFlow for presenting transient flow visualisations.

Usage
-----
::

    from tensorlbm.animation_export import create_animation
    gif_path = create_animation(job_dir, field="speed", fps=10, fmt="gif")

Dependencies
------------
* ``Pillow`` (PIL) – required for GIF output (always available).
* ``matplotlib`` – required for colourbar / overlay annotations.
* ``ffmpeg`` – optional; required only for MP4 output.  Detected at runtime;
  if absent the function falls back to GIF.

"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal

logger = logging.getLogger("tensorlbm.animation_export")

__all__ = [
    "create_animation",
    "frames_from_png_dir",
    "gif_from_frames",
    "mp4_from_frames",
]


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

def frames_from_png_dir(
    job_dir: Path,
    pattern: str = r"step_(\d+)\.png",
    *,
    max_frames: int = 500,
) -> list[Path]:
    """Collect PNG snapshot files from *job_dir*, sorted by step number.

    Args:
        job_dir:    Job output directory.
        pattern:    Regex pattern matching step PNG filenames; must contain
                    exactly one numeric capture group (the step number).
        max_frames: Maximum number of frames to include (uniformly subsample
                    if more are found).

    Returns:
        Sorted list of Path objects.

    Raises:
        FileNotFoundError: If no matching PNG files are found.
    """
    rx = re.compile(pattern)
    pairs: list[tuple[int, Path]] = []
    for p in job_dir.rglob("*.png"):
        m = rx.match(p.name)
        if m:
            pairs.append((int(m.group(1)), p))

    if not pairs:
        raise FileNotFoundError(f"No PNG frames matching '{pattern}' in {job_dir}")

    pairs.sort(key=lambda t: t[0])
    frames = [p for _, p in pairs]

    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]

    return frames


# ---------------------------------------------------------------------------
# GIF builder
# ---------------------------------------------------------------------------

def gif_from_frames(
    frames: list[Path],
    output_path: Path,
    fps: int = 10,
    loop: int = 0,
) -> Path:
    """Build a GIF animation from a list of PNG frame paths.

    Args:
        frames:      Ordered list of PNG file paths.
        output_path: Destination ``.gif`` file path.
        fps:         Frames per second (1–60).
        loop:        Number of GIF loops (0 = infinite).

    Returns:
        Path to the created GIF file.

    Raises:
        ImportError: If Pillow is not installed.
        ValueError:  If *frames* is empty.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Pillow is required for GIF export: pip install Pillow") from exc

    if not frames:
        raise ValueError("frames list is empty")

    duration_ms = max(1, int(1000 / fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imgs = [Image.open(str(p)).convert("RGBA") for p in frames]
    imgs[0].save(
        str(output_path),
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
    )
    logger.info("GIF saved → %s  (%d frames, %d fps)", output_path, len(frames), fps)
    return output_path


# ---------------------------------------------------------------------------
# MP4 builder (ffmpeg)
# ---------------------------------------------------------------------------

def mp4_from_frames(
    frames: list[Path],
    output_path: Path,
    fps: int = 10,
    crf: int = 23,
) -> Path:
    """Build an MP4 animation using ffmpeg.

    Writes frames to a temporary directory as sequentially numbered PNGs,
    then calls ``ffmpeg`` to encode the video.

    Args:
        frames:      Ordered list of PNG file paths.
        output_path: Destination ``.mp4`` file path.
        fps:         Frames per second.
        crf:         Constant-rate factor for H.264 (lower = better quality).

    Returns:
        Path to the created MP4 file.

    Raises:
        RuntimeError: If ffmpeg is not found or encoding fails.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH.  Install ffmpeg or use fmt='gif' instead."
        )
    if not frames:
        raise ValueError("frames list is empty")

    import tempfile  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i, src in enumerate(frames):
            dst = tmp / f"frame_{i:05d}.png"
            import shutil as _shutil  # noqa: PLC0415
            _shutil.copy2(src, dst)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmp / "frame_%05d.png"),
            "-c:v", "libx264",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    logger.info("MP4 saved → %s  (%d frames, %d fps)", output_path, len(frames), fps)
    return output_path


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def create_animation(
    job_dir: Path | str,
    output_dir: Path | str | None = None,
    fps: int = 10,
    fmt: Literal["gif", "mp4"] = "gif",
    pattern: str = r"step_(\d+)\.png",
    max_frames: int = 300,
    *,
    loop: int = 0,
    crf: int = 23,
) -> Path:
    """Create an animation from job snapshot PNGs.

    Automatically discovers all ``step_XXXXXX.png`` images in *job_dir*,
    assembles them into a GIF or MP4 animation, and saves it next to the
    job outputs.

    Args:
        job_dir:    Job output directory containing PNG snapshots.
        output_dir: Directory to save the animation file.  Defaults to
                    *job_dir* itself.
        fps:        Frames per second (1–60, clamped).
        fmt:        Output format: ``'gif'`` or ``'mp4'``.  MP4 requires
                    ffmpeg; if ffmpeg is absent the function automatically
                    falls back to GIF.
        pattern:    Regex for frame PNG filenames.
        max_frames: Maximum number of frames (subsampled if needed).
        loop:       GIF loop count (0 = infinite, GIF only).
        crf:        MP4 CRF quality factor (MP4 only).

    Returns:
        Path to the created animation file.
    """
    job_dir = Path(job_dir)
    output_dir = Path(output_dir) if output_dir is not None else job_dir
    fps = max(1, min(60, fps))

    frames = frames_from_png_dir(job_dir, pattern=pattern, max_frames=max_frames)
    logger.info("Animation: %d frames found in %s", len(frames), job_dir)

    # Choose format; auto-downgrade to GIF if ffmpeg unavailable
    if fmt == "mp4" and shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found – falling back to GIF format")
        fmt = "gif"

    if fmt == "mp4":
        out_path = output_dir / "animation.mp4"
        return mp4_from_frames(frames, out_path, fps=fps, crf=crf)
    else:
        out_path = output_dir / "animation.gif"
        return gif_from_frames(frames, out_path, fps=fps, loop=loop)
