"""Subcell centre of a detached drop's side-view half-fill contour."""

import numpy as np
from contourpy import contour_generator


def centre_z(field, detached):
    projected = field.max(axis=1)
    z, _, x = np.nonzero(detached)
    candidates = []
    for outline in contour_generator(z=projected).lines(0.5):
        if len(outline) < 4 or not np.array_equal(outline[0], outline[-1]):
            continue
        xx, zz = outline.T
        if not (xx.min() <= x.mean() <= xx.max() and zz.min() <= z.mean() <= zz.max()):
            continue
        cross = xx[:-1] * zz[1:] - xx[1:] * zz[:-1]
        twice_area = cross.sum()
        if abs(twice_area) > 0:
            centroid = np.sum((zz[1:] + zz[:-1]) * cross) / (3 * twice_area)
            candidates.append((abs(twice_area), float(centroid)))
    return max(candidates)[1] if candidates else None


def summarize(samples, frame_interval_ms=1.0):
    if len(samples) < 2:
        return dict(
            samples=samples,
            fitted_downward_speed_m_s=None,
            recorded_interval_ms=None,
            camera_interval_speed_m_s=None,
        )
    t, z = np.array([(s["time_ms"], s["centre_height_mm"]) for s in samples]).T
    slope = np.polyfit(t - t.mean(), z, 1)[0]
    camera_speed = None
    if t[-1] - t[0] >= frame_interval_ms:
        previous = np.interp(t[-1] - frame_interval_ms, t, z)
        camera_speed = float((previous - z[-1]) / frame_interval_ms)
    return dict(
        samples=samples,
        camera_interval_speed_m_s=camera_speed,
        fitted_downward_speed_m_s=float(-slope),
        recorded_interval_ms=float(t[-1] - t[0]),
    )
