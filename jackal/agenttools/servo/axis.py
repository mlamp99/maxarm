#!/usr/bin/env python3
"""The arm's axis in the camera's picture, every way the data give it, per camera (the saved three-cube one, and one
refitted to all points' cube edges). No arm.

- the nuts as found: the base finder's 120 mm square laid on the two front nuts;
- the nuts made 120 mm apart: both moved along the camera's line of sight by one height until they are 120 apart
  (they are found 120.7-122), the square laid on them;
- the reaches, offset free: each cube as far from the axis as its raw reach plus one amount (circle.py);
- the reaches, offset held: that amount held at the owner's stacks and grid (raw + 1.9 at the cube tops).

About each: the reach misses, and the turn (firmware minus real) against raw.py's TURN_DEG, by side.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/servo/axis.py
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from applib.camera import store
from applib.camera.base import CORNERS, CUP_SPACING_MM, find_base
from maxarm import raw

C = Path("config")
OWNER_REACH_ADD = 1.9            # raw to real at the cube tops, the owner's 13 Nov grid and stacks (grid.py)
rows = json.loads(Path("agenttools/servo/out/points.json").read_text())
saved = store.load_camera(C)
alone = saved.cube_fit()
frame = store.load_frame(C)
reach = np.array([r["raw_reach"] for r in rows])
bearing = np.array([r["raw_bearing"] for r in rows])
table = np.array([-math.degrees(raw._further_round(math.radians(b))) for b in bearing])   # fw - real by TURN_DEG


def camera_centre(true_to_pixel):
    """The camera's centre in the cubes' desk space, x, y, and its height."""
    centre = cv2.decomposeProjectionMatrix(true_to_pixel)[2]
    centre = (centre[:3] / centre[3]).ravel()
    return np.linalg.solve(alone.desk_to_true, centre[:2]), centre[2]


def square_on(nuts):
    """The 120 mm square laid on nuts {corner: (x, y)}: its centre (the axis), a turn and shift by least squares."""
    model = np.array([[cx * CUP_SPACING_MM / 2, cy * CUP_SPACING_MM / 2] for cx, cy in nuts])
    found = np.array(list(nuts.values()))
    m0, f0 = model.mean(axis=0), found.mean(axis=0)
    h = (model - m0).T @ (found - f0)
    u, _, vt = np.linalg.svd(h)
    rot = (u @ vt).T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = (u @ vt).T
    return f0 - rot @ m0, math.degrees(math.atan2(rot[1, 0], rot[0, 0]))


def radial(seen, add=None):
    params = np.array([2.0, 0.0, 2.0 if add is None else add])
    for _ in range(60):
        d = np.linalg.norm(seen - params[:2], axis=1)
        res = d - (reach + params[2])
        J = np.column_stack([-(seen - params[:2]) / d[:, None]] + ([-np.ones(len(d))] if add is None else []))
        delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        params[:J.shape[1]] += delta
        if np.abs(delta).max() < 1e-9:
            break
    return params[:2], params[2]


def report(name, axis, seen, add=None):
    d = np.linalg.norm(seen - axis, axis=1)
    off = d - reach
    add_used = off.mean() if add is None else add
    rel = seen - axis
    desk_bearing = np.degrees(np.arctan2(rel[:, 0], -rel[:, 1]))
    straight = np.abs(bearing) < 3.0
    frame_turn = np.mean(desk_bearing[straight] - bearing[straight])
    wide = bearing - (desk_bearing - frame_turn) - table
    by_side = [wide[bearing < -45].mean(), wide[(bearing >= -45) & (bearing < -25)].mean(),
               wide[np.abs(bearing) <= 25].mean(), wide[bearing > 25].mean()]
    print(f"  {name:34s} ({axis[0]:+6.2f}, {axis[1]:+6.2f})  reach = raw + {add_used:4.2f}, misses "
          f"{np.sqrt(np.mean((off - add_used) ** 2)):.2f} rms | turn against TURN_DEG, deg: far left "
          f"{by_side[0]:+.2f}, left {by_side[1]:+.2f}, middle {by_side[2]:+.2f}, right {by_side[3]:+.2f}; rms "
          f"{np.sqrt(np.mean(wide ** 2)):.2f}")


cameras = [("saved three-cube camera", alone.desk_to_pixel, alone.true_to_pixel, "seen3")]
refit = Path("agenttools/servo/out/camera-refit.json")
if refit.exists():
    stored = json.loads(refit.read_text())
    lift = np.eye(4); lift[:2, :2] = alone.desk_to_true
    P = np.array(stored["true_to_pixel"])
    cameras.append(("camera refitted to all cubes", P @ lift, P, "seenN"))
for camera, desk_to_pixel, true_to_pixel, key in cameras:
    seen = np.array([r[key] for r in rows])
    found = find_base(frame, desk_to_pixel / np.linalg.norm(desk_to_pixel[2, :3]))
    nuts = {part.corner: np.array([part.x, part.y]) for part in found.parts if part.is_nut and part.found}
    (cx, cy), height = camera_centre(true_to_pixel)
    print(f"\n=== {camera}: camera at desk ({cx:.0f}, {cy:.0f}), {height:.0f} mm up; nuts found: "
          + ", ".join(f"{k} ({v[0]:+.2f}, {v[1]:+.2f})" for k, v in nuts.items()))
    corners = list(nuts)
    if len(corners) >= 2:
        gap = np.linalg.norm(nuts[corners[0]] - nuts[corners[1]])
        print(f"  the two nuts {gap:.2f} mm apart (120 real)")
        report("nuts as found", square_on(nuts)[0], seen)

        def moved(lift_mm):
            out = {}
            for k, v in nuts.items():
                # the nut taken as standing lift_mm higher than the finder's model: along the line of sight from the
                # camera, nearer to it (the model's nut centred about 13 mm up)
                t = (13.0 + lift_mm - height) / (13.0 - height)
                out[k] = np.array([cx, cy]) + t * (v - np.array([cx, cy]))
            return out
        lo, hi = -40.0, 40.0
        for _ in range(80):
            mid = (lo + hi) / 2
            sep = np.linalg.norm(np.subtract(*moved(mid).values()))
            lo, hi = (mid, hi) if sep > 120.0 else (lo, mid)
        print(f"  made 120 apart: the nuts as if {mid:+.1f} mm off the model's height")
        report("nuts made 120 apart", square_on(moved(mid))[0], seen)
    axis, add = radial(seen)
    report("reaches, offset free", axis, seen)
    axis, _ = radial(seen, OWNER_REACH_ADD)
    report(f"reaches, offset held at +{OWNER_REACH_ADD}", axis, seen, OWNER_REACH_ADD)
