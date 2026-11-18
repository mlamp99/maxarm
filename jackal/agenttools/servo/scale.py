#!/usr/bin/env python3
"""The axis held at the nuts (the base finder's square on the two front nuts, per camera), the cube-top reach as
raw + c + s (R - 190) / 100: s is the camera's scale against the arm's raw reach, mm per 100 mm. What the reach
leaves then, by bearing; each point left out. The owner's 13 Nov grid (grid.py) through the same fit, about the
grid's origin. No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/servo/scale.py
"""
import json
import math
from pathlib import Path

import numpy as np

from applib.camera import store
from applib.camera.base import CUP_SPACING_MM, find_base

C = Path("config")
rows = json.loads(Path("agenttools/servo/out/points.json").read_text())
reach = np.array([r["raw_reach"] for r in rows])
bearing = np.array([r["raw_bearing"] for r in rows])
saved = store.load_camera(C)
alone = saved.cube_fit()
frame = store.load_frame(C)


def fit(d, R, scale):
    A = np.column_stack([np.ones(len(R))] + ([(R - 190.0) / 100.0] if scale else []))
    p = np.linalg.lstsq(A, d - R, rcond=None)[0]
    res = d - R - A @ p
    left = []
    for i in range(len(R)):
        keep = np.arange(len(R)) != i
        q = np.linalg.lstsq(A[keep], (d - R)[keep], rcond=None)[0]
        left.append(d[i] - R[i] - A[i] @ q)
    return p, res, np.array(left)


def pattern(res, b):
    """Once- and twice-a-turn parts of what is left, mm, by the firmware's bearing."""
    t = np.radians(b)
    A = np.column_stack([np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)])
    p = np.linalg.lstsq(A, res, rcond=None)[0]
    return math.hypot(p[0], p[1]), math.hypot(p[2], p[3])


def show(name, d, R, b):
    for scale in (False, True):
        p, res, left = fit(d, R, scale)
        once, twice = pattern(res, b)
        text = f"raw + {p[0]:.2f}" + (f" + {p[1]:+.2f} per 100 mm past 190" if scale else "")
        print(f"  {name:28s} {text:40s} misses {np.sqrt(np.mean(res ** 2)):.2f} rms, worst {np.abs(res).max():.2f}; "
              f"left out {np.sqrt(np.mean(left ** 2)):.2f} rms; left over once a turn {once:.2f}, twice {twice:.2f}")


lift = np.eye(4); lift[:2, :2] = alone.desk_to_true
P = np.array(json.loads(Path("agenttools/servo/out/camera-refit.json").read_text())["true_to_pixel"])
print(f"camera points: reach {reach.min():.0f}-{reach.max():.0f} mm, bearing {bearing.min():+.0f} to {bearing.max():+.0f}")
for camera, desk_to_pixel, key in (("saved three-cube camera", alone.desk_to_pixel, "seen3"),
                                   ("camera refitted to all cubes", P @ lift, "seenN")):
    seen = np.array([r[key] for r in rows])
    found = find_base(frame, desk_to_pixel / np.linalg.norm(desk_to_pixel[2, :3]))
    nuts = {part.corner: np.array([part.x, part.y]) for part in found.parts if part.is_nut and part.found}
    model = np.array([[cx * CUP_SPACING_MM / 2, cy * CUP_SPACING_MM / 2] for cx, cy in nuts])
    at = np.array(list(nuts.values()))
    u, _, vt = np.linalg.svd((model - model.mean(0)).T @ (at - at.mean(0)))
    axis = at.mean(0) - (u @ vt).T @ model.mean(0)              # the 120 mm square's centre, laid on the nuts
    gap = np.linalg.norm(at[0] - at[1])
    print(f"\n=== {camera}: nuts {gap:.2f} apart ({(gap / 120 - 1) * 100:+.2f}%), axis ({axis[0]:+.2f}, {axis[1]:+.2f})")
    show("about the nuts", np.linalg.norm(seen - axis, axis=1), reach, bearing)

GRID = [((-160.0, 0.0), (-158.39, -0.66)), ((-120.0, -120.0), (-122.53, -115.06)), ((0.0, -162.25), (0.0, -160.0)),
        ((160.0, -160.0), (164.42, -152.52)), ((160.0, -124.8), (160.45, -120.73)), ((160.0, -100.0), (159.88, -96.16)),
        ((160.0, -100.0), (160.66, -95.64)), ((160.2, -80.25), (160.38, -76.15)), ((200.0, 0.0), (197.74, 9.12)),
        ((205.0, 0.0), (204.07, 9.41))]
real = np.array([g[0] for g in GRID]); raws = np.array([g[1] for g in GRID])
R = np.hypot(raws[:, 0], raws[:, 1]); b = np.degrees(np.arctan2(raws[:, 0], -raws[:, 1]))
print(f"\n=== the owner's 13 Nov grid without the +51 spot: reach {R.min():.0f}-{R.max():.0f} mm")
show("about the grid's origin", np.linalg.norm(real, axis=1), R, b)

# --- the arm's formula (raw.py: the reach formula, TURN_DEG) taken as true, laid on the camera's cubes ------------
print("\n=== raw.py's true x, y laid on the camera's cubes: turn + shift; + one scale; + a stretch (any 2x2). Where the "
      "arm's axis (0, 0) lands against the nuts' centre")
form = np.array([r["formula"] for r in rows])


def lay(src, dst, kind):
    n = len(src)
    if kind == "affine":
        A = np.zeros((2 * n, 6)); A[0::2, 0:2] = src; A[0::2, 4] = 1; A[1::2, 2:4] = src; A[1::2, 5] = 1
        p = np.linalg.lstsq(A, dst.ravel(), rcond=None)[0]
        return p[:4].reshape(2, 2), p[4:]
    ms, md = src.mean(0), dst.mean(0)
    u, sv, vt = np.linalg.svd((src - ms).T @ (dst - md))
    rot = (u @ vt).T
    k = sv.sum() / ((src - ms) ** 2).sum() if kind == "scale" else 1.0
    return k * rot, md - k * rot @ ms


for camera, desk_to_pixel, key in (("saved three-cube camera", alone.desk_to_pixel, "seen3"),
                                   ("camera refitted to all cubes", P @ lift, "seenN")):
    seen = np.array([r[key] for r in rows])
    found = find_base(frame, desk_to_pixel / np.linalg.norm(desk_to_pixel[2, :3]))
    nuts = {part.corner: np.array([part.x, part.y]) for part in found.parts if part.is_nut and part.found}
    model = np.array([[cx * CUP_SPACING_MM / 2, cy * CUP_SPACING_MM / 2] for cx, cy in nuts])
    at = np.array(list(nuts.values()))
    u, _, vt = np.linalg.svd((model - model.mean(0)).T @ (at - at.mean(0)))
    centre = at.mean(0) - (u @ vt).T @ model.mean(0)
    print(f"  {camera}: nuts' centre ({centre[0]:+.2f}, {centre[1]:+.2f})")
    for kind in ("turn", "scale", "affine"):
        M, t = lay(form, seen, kind)
        res = np.linalg.norm(form @ M.T + t - seen, axis=1)
        left = []
        for i in range(len(rows)):
            keep = np.arange(len(rows)) != i
            Mi, ti = lay(form[keep], seen[keep], kind)
            left.append(np.linalg.norm(Mi @ form[i] + ti - seen[i]))
        sv = np.linalg.svd(M, compute_uv=False)
        uu = np.linalg.svd(M)[0]
        long_dir = math.degrees(math.atan2(uu[0, 0], -uu[1, 0])) % 180
        print(f"    {kind:6s} misses {np.sqrt(np.mean(res ** 2)):.2f} rms, worst {res.max():.2f}; left out "
              f"{np.sqrt(np.mean(np.square(left))):.2f} rms, worst {max(left):.2f}; scale {(sv[0] - 1) * 100:+.2f}% / "
              f"{(sv[1] - 1) * 100:+.2f}% (the larger along bearing {long_dir:.0f} deg); the arm's axis lands at "
              f"({t[0]:+.2f}, {t[1]:+.2f}), {np.linalg.norm(t - centre):.2f} mm from the nuts' centre")

# --- a constant extra reach against a scale; the camera's reach minus raw.py's by bearing, about the nuts -------
print("\n=== turn + shift + an extra reach on raw.py's (+ a scale); then, about the nuts, camera reach minus raw.py's")
unit = form / np.linalg.norm(form, axis=1)[:, None]


def with_reach(seen, idx, scale):
    best = None
    for c in np.linspace(-2, 6, 801):
        src = form[idx] + c * unit[idx]
        M, t = lay(src, seen[idx], "scale" if scale else "turn")
        r = np.sqrt(np.mean(np.sum((src @ M.T + t - seen[idx]) ** 2, axis=1)))
        best = best if best and best[0] <= r else (r, c, M, t)
    return best


for camera, desk_to_pixel, key in (("saved three-cube camera", alone.desk_to_pixel, "seen3"),
                                   ("camera refitted to all cubes", P @ lift, "seenN")):
    seen = np.array([r[key] for r in rows])
    found = find_base(frame, desk_to_pixel / np.linalg.norm(desk_to_pixel[2, :3]))
    nuts = {part.corner: np.array([part.x, part.y]) for part in found.parts if part.is_nut and part.found}
    model = np.array([[cx * CUP_SPACING_MM / 2, cy * CUP_SPACING_MM / 2] for cx, cy in nuts])
    at = np.array(list(nuts.values()))
    u, _, vt = np.linalg.svd((model - model.mean(0)).T @ (at - at.mean(0)))
    centre = at.mean(0) - (u @ vt).T @ model.mean(0)
    for scale in (False, True):
        r, c, M, t = with_reach(seen, np.arange(len(rows)), scale)
        left = []
        for i in range(len(rows)):
            _, ci, Mi, ti = with_reach(seen, np.arange(len(rows)) != i, scale)
            left.append(np.linalg.norm((form[i] + ci * unit[i]) @ Mi.T + ti - seen[i]))
        k = np.linalg.svd(M, compute_uv=False)[0]
        print(f"  {camera}: reach {c:+.2f}" + (f", scale {(k - 1) * 100:+.2f}%" if scale else "") + f": misses "
              f"{r:.2f} rms, left out {np.sqrt(np.mean(np.square(left))):.2f}; the arm's axis "
              f"{np.linalg.norm(t - centre):.2f} mm from the nuts' centre")
    gap = np.linalg.norm(seen - centre, axis=1) - np.linalg.norm(form, axis=1)
    print("   by bearing: " + "; ".join(f"{lo:+d}..{hi:+d} {gap[(bearing >= lo) & (bearing < hi)].mean():+.2f}"
                                       for lo, hi in ((-80, -55), (-55, -35), (-35, -10), (-10, 10), (10, 30),
                                                      (30, 50))))
