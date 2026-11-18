#!/usr/bin/env python3
"""What the calibration's 21 cubes (16 Nov, cube tops, seen by today's calibration camera) say about the base's turn,
straight from the firmware's pulses -- not on top of TURN_DEG; the old hand rows not used. Then: does a pattern
fitted to them predict today's hand-measured 90s (pulse 870 = -90, 116 = +90; 500 = 0 exactly)? No arm.

Each cube at its raw.py reach (the reach the same at every turn), at the bearing f - E(f) + the frame's turn, about
the axis: the axis free, or held at the nuts' centre. E = firmware minus real, deg.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/servo/calturn.py
"""
import json
import math
from pathlib import Path

import numpy as np

from applib.camera import store
from applib.camera.base import CUP_SPACING_MM, find_base

PULSE_DEG = 0.24
rows = json.loads(Path("agenttools/servo/out/points.json").read_text())
f = np.array([r["raw_bearing"] for r in rows])
F = np.array([math.hypot(*r["formula"]) for r in rows])      # raw.py's true reach (the turn does not change it)
seen = np.array([r["seen3"] for r in rows])
saved = store.load_camera(Path("config"))
alone = saved.cube_fit()
found = find_base(store.load_frame(Path("config")), alone.desk_to_pixel / np.linalg.norm(alone.desk_to_pixel[2, :3]))
nuts = {p.corner: np.array([p.x, p.y]) for p in found.parts if p.is_nut and p.found}
model = np.array([[cx * CUP_SPACING_MM / 2, cy * CUP_SPACING_MM / 2] for cx, cy in nuts])
at = np.array(list(nuts.values()))
u, _, vt = np.linalg.svd((model - model.mean(0)).T @ (at - at.mean(0)))
NUTS = at.mean(0) - (u @ vt).T @ model.mean(0)


def basis(name, fv):
    fv = np.atleast_1d(np.asarray(fv, float))
    tv = np.radians(fv)
    return {"firmware as is (0.24 deg a pulse)": [],
            "one scale, the same every pulse": [fv / 90],
            "tapered (deg a pulse changing evenly)": [fv / 90, (fv / 90) ** 2],
            "S (sin 2f)": [np.sin(2 * tv)],
            "S + one scale": [np.sin(2 * tv), fv / 90],
            "S + tapered": [np.sin(2 * tv), fv / 90, (fv / 90) ** 2]}[name]


NAMES = ["firmware as is (0.24 deg a pulse)", "one scale, the same every pulse", "tapered (deg a pulse changing evenly)",
         "S (sin 2f)", "S + one scale", "S + tapered"]


def e_of(name, c, fv):
    cols = basis(name, fv)
    return sum(ci * b for ci, b in zip(c, cols)) if cols else np.zeros(np.size(fv))


def misses(name, q, idx, axis_free):
    n = len(basis(name, 0.0))
    c = q[:n]
    phi = q[n]
    axis = q[n + 1:n + 3] if axis_free else NUTS
    b = np.radians(f[idx] - e_of(name, c, f[idx]) + phi)
    pred = axis + F[idx, None] * np.column_stack([np.sin(b), -np.cos(b)])
    return (pred - seen[idx]).ravel()


def solve(name, idx, axis_free):
    n = len(basis(name, 0.0))
    q = np.zeros(n + 1 + (2 if axis_free else 0))
    if axis_free:
        q[n + 1:] = NUTS
    for _ in range(80):
        r0 = misses(name, q, idx, axis_free)
        J = np.column_stack([(misses(name, q + d, idx, axis_free) - r0) / 1e-5 for d in np.eye(len(q)) * 1e-5])
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        q = q + step
        if np.abs(step).max() < 1e-8:
            break
    return q, misses(name, q, idx, axis_free)


def pulse_for(name, c, target):
    """Today's paper: 500 is exactly 0, so E counted from E(0)."""
    fv = target
    for _ in range(60):
        fv = target + float(e_of(name, c, fv)[0] - e_of(name, c, 0.0)[0])
    return 500 - fv / PULSE_DEG


every = np.arange(len(rows))
print(f"{len(rows)} cubes, firmware bearing {f.min():+.0f} to {f.max():+.0f}, reach {F.min():.0f}-{F.max():.0f} mm; "
      f"nuts' centre {np.round(NUTS, 2)}")
for axis_free in (True, False):
    print(f"\n=== the axis {'free' if axis_free else 'held at the nuts centre'}; mm across the desk (in and out and "
          f"sideways together), each cube left out; then today's 90s from it (measured 870, 116)")
    print(f"{'pattern':40s} {'terms (deg at f=90)':>26s} {'rms':>5s} {'left out':>8s} | -90 pulse, +90 pulse | "
          f"deg a pulse at -75 / -45 / 0 / +45 / +90")
    for name in NAMES:
        q, res = solve(name, every, axis_free)
        n = len(basis(name, 0.0))
        lo = []
        for i in every:
            qi, _ = solve(name, every[every != i], axis_free)
            lo.append(np.linalg.norm(misses(name, qi, np.array([i]), axis_free)))
        rms = np.sqrt(np.mean(res.reshape(-1, 2) ** 2) * 2)
        terms = ", ".join(f"{v:+.2f}" for v in q[:n]) or "-"
        slopes = []
        for fv in (-75.0, -45.0, 0.0, 45.0, 90.0):
            h = 0.01
            de = (e_of(name, q[:n], fv + h)[0] - e_of(name, q[:n], fv - h)[0]) / (2 * h)
            slopes.append(PULSE_DEG * (1 - de))
        print(f"{name:40s} {terms:>26s} {rms:5.2f} {np.sqrt(np.mean(np.square(lo))):8.2f} | "
              f"{pulse_for(name, q[:n], -90.0):6.1f}, {pulse_for(name, q[:n], 90.0):6.1f}     | "
              + " / ".join(f"{s:.4f}" for s in slopes))

# --- today's three, with the calibration's cubes: one fit; how far each lands --------------------------------------
print("\n=== the calibration's cubes and today's 870 / 500 / 116 together (axis free): one scale against tapered")
for name in ("one scale, the same every pulse", "tapered (deg a pulse changing evenly)", "S + tapered"):
    n = len(basis(name, 0.0))

    def all_misses(q):
        cubes = misses(name, q, every, True)
        e0 = e_of(name, q[:n], 0.0)[0]
        hand = []
        for p, real in ((870, -90.0), (116, 90.0)):
            fv = (500 - p) * PULSE_DEG
            hand.append((fv - (e_of(name, q[:n], fv)[0] - e0) - real) * math.radians(1) * 240.0)   # mm at 240 out
        return np.concatenate([cubes, np.array(hand)])
    q = np.zeros(n + 3)
    q[n + 1:] = NUTS
    for _ in range(80):
        r0 = all_misses(q)
        J = np.column_stack([(all_misses(q + d) - r0) / 1e-5 for d in np.eye(len(q)) * 1e-5])
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        q = q + step
        if np.abs(step).max() < 1e-8:
            break
    r0 = all_misses(q)
    cubes = r0[:-2].reshape(-1, 2)
    print(f"   {name:40s} cubes {np.sqrt(np.mean(cubes ** 2) * 2):.2f} mm rms; today's -90 {r0[-2]:+.2f} mm, +90 "
          f"{r0[-1]:+.2f} mm at 240 out; terms " + ", ".join(f"{v:+.2f}" for v in q[:n]))
