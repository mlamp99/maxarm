#!/usr/bin/env python3
"""The calibration points on file at the cube tops, servo by servo, against where the camera saw each cube -- with
the arm's axis left free in the camera's picture instead of where the base's nuts put it. No arm.

For each point: the firmware's reading (raw x, y, z), its servo angles and pulses (geometry.inverse), the formula's
true bearing and reach, and the cube by the camera -- the saved three-cube camera, and one camera refitted to every
point's cube edges (40 mm cubes, no arm readings). Then shapes of the arm's error fitted with the frame free (the
arm's axis anywhere, its turn any): which reach and turn terms the points want, judged one point left out.
Writes out/points.json.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/servo/points.py
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from applib.camera import edges, store
from applib.camera.model import CUBE_MM, cube_corners, least_squares
from applib.camera.sighting import in_frame, place, start
from calibration.camera import anchor
from maxarm import geometry, raw

C = Path("config")
OUT = Path("agenttools/servo/out")
saved = store.load_camera(C)
alone = saved.cube_fit()
points = [p for p in store.load_refine_points(C) if p.edges and p.raw]
tps = anchor.tie_points(alone, points)


def refit_camera():
    """One camera fitted to every point's cube by its edges alone: each cube's desk x, y in the cubes' desk space."""
    K, R, centre = cv2.decomposeProjectionMatrix(alone.true_to_pixel)[:3]
    K = K / K[2, 2]; centre = (centre[:3] / centre[3]).ravel()
    cubes = []
    for p in points:
        m = in_frame(alone.true_to_pixel, alone.frame_size, p.sighting)
        cubes.append(place(m, p.sighting, start(m, p.sighting)[:3]))
    cam = 9
    first = np.array([K[0, 0], K[0, 2], K[1, 2], *cv2.Rodrigues(R)[0].ravel(), *(-R @ centre), *np.concatenate(cubes)])
    steps = np.array([0.05, 0.05, 0.05, 1e-6, 1e-6, 1e-6, 0.005, 0.005, 0.005] + [0.005, 0.005, 1e-5] * len(cubes))

    def matrix(params):
        k = np.array([[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1.0]])
        return k @ np.column_stack((cv2.Rodrigues(params[3:6])[0], params[6:9]))

    def misses(params):
        P = matrix(params); out = []
        for i, p in enumerate(points):
            x, y, t = params[cam + 3 * i: cam + 3 * i + 3]
            w = np.column_stack((cube_corners((x, y), t, CUBE_MM), np.ones(8))) @ P.T
            out.append(edges.signed_distances(w[:, :2] / w[:, 2:], *p.sighting.edges))
        return np.concatenate(out)

    free = [i for i in range(len(first)) if i not in (cam, cam + 1, cam + 2)]     # the first cube holds the world
    solved, res, _ = least_squares(misses, first, steps, huber=1.0, free=free, iterations=100)
    seen = [np.linalg.solve(alone.desk_to_true, solved[cam + 3 * i: cam + 3 * i + 2]) for i in range(len(points))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "camera-refit.json").write_text(json.dumps({"true_to_pixel": matrix(solved).tolist()}))
    return seen, solved[:3], float(np.median(np.abs(res)))


def row_of(point, tp):
    rx, ry, rz = tp.raw
    angles = geometry.inverse(tp.raw)
    pulses = geometry.deg_to_pulse(angles)
    true = raw.formula_true(tp.raw)
    return {"id": point.id, "cube": point.cube, "taken": point.taken, "raw": [rx, ry, rz],
            "raw_bearing": math.degrees(math.atan2(rx, -ry)), "raw_reach": math.hypot(rx, ry),
            "angles": list(angles), "pulses": list(pulses),
            "bearing": math.degrees(math.atan2(true[0], -true[1])), "reach": math.hypot(true[0], true[1]),
            "true_z": true[2], "formula": [true[0], true[1]], "seen3": list(map(float, tp.seen))}


rows = [row_of(p, tp) for p, tp in zip(points, tps)]
seen_all, intrinsics, edge_px = refit_camera()
for row, seen in zip(rows, seen_all):
    row["seenN"] = list(map(float, seen))
print(f"{len(rows)} points; the camera refitted to all their cubes: f {intrinsics[0]:.1f}, principal "
      f"({intrinsics[1]:.0f}, {intrinsics[2]:.0f}), edges median {edge_px:.2f} px")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "points.json").write_text(json.dumps(rows, indent=1))

# --- the shapes, the frame free ---------------------------------------------------------------------------------

TERMS = {      # name: (out mm, across mm) per unit, at formula bearing b (deg) and reach R (mm)
    "reach +": lambda b, R: (1.0, 0.0),
    "reach per 100 mm": lambda b, R: ((R - 190.0) / 100.0, 0.0),
    "turn wider per 60 deg": lambda b, R: (0.0, R * math.radians(b / 60.0)),
    "cup aside": lambda b, R: (0.0, 1.0),
}
MODELS = [[], ["reach +"], ["reach +", "reach per 100 mm"], ["reach +", "turn wider per 60 deg"],
          ["reach +", "cup aside"], ["reach +", "reach per 100 mm", "turn wider per 60 deg", "cup aside"]]


def predicted(params, model, row):
    """Where the arm puts the cube in the camera's desk space: the formula's place, the model's terms on it, laid by
    the frame (turn, then the axis)."""
    turn, ax, ay = params[:3]
    b, R = row["bearing"], row["reach"]
    u = np.array([math.sin(math.radians(b)), -math.cos(math.radians(b))]); v = np.array([-u[1], u[0]])
    place = np.array(row["formula"], float)
    for value, name in zip(params[3:], model):
        o, a = TERMS[name](b, R)
        place = place + value * (o * u + a * v)
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    return np.array([[c, -s], [s, c]]) @ place + np.array([ax, ay])


def solve(model, rows, seen_key):
    params = np.array([saved.anchor["turn_deg"], *saved.anchor["axis"]] + [0.0] * len(model))
    for _ in range(30):
        res = np.concatenate([predicted(params, model, r) - r[seen_key] for r in rows])
        J = np.zeros((len(res), len(params)))
        for j in range(len(params)):
            step = np.zeros(len(params)); step[j] = 1e-4
            J[:, j] = (np.concatenate([predicted(params + step, model, r) - r[seen_key] for r in rows]) - res) / 1e-4
        delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        params = params + delta
        if np.abs(delta).max() < 1e-7:
            break
    return params


def split(params, model, row, seen_key):
    """The point's miss, out and across (mm, about the arm's axis as the fit lays it)."""
    turn, ax, ay = params[:3]
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    miss = np.array([[c, s], [-s, c]]) @ (np.array(row[seen_key]) - predicted(params, model, row))
    b = math.radians(row["bearing"]); u = np.array([math.sin(b), -math.cos(b)]); v = np.array([-u[1], u[0]])
    return float(miss @ u), float(miss @ v)


for seen_key, camera in (("seen3", "saved three-cube camera"), ("seenN", f"camera refitted to all {len(rows)} cubes")):
    print(f"\n=== {camera}: the frame free (axis, turn), each model's terms; misses mm (typical / worst), each point "
          f"left out")
    for model in MODELS:
        params = solve(model, rows, seen_key)
        fitted = [math.hypot(*split(params, model, r, seen_key)) for r in rows]
        left = []
        for i, r in enumerate(rows):
            others = rows[:i] + rows[i + 1:]
            left.append(math.hypot(*split(solve(model, others, seen_key), model, r, seen_key)))
        base = saved.anchor["base"]
        axis_off = math.dist(params[1:3], (base["x"], base["y"]))
        terms = ", ".join(f"{name} {value:+.2f}" for name, value in zip(model, params[3:])) or "the formula alone"
        print(f"  {terms:72s} fitted {np.median(fitted):.2f} / {max(fitted):.2f}  left out {np.median(left):.2f} / "
              f"{max(left):.2f}  axis ({params[1]:+.2f}, {params[2]:+.2f}), {axis_off:.2f} from the nuts' base")

# --- each point, the frame free ---------------------------------------------------------------------------------
print("\n=== each point through the refitted camera; out / across mm after: [nuts] the saved frame, formula alone; "
      "[free] the frame free, formula alone; [free+reach] and a reach offset; [full] the last model above")
tie = anchor.Tie(tuple(saved.anchor["axis"]), float(saved.anchor["turn_deg"]), None, [])
fits = {name: (model, solve(model, rows, "seenN")) for name, model in
        (("free", []), ("free+reach", ["reach +"]), ("full", MODELS[-1]))}
print(f"{'#':>3} {'taken':>8} {'raw b':>7} {'raw R':>6} {'raw z':>5} {'p1':>6} {'p2':>6} {'p3':>6} | "
      + " | ".join(f"{n:>13}" for n in ["nuts"] + list(fits)))
for r in sorted(rows, key=lambda r: r["raw_bearing"]):
    s = tie.to_arm(r["seenN"]); q = np.array(r["formula"])
    b = math.radians(r["bearing"]); u = np.array([math.sin(b), -math.cos(b)]); v = np.array([-u[1], u[0]])
    cells = [f"{(s - q) @ u:+5.2f} {(s - q) @ v:+5.2f}"]
    for name, (model, params) in fits.items():
        o, a = split(params, model, r, "seenN"); cells.append(f"{o:+5.2f} {a:+5.2f}")
    p1, p2, p3 = r["pulses"]
    print(f"{r['id']:3d} {r['taken'][7:15]:>8} {r['raw_bearing']:+7.2f} {r['raw_reach']:6.1f} {r['raw'][2]:5.1f} "
          f"{p1:6.1f} {p2:6.1f} {p3:6.1f} | " + " | ".join(f"{c:>13}" for c in cells))
