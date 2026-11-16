#!/usr/bin/env python3
"""The camera refitted to every point's cube by its edges alone -- 40 mm cubes, no arm reading -- against the saved
three-cube camera: the base's nuts through each (how far apart), and each point's reach and turn error after the
formula through each, about the base each finds. No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/shape/camera18.py
"""
import math
from pathlib import Path

import cv2
import numpy as np

from applib.camera import edges, store
from applib.camera.base import find_base
from applib.camera.model import CUBE_MM, cube_corners, least_squares
from applib.camera.sighting import in_frame, place, start
from calibration.camera import anchor

C = Path("config")
saved = store.load_camera(C)
fit = saved.cube_fit()
points = [p for p in store.load_refine_points(C) if p.edges]
frame = store.load_frame(C)
K, R, centre = cv2.decomposeProjectionMatrix(fit.true_to_pixel)[:3]
K = K / K[2, 2]; centre = (centre[:3] / centre[3]).ravel()
cubes = []
for p in points:
    m = in_frame(fit.true_to_pixel, fit.frame_size, p.sighting)
    cubes.append(place(m, p.sighting, start(m, p.sighting)[:3]))
CAM = 9
first = np.array([K[0, 0], K[0, 2], K[1, 2], *cv2.Rodrigues(R)[0].ravel(), *(-R @ centre), *np.concatenate(cubes)])
steps = np.array([0.05, 0.05, 0.05, 1e-6, 1e-6, 1e-6, 0.005, 0.005, 0.005] + [0.005, 0.005, 1e-5] * len(cubes))


def matrix(params):
    k = np.array([[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1.0]])
    return k @ np.column_stack((cv2.Rodrigues(params[3:6])[0], params[6:9]))


def misses(params):
    P = matrix(params); out = []
    for i, p in enumerate(points):
        x, y, t = params[CAM + 3 * i: CAM + 3 * i + 3]
        w = np.column_stack((cube_corners((x, y), t, CUBE_MM), np.ones(8))) @ P.T
        out.append(edges.signed_distances(w[:, :2] / w[:, 2:], *p.sighting.edges))
    return np.concatenate(out)


free = [i for i in range(len(first)) if i not in (CAM, CAM + 1, CAM + 2)]     # the first point holds the world
solved, res, _ = least_squares(misses, first, steps, huber=1.0, free=free, iterations=100)
P18 = matrix(solved)
print(f"18-cube camera: f {solved[0]:.1f} (saved {K[0,0]:.1f}), principal ({solved[1]:.0f}, {solved[2]:.0f}) (saved "
      f"({K[0,2]:.0f}, {K[1,2]:.0f})); edges median {np.median(np.abs(res)):.2f} px, 90% {np.percentile(np.abs(res), 90):.2f}")

lift = np.eye(4); lift[:2, :2] = fit.desk_to_true
for name, P in (("saved 3-cube", fit.true_to_pixel), ("18-cube", P18)):
    d2p = P @ lift
    base = find_base(frame, d2p / np.linalg.norm(d2p[2, :3]))
    nuts = [part for part in base.parts if part.is_nut and part.found]
    gap = math.dist((nuts[0].x, nuts[0].y), (nuts[1].x, nuts[1].y)) if len(nuts) == 2 else float("nan")
    print(f"\n{name}: {base.describe()}; the two front nuts found alone {gap:.1f} mm apart (120 real)")
    tie = anchor.Tie((base.x, base.y), base.turn_deg, base, [])
    tps = anchor.tie_points(fit, points)
    rows = []
    for i, (p, tp) in enumerate(zip(points, tps)):
        true = (solved[CAM + 3 * i: CAM + 3 * i + 2] if name == "18-cube" else None)
        seen = np.linalg.solve(fit.desk_to_true, true) if true is not None else np.asarray(tp.seen)
        s = tie.to_arm(seen); q = tp.reading
        r = math.hypot(*q); u = q / r; v = np.array([-u[1], u[0]]); d = s - q
        rows.append((math.degrees(math.atan2(q[0], -q[1])), r, p.id, d @ u, d @ v))
    for b, r, i, o, a in sorted(rows):
        print(f"   #{i:2d} b {b:+6.1f} R {r:5.1f}: out {o:+5.2f}  across {a:+5.2f}")
    outs = np.array([o for *_, o, a in rows]); acr = np.array([a for *_, a in rows])
    print(f"   out: mean {outs.mean():+.2f}, spread (sd) {outs.std():.2f}; across sd {acr.std():.2f}")
