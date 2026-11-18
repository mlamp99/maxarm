#!/usr/bin/env python3
"""The owner's real points of 13 Nov (STATUS-jackal §19: the fixed grid and the plate's stack, the cup centred on a
cube by hand, the servos' reading over it) put through the same fit as circle.py: where the axis is from reaches
alone, then the base's turn about it. No camera, no arm.

    PYTHONPATH=src ../.venv/bin/python3 agenttools/servo/grid.py
"""
import math

import numpy as np

from maxarm import raw

# real x, y (the grid), raw x, y, z (the servos), what it was
GRID = [
    ((-160.0, 0.0), (-158.39, -0.66, 87.85), "left 90"),
    ((-120.0, -120.0), (-122.53, -115.06, 88.74), "left 45"),
    ((0.0, -162.25), (0.0, -160.0, 89.0), "straight, plate's stack (raw whole mm)"),
    ((160.0, -160.0), (164.42, -152.52, 83.91), "right 45"),
    ((160.0, -130.0), (163.45, -125.24, 85.79), "right 51"),
    ((160.0, -130.0), (163.27, -125.23, 85.64), "right 51 again"),
    ((160.0, -130.0), (162.22, -126.28, 85.45), "right 51 third"),
    ((160.0, -124.8), (160.45, -120.73, 85.99), "right 52"),
    ((160.0, -100.0), (159.88, -96.16, 86.61), "right 58"),
    ((160.0, -100.0), (160.66, -95.64, 86.89), "right 58 again"),
    ((160.2, -80.25), (160.38, -76.15, 87.02), "right 63"),
    ((200.0, 0.0), (197.74, 9.12, 86.82), "right 90, 200 out (owner's best)"),
    ((205.0, 0.0), (204.07, 9.41, 86.56), "right 90, 205 out, PRECISE"),
]
# Not used: the two 160 mm right-90 landings (raw y 10.57 / 10.81), their turn misrecorded (the owner, 13 Nov).
real = np.array([g[0] for g in GRID])
raws = np.array([g[1] for g in GRID])
reach = np.hypot(raws[:, 0], raws[:, 1])
bearing = np.degrees(np.arctan2(raws[:, 0], -raws[:, 1]))
formula = np.array([raw.formula_true(tuple(r)) for r in raws])
print(f"raw.py adds {np.mean(np.hypot(formula[:, 0], formula[:, 1]) - reach):.2f} mm at these tops")


def radial_fit(seen, reach, keep):
    params = np.array([0.0, 0.0, 2.0])
    for _ in range(50):
        d = np.linalg.norm(seen[keep] - params[:2], axis=1)
        res = d - (reach[keep] + params[2])
        J = np.column_stack([-(seen[keep] - params[:2]) / d[:, None], -np.ones(len(d))])
        delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        params = params + delta
        if np.abs(delta).max() < 1e-9:
            break
    return params


for name, keep in (("all", np.ones(len(GRID), bool)),
                   ("without the +51 spot", np.array(["right 51" not in g[2] for g in GRID]))):
    p = radial_fit(real, reach, keep)
    d = np.linalg.norm(real - p[:2], axis=1)
    miss = d - (reach + p[2])
    print(f"\n=== {name}: the axis at grid ({p[0]:+.2f}, {p[1]:+.2f}), {math.hypot(*p[:2]):.2f} mm from the grid's "
          f"origin; reach = raw + {p[2]:.2f}; reach misses {np.sqrt(np.mean(miss[keep] ** 2)):.2f} rms")
    print("   point                                  fw bearing  reach miss | about the origin: reach miss  turn, fw - real"
          " | about the fitted axis: fw - real")
    origin_miss = np.linalg.norm(real, axis=1) - np.hypot(formula[:, 0], formula[:, 1])
    for g, b, m, om in zip(GRID, bearing, miss, origin_miss):
        about_origin = math.degrees(math.atan2(g[0][0], -g[0][1]))
        rel = np.array(g[0]) - p[:2]
        about_axis = math.degrees(math.atan2(rel[0], -rel[1]))
        print(f"   {g[2]:38s} {b:+7.2f}   {m:+6.2f}     | {-om:+6.2f}                  {b - about_origin:+6.2f}"
              f"          | {b - about_axis:+6.2f}")
