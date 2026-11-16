#!/usr/bin/env python3
"""A calibration anchored on reality alone, tried on today's points (no app change): the camera from the cubes' edges
(camera.json's own, `cube_fit()`), the arm's axis where that camera finds the base by its edges (base-from-edges.py),
the arm turned about it to fit the points' bearings -- turn only, no shift, reach not trusted -- and what is left,
point by point, the arm's error: how much further out and further round each cube is than the arm read it. Those,
straight between neighbours by the arm's bearing and held beyond the ends, are a correction table, and each point
is checked against the table the others make.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/base-anchor.py
"""
import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np

from applib.camera import store
from applib.camera.fit import CameraFit
from applib.camera.sighting import in_frame
from camera.refine import _seen

CONFIG = Path("config")
sys.argv = ["base-from-edges"]
base = runpy.run_path("agenttools/fitcheck/base-from-edges.py", run_name="__main__")   # CALIBRATE's frame
dx, dy, base_turn = base["pose"]
fit = CameraFit.from_json(json.loads((CONFIG / "camera.json").read_text())).cube_fit()
points = store.load_refine_points(CONFIG)


def polar(x, y):
    return math.hypot(x, y), math.degrees(math.atan2(x, -y))


rows = []
for point in points:
    seen_true = _seen(in_frame(fit.true_to_pixel, fit.frame_size, point.sighting), point.sighting)
    seen = np.linalg.solve(fit.desk_to_true, seen_true) - (dx, dy)                  # desk mm round the found base
    c, s = math.cos(-base_turn), math.sin(-base_turn)
    seen = np.array((c * seen[0] - s * seen[1], s * seen[0] + c * seen[1]))
    rows.append((point.id, point.cube or "red", polar(point.x, point.y), polar(*seen)))

turn = float(np.mean([cam[1] - arm[1] for _, _, arm, cam in rows]))                 # the arm's turn about the base
print(f"base found by its edges ({dx:+.1f}, {dy:+.1f}) mm, {math.degrees(base_turn):+.2f} deg from camera.json's; "
      f"the arm turned {turn:+.2f} deg about it")
print(" #  cube   arm reach/bearing   camera reach/bearing   further out  further round (mm)")
errors = []
for pid, cube, arm, cam in rows:
    out, round_deg = cam[0] - arm[0], cam[1] - arm[1] - turn
    errors.append((arm[1], out, round_deg))
    print(f"{pid:2}  {cube:5}  {arm[0]:6.1f} {arm[1]:+7.2f}      {cam[0]:6.1f} {cam[1]:+7.2f}        "
          f"{out:+5.2f}      {math.radians(round_deg) * cam[0]:+5.2f}")


def table_at(bearing, table):
    table = sorted(table)
    if bearing <= table[0][0]:
        return table[0][1:]
    for (b0, o0, r0), (b1, o1, r1) in zip(table, table[1:]):
        if bearing <= b1:
            share = (bearing - b0) / (b1 - b0) if b1 > b0 else 0.0
            return o0 + (o1 - o0) * share, r0 + (r1 - r0) * share
    return table[-1][1:]


print("each point through the table the others make (mm off):")
for k, (pid, cube, arm, cam) in enumerate(rows):
    out, round_deg = table_at(arm[1], [e for j, e in enumerate(errors) if j != k])
    reach, bearing = arm[0] + out, arm[1] + turn + round_deg
    miss_out, miss_round = cam[0] - reach, math.radians(cam[1] - bearing) * cam[0]
    print(f"  #{pid} {cube:5}: {math.hypot(miss_out, miss_round):5.2f}  (out {miss_out:+.2f}, round {miss_round:+.2f})")
