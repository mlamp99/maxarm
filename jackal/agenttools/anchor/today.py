#!/usr/bin/env python3
"""The anchored calibration on the points on file, no app change: the base by its edges in CALIBRATE's frame, the
tie (axis, turn) and its report, the correction rows, each point judged by the others; and, if given, a snapshot's
red cube against where the owner's grid has it.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/anchor/today.py [snapshot.png real_x real_y]
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from maxarm import raw, Space, Spaces
from applib.camera import store
from applib.camera.base import find_base
from applib.camera.fit import CameraFit
from applib.camera.locate import find_red_cube
from calibration.camera import anchor

CONFIG = Path("config")
fit = CameraFit.from_json(json.loads((CONFIG / "camera.json").read_text()))
alone = fit.cube_fit()
found = find_base(cv2.imread(str(CONFIG / "camera-frame.png")), alone.desk_to_pixel)
print(found.describe())
spaces = Spaces(CONFIG)
points = store.load_refine_points(CONFIG)
tps = []
for point in points:
    the_raw = getattr(point, 'raw', None) or raw.formula_raw(spaces.convert((point.x, point.y, point.z), Space.DESK, Space.ARM))
    tps.append(anchor.TiePoint(point.id, tuple(the_raw), tuple(anchor.seen_desk(alone, point.sighting))))
tie = anchor.fit_tie(found, tps)
print(json.dumps(tie.report, indent=1))
for row in tie.rows:
    print(f"  #{row.point}: bearing {row.bearing_deg:+7.2f} reach {row.reach_mm:6.1f}: "
          f"further out {row.further_out_mm:+5.2f} mm, further round {row.further_round_deg:+6.3f} deg")
anchored = anchor.anchored(alone, tie)
mesh = tie.mesh()
print("each point's reading through the formula and the mesh, against its cube through the anchored camera, mm:")
for point, tp in zip(points, tps):
    true = raw.formula_true(tp.raw)
    off = mesh.at(tp.raw, true)
    seen = tie.to_arm(tp.seen)
    print(f"  #{point.id}: formula ({true[0]:7.2f}, {true[1]:7.2f}), corrected ({true[0] + off[0]:7.2f}, "
          f"{true[1] + off[1]:7.2f}), seen ({seen[0]:7.2f}, {seen[1]:7.2f})")
if len(sys.argv) == 4:
    frame = cv2.imread(sys.argv[1])
    real = (float(sys.argv[2]), float(sys.argv[3]))
    cube = find_red_cube(frame, anchored)
    print(f"the snapshot's red cube through the anchored camera: ({cube.x:.2f}, {cube.y:.2f}), "
          f"{math.dist((cube.x, cube.y), real):.2f} mm from the grid's {real}; through CALIBRATE's "
          f"(the readings' lay): {find_red_cube(frame, alone).x:.2f}, {find_red_cube(frame, alone).y:.2f}")
