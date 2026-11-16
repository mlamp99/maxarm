#!/usr/bin/env python3
"""Today's cubes against their readings through the 13 Nov camera (the readings' lay of 13 Nov, `cubes_alone`), next
to 13 Nov's own points through it: is one of today's readings out of line? The camera moved 2-7 px since (~1-2 mm).

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/anchor/blue-check.py
"""
import json
import math
from pathlib import Path

import numpy as np

from applib.camera import store
from applib.camera.fit import CameraFit
from calibration.camera import anchor
from maxarm import raw

OLD = Path("agenttools/anchor/out/config-dry")
old = CameraFit.from_json(json.loads((OLD / "camera.json").read_text())).cube_fit()
for label, config in (("13 Nov", OLD), ("today", Path("config"))):
    print(label, "-- through 13 Nov's camera: reading (formula), seen, seen minus reading; reach and bearing about 0")
    for point in store.load_refine_points(config):
        reading = np.array(raw.formula_true(point.raw)[:2]) if point.raw else np.array((point.x, point.y))
        seen = anchor.seen_desk(old, point.sighting)
        d = seen - reading
        out = math.hypot(*seen) - math.hypot(*reading)
        rnd = math.degrees(math.atan2(seen[0], -seen[1]) - math.atan2(reading[0], -reading[1]))
        print(f"  #{point.id} {point.cube or '':5}: reading ({reading[0]:7.2f}, {reading[1]:7.2f})  seen ({seen[0]:7.2f}, "
              f"{seen[1]:7.2f})  d ({d[0]:+6.2f}, {d[1]:+6.2f}) = {math.hypot(*d):5.2f} mm; out {out:+6.2f}, round {rnd:+6.2f} deg")
