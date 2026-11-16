#!/usr/bin/env python3
"""Every point on file as the shape fits see it: the firmware's position, the formula's (bearing, reach), and how
much further out (mm) and further round (mm across) the camera saw the cube, about the saved frame's axis and turn
(camera.json's anchor). Writes out/rows.json.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/shape/rows.py
"""
import json, math
from pathlib import Path
from applib.camera import store
from calibration.camera import anchor
from maxarm import raw

C = Path("config")
fit = store.load_camera(C)
alone = fit.cube_fit()
points = store.load_refine_points(C)
tps = anchor.tie_points(alone, points)
tie = anchor.Tie(tuple(fit.anchor["axis"]), float(fit.anchor["turn_deg"]), None, [])
rows = []
for point, tp in zip(points, tps):
    q = tp.reading
    s = tie.to_arm(tp.seen)
    r, b = math.hypot(*q), math.atan2(q[0], -q[1])
    u = q / r; v = (-u[1], u[0])
    d = s - q
    rr = math.hypot(*tp.raw[:2]); rb = math.degrees(math.atan2(tp.raw[0], -tp.raw[1]))
    rows.append({"id": point.id, "cube": point.cube, "raw": list(tp.raw), "raw_bearing": rb, "raw_reach": rr,
                 "bearing": math.degrees(b), "reach": r, "out": float(d @ u), "across": float(d @ v),
                 "seen": list(map(float, s)), "reading": list(map(float, q))})
Path("agenttools/shape/out").mkdir(parents=True, exist_ok=True)
Path("agenttools/shape/out/rows.json").write_text(json.dumps(rows, indent=1))
for row in sorted(rows, key=lambda r: r["bearing"]):
    print(f"#{row['id']:2d} raw b {row['raw_bearing']:+6.1f} R {row['raw_reach']:5.1f} z {row['raw'][2]:5.1f} | formula b {row['bearing']:+6.1f} "
          f"R {row['reach']:5.1f} | camera: out {row['out']:+5.2f} across {row['across']:+5.2f} ({math.degrees(row['across']/row['reach']):+.2f} deg)")
