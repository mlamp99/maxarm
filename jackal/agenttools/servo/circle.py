#!/usr/bin/env python3
"""Where the arm's axis really is, from reaches alone, and the base's turn measured about it. No arm.

The axis cannot move, and at cube-top height the cup's reach from it is set by the shoulder and elbow alone: each
cube's distance from the axis is its raw reach plus one amount (or plus a straight line in the reach). How far the
base turned does not enter, so the base servo's uneven turn cannot spoil it. Fitted here: the axis (x, y) in the
camera's desk space and that amount; then each cube's real bearing about that axis against the firmware's.

Reads out/points.json (points.py).

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/servo/circle.py
"""
import json
import math
from pathlib import Path

import numpy as np

from applib.camera import store
from maxarm import raw

rows = json.loads(Path("agenttools/servo/out/points.json").read_text())
saved = store.load_camera(Path("config"))
nuts = np.array([saved.anchor["base"]["x"], saved.anchor["base"]["y"]])


def radial_fit(seen, reach, slope=False):
    """Axis (x, y), the amount added to the raw reach, and the slope per 100 mm: least squares on distances."""
    params = np.array([*nuts, 2.0] + ([0.0] if slope else []))
    for _ in range(50):
        d = np.linalg.norm(seen - params[:2], axis=1)
        model = reach + params[2] + (params[3] * (reach - 190.0) / 100.0 if slope else 0.0)
        res = d - model
        J = np.column_stack([-(seen - params[:2]) / d[:, None], -np.ones(len(d))]
                            + ([-(reach - 190.0) / 100.0] if slope else []))
        delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        params = params + delta
        if np.abs(delta).max() < 1e-9:
            break
    d = np.linalg.norm(seen - params[:2], axis=1)
    model = reach + params[2] + (params[3] * (reach - 190.0) / 100.0 if slope else 0.0)
    return params, d - model


reach = np.array([r["raw_reach"] for r in rows])
bearing = np.array([r["raw_bearing"] for r in rows])
formula_add = np.array([r["reach"] - r["raw_reach"] for r in rows])          # what raw.py adds at each point's top
print(f"raw.py adds {formula_add.mean():.2f} mm to the raw reach at these cube tops ({formula_add.min():.2f} to "
      f"{formula_add.max():.2f})")
for key, camera in (("seen3", "saved three-cube camera"), ("seenN", "camera refitted to all cubes")):
    seen = np.array([r[key] for r in rows])
    print(f"\n=== {camera}")
    for slope in (False, True):
        params, res = radial_fit(seen, reach, slope)
        left = []
        for i in range(len(rows)):
            keep = np.arange(len(rows)) != i
            p, _ = radial_fit(seen[keep], reach[keep], slope)
            d = np.linalg.norm(seen[i] - p[:2])
            left.append(d - (reach[i] + p[2] + (p[3] * (reach[i] - 190.0) / 100.0 if slope else 0.0)))
        text = f"reach = raw + {params[2]:.2f}" + (f" + {params[3]:.2f} per 100 mm past 190" if slope else "")
        print(f"  {text:42s} axis ({params[0]:+.2f}, {params[1]:+.2f}): {np.linalg.norm(params[:2] - nuts):.2f} mm "
              f"from the nuts' base; reach misses {np.sqrt(np.mean(res ** 2)):.2f} rms, worst {np.abs(res).max():.2f}; "
              f"left out {np.sqrt(np.mean(np.square(left))):.2f} rms, worst {np.abs(left).max():.2f}")
    params, res = radial_fit(seen, reach, False)
    print("  each point, by the firmware's bearing: reach miss after the circle (mm)")
    print("   " + "  ".join(f"#{r['id']} {b:+.0f}: {m:+.2f}" for r, b, m in sorted(zip(rows, bearing, res),
                                                                          key=lambda t: t[1])))

# --- the turn about the axis the reaches found ------------------------------------------------------------------
print("\n=== the base's turn about the axis the reaches found (camera refitted to all cubes): firmware bearing minus "
      "real, deg; the frame's own turn set so that the points within 3 deg of straight out average 0, as raw.py's "
      "TURN_DEG has it. Beside it: TURN_DEG's (minus: raw.py's 'true further round'), and the same about the nuts")
seen = np.array([r["seenN"] for r in rows])
params, _ = radial_fit(seen, reach, False)
tie_turn = saved.anchor["turn_deg"]
for name, axis in (("the reaches' axis", params[:2]), ("the nuts' base", nuts)):
    rel = seen - axis
    desk_bearing = np.degrees(np.arctan2(rel[:, 0], -rel[:, 1]))
    straight = np.abs(bearing) < 3.0
    frame = np.mean(desk_bearing[straight] - bearing[straight])
    wide = bearing - (desk_bearing - frame)
    table = np.array([-math.degrees(raw._further_round(math.radians(b))) for b in bearing])
    print(f"  about {name} (frame turned {frame:+.2f} deg); fw - real, and minus TURN_DEG's:")
    order = np.argsort(bearing)
    print("   " + "  ".join(f"#{rows[i]['id']} {bearing[i]:+.0f}: {wide[i]:+.2f} ({wide[i] - table[i]:+.2f})"
                         for i in order))
    off = wide - table
    print(f"   left of -25: mean {off[bearing < -25].mean():+.2f}; -25..+25: {off[np.abs(bearing) <= 25].mean():+.2f}; "
          f"right of +25: {off[bearing > 25].mean():+.2f}; rms {np.sqrt(np.mean(off ** 2)):.2f} deg")
