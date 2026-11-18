#!/usr/bin/env python3
"""Which turn formula for raw.py: each candidate put in raw.py's place for a moment and the calibration made again
from the saved points as SAVE makes it (CALIBRATE's frame kept), judged by
  - each point's miss through the mesh of the others (the calibration's own "without it", mm), and
  - today's hand-matched 90s (pulse 870 = -90, 116 = +90, 500 = 0, far landings) through formula and mesh: how far
    round from 90 deg off the 500 line, mm at 240 out.
A formula fitted to the cubes is also judged fitted without the point it predicts (refitted 21 times). No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/servo/applyturn.py
"""
import json
import math
from pathlib import Path

import numpy as np

from applib.camera import store
from calibration.camera import anchor, refine
from maxarm import geometry, raw
from maxarm.mesh import Mesh

C = Path("config")
fit = store.load_camera(C)
points = store.load_refine_points(C, fit)
found = anchor.base_of(fit, store.load_frame(C))
rows = {r["id"]: r for r in json.loads(Path("agenttools/servo/out/points.json").read_text())}
TABLE = raw.TURN_DEG
HANDS = {"500": (500.0, 239.0, 694.44), "870": (870.0, 239.5, 694.33), "116": (116.0, 241.83, 690.17)}
REAL = {"870": -90.0, "116": 90.0}


def table(rows_):
    return lambda f: raw._between(rows_, f)


def s_taper(c):
    return lambda f: -(c[0] * math.sin(math.radians(2 * f)) + c[1] * f / 90 + c[2] * (f / 90) ** 2)


def s_scale(c):
    return lambda f: -(c[0] * math.sin(math.radians(2 * f)) + c[1] * f / 90)


def with_turn(turn):
    original = raw._further_round
    raw._further_round = lambda bearing: math.radians(turn(math.degrees(bearing)))
    return original


def judged(turn):
    original = with_turn(turn)
    try:
        _, tie, _ = refine.refine_and_leave_out(fit, points, found)
        mesh = Mesh(tie.rows)
        bearings = {}
        for name, pulses in HANDS.items():
            rawp = geometry.forward(geometry.pulse_to_deg(pulses))
            true = raw.formula_true(rawp)
            dx, dy, _ = mesh.at(rawp, true)
            x, y = true[0] + dx, true[1] + dy
            bearings[name] = math.degrees(math.atan2(x, -y))
        nineties = {k: (bearings[k] - bearings["500"] - REAL[k]) * math.radians(1) * 240.0 for k in REAL}
        return tie, {int(k): v for k, v in tie.report["left_out_mm"].items()}, nineties
    finally:
        raw._further_round = original


# --- the cubes' own fit of S + taper (calturn.py's, with today's 90s), for the refits ---------------------------
ids = [p.id for p in points]
f_all = np.array([rows[i]["raw_bearing"] for i in ids])
F_all = np.array([math.hypot(*rows[i]["formula"]) for i in ids])
seen_all = np.array([rows[i]["seen3"] for i in ids])


def fit_s_taper(keep, with_hands=True, terms=3):
    def e_of(c, fv):
        fv = np.asarray(fv, float)
        cols = [np.sin(np.radians(2 * fv)), fv / 90, (fv / 90) ** 2][:terms]
        return sum(ci * b for ci, b in zip(c, cols))

    def misses(q):
        c, phi, axis = q[:terms], q[terms], q[terms + 1:terms + 3]
        b = np.radians(f_all[keep] - e_of(c, f_all[keep]) + phi)
        pred = axis + F_all[keep, None] * np.column_stack([np.sin(b), -np.cos(b)])
        out = [(pred - seen_all[keep]).ravel()]
        if with_hands:
            for p, real in ((870, -90.0), (116, 90.0)):
                fv = (500 - p) * 0.24
                out.append(np.array([(fv - e_of(c, fv) - real) * math.radians(1) * 240.0]))
        return np.concatenate(out)

    q = np.zeros(terms + 3)
    q[terms + 1:] = seen_all.mean(0) * 0 + np.array([2.4, 2.05])
    for _ in range(100):
        r0 = misses(q)
        J = np.column_stack([(misses(q + d) - r0) / 1e-6 for d in np.eye(len(q)) * 1e-6])
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        q = q + step
        if np.abs(step).max() < 1e-9:
            break
    c = list(q[:terms]) + [0.0] * (3 - terms)
    return c


def summary(name, left, nineties, extra=""):
    v = np.array(list(left.values()))
    print(f"{name:44s} without it: median {np.median(v):.2f}, rms {math.sqrt(np.mean(v ** 2)):.2f}, worst "
          f"{v.max():.2f} mm | -90 {nineties['870']:+.2f} mm, +90 {nineties['116']:+.2f} mm{extra}")


every = np.arange(len(ids))
c_joint = fit_s_taper(every)
c_cubes = fit_s_taper(every, with_hands=False)
nineties_rows = list(TABLE)
nineties_rows[0] = (-88.80, -1.20)            # pulse 870 is -90
nineties_rows[-1] = (92.16, -2.16)            # pulse 116 is +90
candidates = {
    "the turn table now (TURN_DEG)": table(TABLE),
    "the table, its 90s rows from today (870, 116)": table(tuple(nineties_rows)),
    "none: the firmware's bearing": lambda f: 0.0,
    f"S + taper, cubes + today's 90s {np.round(c_joint, 2)}": s_taper(c_joint),
    f"S + taper, cubes alone {np.round(c_cubes, 2)}": s_taper(c_cubes),
}
print(f"{len(points)} points; the turn at the 90s and 45s, deg (true minus firmware):")
for name, turn in candidates.items():
    at = " ".join(f"{f:+.0f}:{turn(f):+.2f}" for f in (-88.8, -45, -20, 20, 45, 70, 92.16))
    print(f"   {name:44s} {at}")
print()
results = {}
for name, turn in candidates.items():
    tie, left, nineties = judged(turn)
    results[name] = (left, nineties)
    summary(name, left, nineties, f"; pattern {tie.mesh().describe_pattern()}")

# refitted without the point predicted
print()
for label, hands in (("S + taper, cubes + 90s, refitted without it", True), ("S + taper, cubes alone, refitted without it", False)):
    left = {}
    for k, pid in enumerate(ids):
        c = fit_s_taper(every[every != k], with_hands=hands)
        _, lo, _ = judged(s_taper(c))
        left[pid] = lo[pid]
    v = np.array(list(left.values()))
    print(f"{label:44s} without it: median {np.median(v):.2f}, rms {math.sqrt(np.mean(v ** 2)):.2f}, worst {v.max():.2f}")
    results[label] = (left, None)

print("\nper point, mm without it:")
names = list(results)
print("id   " + " | ".join(f"{n[:18]:>18s}" for n in names))
for pid in ids:
    print(f"{pid:3d}  " + " | ".join(f"{results[n][0][pid]:18.2f}" for n in names) +
          f"   f {rows[pid]['raw_bearing']:+.0f} r {math.hypot(*rows[pid]['formula']):.0f}")
