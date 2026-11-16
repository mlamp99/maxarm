#!/usr/bin/env python3
"""Shapes for the arm's error after the formula, fitted to the points on file (out/rows.json), each judged by points
it was not fitted to: one at a time, the three cubes alone predicting the rest, and the 11:22 SAVE's points (2-10)
predicting those recorded since (11-19). Against the mesh, judged the same way. Misses in mm, x-y.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/shape/fits.py
"""
import json, math
from pathlib import Path
import numpy as np
from maxarm.mesh import Mesh, Row

rows = json.loads(Path("agenttools/shape/out/rows.json").read_text())
ids = [r["id"] for r in rows]


def basis(model, r):
    """The model's terms for one point: two rows (out, across), mm."""
    b, R = math.radians(r["bearing"]), r["reach"]
    u = np.array([math.sin(b), -math.cos(b)]); v = np.array([math.cos(b), math.sin(b)])
    bn, rn = r["bearing"] / 60.0, (R - 190.0) / 70.0
    out_terms, across_terms = [], []
    def axis():                      # the arm's axis off the base's: dx, dy
        out_terms.extend([u[0], u[1]]); across_terms.extend([v[0], v[1]])
    def reach(*fs):                  # out terms only
        for f in fs: out_terms.append(f); across_terms.append(0.0)
    def turn(*fs):                   # round terms, degrees, as mm across at R
        for f in fs: out_terms.append(0.0); across_terms.append(R * math.radians(f))
    if model == "const":             reach(1); turn(1)
    elif model == "axis":            axis(); reach(1); turn(1)
    elif model == "axis+turnscale":  axis(); reach(1); turn(1, bn)
    elif model == "axis+scales":     axis(); reach(1, rn); turn(1, bn)
    elif model == "poly":            reach(1, bn, rn, bn * bn, bn * rn); turn(1, bn, bn * bn)
    elif model == "axis+poly":       axis(); reach(1, bn, rn, bn * bn, bn * rn); turn(1, bn, bn * bn)
    return np.array([out_terms, across_terms])


def fit(model, train, ridge):
    A = np.vstack([basis(model, r) for r in train]); y = np.concatenate([[r["out"], r["across"]] for r in train])
    n = A.shape[1]
    A2 = np.vstack([A, math.sqrt(ridge) * np.eye(n)]); y2 = np.concatenate([y, np.zeros(n)])
    return np.linalg.lstsq(A2, y2, rcond=None)[0]


def miss(model, coef, r):
    p = basis(model, r) @ coef
    return math.hypot(r["out"] - p[0], r["across"] - p[1]), r["out"] - p[0], r["across"] - p[1]


def mesh_miss(train, r):
    m = Mesh([Row(t["bearing"], t["reach"], t["out"], math.degrees(t["across"] / t["reach"])) for t in train])
    o, d = m.value(r["bearing"], r["reach"])
    return math.hypot(r["out"] - o, r["across"] - r["reach"] * math.radians(d)), r["out"] - o, r["across"] - r["reach"] * math.radians(d)


MODELS = ["const", "axis", "axis+turnscale", "axis+scales", "poly", "axis+poly"]
RIDGE = 0.3
judge = {
    "one left out": [([r for r in rows if r is not t], [t]) for t in rows],
    "3 cubes -> rest": [([r for r in rows if r["cube"]], [r for r in rows if not r["cube"]])],
    "SAVE 2-10 -> 11-19": [([r for r in rows if r["id"] <= 10], [r for r in rows if r["id"] > 10])],
}
print("misses, mm: median / 90% / worst (worst point)")
for name, splits in judge.items():
    print(f"== {name}")
    for model in ["formula"] + MODELS + ["mesh"]:
        out = []
        for train, test in splits:
            for t in test:
                if model == "formula":
                    out.append((math.hypot(t["out"], t["across"]), t["id"]))
                elif model == "mesh":
                    out.append((mesh_miss(train, t)[0], t["id"]))
                else:
                    if len(train) < 3 and model not in ("const",):
                        continue
                    out.append((miss(model, fit(model, train, RIDGE), t)[0], t["id"]))
        if not out:
            continue
        v = np.array([o[0] for o in out]); w = max(out)
        print(f"   {model:15s} {np.median(v):4.2f} / {np.percentile(v, 90):4.2f} / {w[0]:4.2f} (#{w[1]})")
for model in ("axis+turnscale", "axis+scales"):
    c = fit(model, rows, RIDGE)
    print(f"\n{model} on all 18: " + ", ".join(f"{x:+.3f}" for x in c))
    print("   each point's residual (out, across) mm:")
    print("   " + "  ".join(f"#{r['id']} {miss(model, c, r)[1]:+.1f},{miss(model, c, r)[2]:+.1f}" for r in sorted(rows, key=lambda r: r['bearing'])))
