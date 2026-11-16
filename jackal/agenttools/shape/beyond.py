#!/usr/bin/env python3
"""Beyond the points: the mesh holding its nearest edge (now) against a smooth shape in bearing and reach fitted to
the points, alone or with the mesh of what it leaves at each point on top (that faded to nothing beyond the points).
Judged on points left out where they lie beyond the others: one side, far, near, one at a time. Misses mm, x-y.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/shape/beyond.py
"""
import json, math
from pathlib import Path
import numpy as np
from maxarm.mesh import Mesh, Row, REACH_SCALE_MM

rows = json.loads(Path("agenttools/shape/out/rows.json").read_text())


def terms(model, r):
    b, R = math.radians(r["bearing"]), r["reach"]
    u = [math.sin(b), -math.cos(b)]; v = [math.cos(b), math.sin(b)]
    bn, rn = r["bearing"] / 60.0, (R - 190.0) / 70.0
    o, a = [], []
    def axis(): o.extend(u); a.extend(v)
    def reach(*fs): o.extend(fs); a.extend([0.0] * len(fs))
    def turn(*fs): o.extend([0.0] * len(fs)); a.extend(R * math.radians(f) for f in fs)
    if model == "offsets":           reach(1); turn(1)              # one reach offset, one turn offset
    elif model == "by bearing":      reach(1, bn); turn(1, bn)      # both straight in bearing
    elif model == "bearing+reach":   reach(1, bn, rn); turn(1, bn)
    elif model == "axis+turnscale":  axis(); reach(1); turn(1, bn)
    return np.array([o, a])


def fit(model, train, ridge=0.3):
    A = np.vstack([terms(model, r) for r in train]); y = np.concatenate([[r["out"], r["across"]] for r in train])
    n = A.shape[1]
    return np.linalg.lstsq(np.vstack([A, math.sqrt(ridge) * np.eye(n)]), np.concatenate([y, np.zeros(n)]), rcond=None)[0]


def plane(r): return np.array([math.radians(r["bearing"]) * REACH_SCALE_MM, r["reach"]])


def mesh_of(train, value):
    return Mesh([Row(t["bearing"], t["reach"], value(t)[0], math.degrees(value(t)[1] / t["reach"])) for t in train])


def gap(m, r):
    """How far beyond the mesh, in its plane, mm; 0 inside."""
    here = plane(r)
    from maxarm.mesh import _barycentric
    for c in m.triangles:
        if _barycentric(m.plane[list(c)], here).min() >= -1e-9:
            return 0.0
    best = 1e9
    for s, e in m.edges:
        a, b = m.plane[s], m.plane[e]
        k = float(np.clip((here - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0, 1))
        best = min(best, float(np.linalg.norm(a + k * (b - a) - here)))
    return best


def predict(how, train, r):
    """(out, across) mm at r from train."""
    if how == "mesh now":
        o, d = mesh_of(train, lambda t: (t["out"], t["across"])).value(r["bearing"], r["reach"])
        return o, r["reach"] * math.radians(d)
    model, rest = how.split(" | ") if " | " in how else (how, None)
    c = fit(model, train)
    p = terms(model, r) @ c
    if rest is None:
        return p
    left = mesh_of(train, lambda t: tuple(np.array([t["out"], t["across"]]) - terms(model, t) @ c))
    o, d = left.value(r["bearing"], r["reach"])
    fade = 1.0 if rest == "held" else max(0.0, 1 - gap(left, r) / float(rest))
    return p + fade * np.array([o, r["reach"] * math.radians(d)])


HOWS = ["mesh now", "offsets", "by bearing", "bearing+reach", "axis+turnscale",
        "by bearing | 40", "by bearing | 80", "axis+turnscale | 40", "axis+turnscale | 80", "by bearing | held"]
splits = {
    "left beyond -50 deg": lambda r: r["bearing"] <= -50,
    "right beyond +30 deg": lambda r: r["bearing"] >= 30,
    "far, beyond 230 mm": lambda r: r["reach"] >= 230,
    "near, inside 140 mm": lambda r: r["reach"] <= 140,
}
print("misses mm: median / worst  (each point)")
for name, test in splits.items():
    train = [r for r in rows if not test(r)]; tests = [r for r in rows if test(r)]
    print(f"== {name}: {len(train)} points predict #" + ", #".join(str(t['id']) for t in tests))
    for how in HOWS:
        m = [math.hypot(*(np.array([t["out"], t["across"]]) - predict(how, train, t))) for t in tests]
        print(f"   {how:22s} {np.median(m):4.2f} / {max(m):4.2f}   " + " ".join(f"{x:4.1f}" for x in m))
print("== one left out, only those beyond the others' mesh (gap mm)")
outside = [t for t in rows if gap(mesh_of([r for r in rows if r is not t], lambda s: (0, 0)), t) > 0]
print("   points: " + ", ".join(f"#{t['id']} ({gap(mesh_of([r for r in rows if r is not t], lambda s: (0, 0)), t):.0f})" for t in outside))
for how in HOWS:
    m = [math.hypot(*(np.array([t["out"], t["across"]]) - predict(how, [r for r in rows if r is not t], t))) for t in outside]
    print(f"   {how:22s} {np.median(m):4.2f} / {max(m):4.2f}   " + " ".join(f"{x:4.1f}" for x in m))
print("== one left out, all (inside: the mesh of what is left on top)")
for how in HOWS:
    m = [math.hypot(*(np.array([t["out"], t["across"]]) - predict(how, [r for r in rows if r is not t], t))) for t in rows]
    print(f"   {how:22s} {np.median(m):4.2f} / {max(m):4.2f}")
for model in ("by bearing", "axis+turnscale"):
    print(f"{model} on all 18:", ", ".join(f"{x:+.3f}" for x in fit(model, rows)))
print("== few points predicting the rest (in recording order)")
order = sorted(rows, key=lambda r: r["id"])
for n in (3, 4, 5, 6, 8, 10):
    train, tests = order[:n], order[n:]
    line = []
    for how in ("mesh now", "offsets | 80", "axis+turnscale | 80"):
        m = [math.hypot(*(np.array([t["out"], t["across"]]) - predict(how, train, t))) for t in tests]
        line.append(f"{how} {np.median(m):4.2f}/{max(m):4.2f}")
    print(f"   {n:2d} points: " + "   ".join(line))


def fit_shrunk(train, hold):
    """axis+turnscale with the offsets free and the base's shift and the turn's scale held to 0 by `hold`."""
    A = np.vstack([terms("axis+turnscale", r) for r in train]); y = np.concatenate([[r["out"], r["across"]] for r in train])
    pen = np.diag([hold, hold, 0.01, 0.01, hold])
    return np.linalg.lstsq(np.vstack([A, math.sqrt(1.0) * pen]), np.concatenate([y, np.zeros(5)]), rcond=None)[0]


def predict_shrunk(hold, train, r, fade=80.0):
    c = fit_shrunk(train, hold)
    p = terms("axis+turnscale", r) @ c
    left = mesh_of(train, lambda t: tuple(np.array([t["out"], t["across"]]) - terms("axis+turnscale", t) @ c))
    o, d = left.value(r["bearing"], r["reach"])
    return p + max(0.0, 1 - gap(left, r) / fade) * np.array([o, r["reach"] * math.radians(d)])


print("== held back: base shift and turn scale shrunk by hold (sqrt of ridge); median/worst")
cases = {"left beyond": splits["left beyond -50 deg"], "right beyond": splits["right beyond +30 deg"],
         "far": splits["far, beyond 230 mm"], "near": splits["near, inside 140 mm"]}
for hold in (0.0, 2.0, 4.0, 8.0):
    line = []
    for name, test in cases.items():
        train = [r for r in rows if not test(r)]; tests = [r for r in rows if test(r)]
        m = [math.hypot(*(np.array([t["out"], t["across"]]) - predict_shrunk(hold, train, t))) for t in tests]
        line.append(f"{name} {np.median(m):.2f}/{max(m):.2f}")
    m = [math.hypot(*(np.array([t["out"], t["across"]]) - predict_shrunk(hold, [r for r in rows if r is not t], t))) for t in rows]
    line.append(f"one out {np.median(m):.2f}/{max(m):.2f}")
    for n in (3, 4, 6, 10):
        m = [math.hypot(*(np.array([t["out"], t["across"]]) - predict_shrunk(hold, order[:n], t))) for t in order[n:]]
        line.append(f"{n}pts {np.median(m):.2f}/{max(m):.2f}")
    print(f"   hold {hold:3.1f}: " + "  ".join(line))
    print(f"      on all 18: " + ", ".join(f"{x:+.2f}" for x in fit_shrunk(rows, hold)) + "   on 3 cubes: "
          + ", ".join(f"{x:+.2f}" for x in fit_shrunk(order[:3], hold)))


def fit_rule(train, hold):
    """Below 5 points the two offsets alone; from 5 the base's shift and the turn's scale too, held by `hold`."""
    model = "axis+turnscale" if len(train) >= 5 else "offsets"
    A = np.vstack([terms(model, r) for r in train]); y = np.concatenate([[r["out"], r["across"]] for r in train])
    pen = np.diag([hold, hold, 0.0, 0.0, hold] if model == "axis+turnscale" else [0.0, 0.0])
    return model, np.linalg.lstsq(np.vstack([A, pen]), np.concatenate([y, np.zeros(len(pen))]), rcond=None)[0]


def predict_rule(hold, train, r, fade=80.0):
    model, c = fit_rule(train, hold)
    p = terms(model, r) @ c
    left = mesh_of(train, lambda t: tuple(np.array([t["out"], t["across"]]) - terms(model, t) @ c))
    o, d = left.value(r["bearing"], r["reach"])
    return p + max(0.0, 1 - gap(left, r) / fade) * np.array([o, r["reach"] * math.radians(d)])


print("== the rule (offsets below 5 points), and the mesh now; median/worst")
for hold in (None, 0.0, 0.55, 1.0):
    pred = (lambda tr, t: predict("mesh now", tr, t)) if hold is None else (lambda tr, t, h=hold: predict_rule(h, tr, t))
    line = []
    for name, test in cases.items():
        train = [r for r in rows if not test(r)]; tests = [r for r in rows if test(r)]
        m = [math.hypot(*(np.array([t["out"], t["across"]]) - pred(train, t))) for t in tests]
        line.append(f"{name} {np.median(m):.2f}/{max(m):.2f}")
    m = [math.hypot(*(np.array([t["out"], t["across"]]) - pred([r for r in rows if r is not t], t))) for t in rows]
    line.append(f"one out {np.median(m):.2f}/{max(m):.2f}")
    for n in (3, 4, 5, 6, 10):
        m = [math.hypot(*(np.array([t["out"], t["across"]]) - pred(order[:n], t))) for t in order[n:]]
        line.append(f"{n}pts {np.median(m):.2f}/{max(m):.2f}")
    print(f"   {'mesh now' if hold is None else f'hold {hold:.2f}':9s} " + "  ".join(line))
