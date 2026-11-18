#!/usr/bin/env python3
"""The base servo's turn against reality, every hand-measured point together; which pattern -- physical or not --
predicts the turn pulses best. No arm, no camera (the calibration's cubes only as a check at the end).

Each point: the firmware's bearing f (its base pulse p = 500 - f / 0.24) and the real bearing r about the arm's
axis, on that day's paper or grid. E = f - r: how much wider (+ right) the firmware reads than real. Each sheet gets
its own small turn (nuisance), today's none (500 and 870 measured on its lines). Where a sheet has enough points its
axis is placed from the reaches (raw.py's reach as it is: the reach is the same at every turn).

    PYTHONPATH=src ../.venv/bin/python3 agenttools/servo/turnpattern.py
"""
import math

import numpy as np

from maxarm import raw

PULSE_DEG = 0.24


def fw_bearing(rx, ry):
    return math.degrees(math.atan2(rx, -ry))


def true_reach(rx, ry, rz):
    t = raw.formula_true((rx, ry, rz))
    return math.hypot(t[0], t[1])


def place_axis(real, raws):
    """The axis where every point sits at raw.py's true reach for its reading (2 numbers, least squares)."""
    R = np.array([true_reach(*r) for r in raws])
    a = np.zeros(2)
    for _ in range(60):
        d = np.linalg.norm(real - a, axis=1)
        J = -(real - a) / d[:, None]
        step = np.linalg.lstsq(J, -(d - R), rcond=None)[0]
        a = a + step
        if np.abs(step).max() < 1e-9:
            break
    return a, d - R


points = []      # (sheet, label, f, r, weight)


def add(sheet, label, real_xy, raw_xyz, axis=(0.0, 0.0), w=1.0):
    f = fw_bearing(raw_xyz[0], raw_xyz[1])
    r = math.degrees(math.atan2(real_xy[0] - axis[0], -(real_xy[1] - axis[1])))
    points.append((sheet, label, f, r, w))


# --- today, 17 Nov: the paper's centre line and 90 deg lines, measured on (pulses; far 240 mm) -----------------------
for p, r in ((870, -90.0), (500, 0.0), (116, 90.0)):
    points.append(("17 Nov paper", f"pulse {p}", (500 - p) * PULSE_DEG, r, 1.0))

# --- 13 Nov fixed grid, the cup centred on a cube by hand (STATUS-jackal §19) ---------------------------------------
G13 = [((-160.0, 0.0), (-158.39, -0.66, 87.85), "left 90"),
       ((-120.0, -120.0), (-122.53, -115.06, 88.74), "left 45"),
       ((0.0, -162.25), (0.0, -160.0, 89.0), "straight"),
       ((160.0, -160.0), (164.42, -152.52, 83.91), "right 45 (nudged, old)"),
       ((160.0, -124.8), (160.45, -120.73, 85.99), "right 52"),
       ((160.0, -100.0), (159.88, -96.16, 86.61), "right 58"),
       ((160.0, -100.0), (160.66, -95.64, 86.89), "right 58 again"),
       ((160.2, -80.25), (160.38, -76.15, 87.02), "right 63"),
       ((200.0, 0.0), (197.74, 9.12, 86.82), "right 90 (200)"),
       ((205.0, 0.0), (204.07, 9.41, 86.56), "right 90 (205)")]
axis13, miss13 = place_axis(np.array([g[0] for g in G13]), [g[1] for g in G13])
for real_xy, raw_xyz, label in G13:
    add("13 Nov grid", label, real_xy, raw_xyz, axis13)

# --- 10 Nov paper (whole-mm readings), and the left 90 desk check that evening ------------------------------------
add("10 Nov paper", "left far", (-125.0, -231.75), (-132.0, -226.0, 86.0))
add("10 Nov paper", "left near", (-130.0, -131.75), (-132.0, -127.0, 86.0))
add("10 Nov paper", "straight 182", (0.0, -181.75), (1.0, -180.0, 86.0))
points.append(("10 Nov paper", "left 90 desk", fw_bearing(-204.0, -2.0), -89.35, 1.0))

# --- 3 Nov grid paper, 25 desk points (GRID-PAPER-jackal.md): paper x, y; raw x, y, z --------------------------------
P3 = [((22.64, 132.41), (-81.45, -99.30, 49.49)), ((63.67, 130.76), (-41.37, -100.77, 50.20)),
      ((105.20, 131.41), (0.42, -100.41, 50.28)), ((148.10, 130.63), (42.97, -101.04, 49.56)),
      ((187.40, 132.88), (83.57, -98.03, 49.15)), ((186.72, 102.48), (84.69, -127.47, 46.96)),
      ((145.02, 102.96), (42.09, -129.53, 48.94)), ((104.81, 102.43), (0.55, -130.41, 49.03)),
      ((64.33, 102.89), (-41.17, -130.41, 49.06)), ((22.89, 103.41), (-82.20, -128.33, 48.22)),
      ((23.75, 72.83), (-82.72, -158.23, 46.89)), ((63.02, 73.25), (-41.67, -159.53, 47.03)),
      ((103.38, 71.70), (0.00, -161.02, 47.14)), ((145.20, 72.67), (43.15, -159.70, 47.24)),
      ((185.60, 71.91), (84.12, -157.67, 47.53)), ((185.83, 43.43), (85.35, -187.47, 44.81)),
      ((145.28, 43.64), (42.58, -188.65, 45.20)), ((105.00, 43.12), (0.79, -189.67, 46.14)),
      ((64.21, 43.43), (-39.42, -189.33, 45.20)), ((25.54, 44.16), (-81.16, -188.63, 45.68)),
      ((23.00, 14.55), (-83.17, -218.04, 44.16)), ((63.45, 14.29), (-40.98, -219.83, 45.32)),
      ((101.91, 13.57), (-0.92, -218.90, 44.24)), ((143.53, 14.15), (41.74, -218.79, 44.58)),
      ((183.51, 13.06), (84.09, -217.69, 44.16))]
paper3 = np.array([[x, y] for (x, y), _ in P3])
paper3 = np.column_stack([paper3[:, 0], paper3[:, 1] - 250.0])   # paper y toward the base: the arm's y, shifted
axis3, miss3 = place_axis(paper3, [r for _, r in P3])
for (xy, raw_xyz), pt in zip(P3, paper3):
    add("3 Nov paper", f"({raw_xyz[0]:.0f}, {raw_xyz[1]:.0f})", pt, raw_xyz, axis3)

SHEETS = ["17 Nov paper", "13 Nov grid", "10 Nov paper", "3 Nov paper"]
sheet = np.array([SHEETS.index(p[0]) for p in points])
f = np.array([p[2] for p in points])
r = np.array([p[3] for p in points])
E = f - r
print(f"axes from reaches: 13 Nov grid {np.round(axis13, 2)} (reach misses {np.sqrt(np.mean(miss13**2)):.2f} rms); "
      f"3 Nov paper {np.round(axis3, 2)} ({np.sqrt(np.mean(miss3**2)):.2f} rms)")

# --- models: E as a function of the firmware's bearing f (deg) -------------------------------------------------------


def basis(name, fv):
    fv = np.atleast_1d(np.asarray(fv, float))
    tv = np.radians(fv)
    return {"firmware as is (0.24 deg a pulse)": [],
            "one scale (a wrong deg a pulse)": [fv / 90],
            "tapered track (deg a pulse changing evenly along it)": [fv / 90, (fv / 90) ** 2],
            "S: middle of the track worn (A sin 2f)": [np.sin(2 * tv)],
            "S + one scale": [np.sin(2 * tv), fv / 90],
            "S + tapered": [np.sin(2 * tv), fv / 90, (fv / 90) ** 2],
            "S + eccentric (B sin f)": [np.sin(2 * tv), np.sin(tv)],
            "S + eccentric + even (C sin^2 f)": [np.sin(2 * tv), np.sin(tv), np.sin(tv) ** 2]}[name]


def fixed_e(name, fv):
    fv = np.atleast_1d(np.asarray(fv, float))
    tv = np.radians(fv)
    if name.startswith("gecko"):
        return 1.36 * np.sin(2 * tv) - 0.33 * (np.cos(2 * tv) - 1)
    if name.startswith("-2.25"):
        return 2.25 * np.sin(2 * tv)
    return np.array([-math.degrees(raw._further_round(math.radians(v))) for v in fv])


MODELS = ["firmware as is (0.24 deg a pulse)", "one scale (a wrong deg a pulse)",
          "tapered track (deg a pulse changing evenly along it)", "S: middle of the track worn (A sin 2f)",
          "S + one scale", "S + tapered", "S + eccentric (B sin f)", "S + eccentric + even (C sin^2 f)"]
FIXED = ["gecko's (-1.36 sin 2b + 0.33 (cos 2b - 1))", "-2.25 sin 2b (10-13 Nov)", "TURN_DEG table (now)"]


def design(name, rows):
    cols = [] if name in FIXED else [c[rows] for c in basis(name, f)]
    nuis = [(sheet[rows] == k).astype(float) for k in range(1, len(SHEETS))]   # today's sheet: no turn of its own
    return np.column_stack(cols + nuis)


def fit(name, rows):
    y = E[rows] - (fixed_e(name, f[rows]) if name in FIXED else 0.0)
    A = design(name, rows)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return coef, y - A @ coef


def model_e(name, coef, fv):
    if name in FIXED:
        return fixed_e(name, fv)
    cols = basis(name, fv)
    return sum(c * b for c, b in zip(coef[:len(cols)], cols)) if cols else np.zeros(np.size(fv))


def pulse_for(name, coef, target):
    fv = target
    for _ in range(50):
        fv = target + float(model_e(name, coef, fv)[0])
    return 500 - fv / PULSE_DEG


every = np.arange(len(points))
not_today = np.where(sheet != 0)[0]
print("\nE = firmware minus real (deg, + = reads wider to the right / narrower to the left). Fitted to every point "
      "(each sheet its own turn); then each point left out; then today's -90 / +90 pulses predicted from the older "
      "sheets alone")
print(f"{'model':52s} {'terms':>24s} {'rms':>5s} {'left out':>8s} | from older sheets: -90, +90 (measured 870, 116)")
for name in MODELS + FIXED:
    coef, res = fit(name, every)
    lo = []
    for i in every:
        c2, _ = fit(name, every[every != i])
        nuis = c2[len(basis(name, f)) if name not in FIXED else 0:]
        own = nuis[sheet[i] - 1] if sheet[i] else 0.0
        lo.append(E[i] - float(model_e(name, c2, f[i])[0]) - own)
    c3, _ = fit(name, not_today)
    nterms = 0 if name in FIXED else len(basis(name, f))
    terms = ", ".join(f"{v:+.2f}" for v in coef[:nterms]) or "-"
    print(f"{name:52s} {terms:>24s} {np.sqrt(np.mean(res ** 2)):5.2f} {np.sqrt(np.mean(np.square(lo))):8.2f} | "
          f"{pulse_for(name, c3, -90.0):6.1f}, {pulse_for(name, c3, 90.0):6.1f}")

# --- the points, against the best physical candidate ---------------------------------------------------------------
best = "S + eccentric + even (C sin^2 f)"
coef, res = fit(best, every)
off = {SHEETS[k]: (coef[len(basis(best, f)) + k - 1] if k else 0.0) for k in range(len(SHEETS))}
print(f"\nsheets' own turns (deg), fitted with '{best}': " + ", ".join(f"{k} {v:+.2f}" for k, v in off.items()))
print(f"{'sheet':14s} {'point':24s} {'pulse':>6s} {'fw':>7s} {'real':>7s} {'E':>6s} {'E, sheet turned':>15s} "
      f"{'left':>6s}")
for i in np.argsort(f):
    s, label = points[i][0], points[i][1]
    e_sheet = E[i] + off[s]
    print(f"{s:14s} {label:24s} {500 - f[i] / PULSE_DEG:6.1f} {f[i]:+7.2f} {r[i]:+7.2f} {E[i]:+6.2f} {e_sheet:+15.2f} "
          f"{res[i]:+6.2f}")

# --- degrees a pulse, region by region (each sheet turned as fitted) -----------------------------------------------
print("\nreal degrees a pulse between regions (all sheets, each turned as fitted): a line through the points in each")
real_corr = r - np.array([off[p[0]] for p in points])
pul = 500 - f / PULSE_DEG
for lo_, hi_ in ((-95, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 95)):
    m = (real_corr >= lo_) & (real_corr <= hi_)
    near0 = np.abs(real_corr) < 0.5
    use = m | (near0 & ((lo_ == -30) | (lo_ == 0)))
    if use.sum() >= 2:
        slope = np.polyfit(pul[use], real_corr[use], 1)[0]
        print(f"   real {lo_:+d} to {hi_:+d}: {abs(slope):.4f} deg a pulse ({use.sum()} points)")

# === second pass: each sheet turned by its own straight-out points only; models fitted with no sheet turns =========
print("\n\n=== each sheet turned by its own straight-out points (|fw| < 1 deg), nothing else free")
anchor = {}
for k, s in enumerate(SHEETS):
    m = (sheet == k) & (np.abs(f) < 1.0)
    anchor[s] = float(np.mean(E[m])) if m.any() and s != "17 Nov paper" else 0.0
print("   sheets' turns: " + ", ".join(f"{s} {v:+.2f} deg" for s, v in anchor.items()))
Ea = E - np.array([anchor[p[0]] for p in points])
ra = f - Ea


def fit2(name, rows):
    y = Ea[rows] - (fixed_e(name, f[rows]) if name in FIXED else 0.0)
    cols = [] if name in FIXED else [c[rows] for c in basis(name, f)]
    if not cols:
        return np.zeros(0), y
    A = np.column_stack(cols)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return coef, y - A @ coef


print(f"{'model':52s} {'terms':>24s} {'rms':>5s} {'pt out':>6s} | each sheet predicted from the others, rms: "
      + ", ".join(s[:6] for s in SHEETS) + " | today -90, +90 from the others")
for name in MODELS + FIXED:
    coef, res = fit2(name, every)
    lo = [Ea[i] - float(model_e(name, fit2(name, every[every != i])[0], f[i])[0]) for i in every]
    by_sheet = []
    for k in range(len(SHEETS)):
        rows_out = np.where(sheet == k)[0]
        c2, _ = fit2(name, np.where(sheet != k)[0])
        pred = model_e(name, c2, f[rows_out])
        by_sheet.append(np.sqrt(np.mean((Ea[rows_out] - pred) ** 2)))
    c3, _ = fit2(name, not_today)
    nterms = 0 if name in FIXED else len(basis(name, f))
    terms = ", ".join(f"{v:+.2f}" for v in coef[:nterms]) or "-"
    print(f"{name:52s} {terms:>24s} {np.sqrt(np.mean(res ** 2)):5.2f} {np.sqrt(np.mean(np.square(lo))):6.2f} | "
          + "  ".join(f"{v:5.2f}" for v in by_sheet)
          + f" | {pulse_for(name, c3, -90.0):6.1f}, {pulse_for(name, c3, 90.0):6.1f}")

print("\nreal degrees a pulse between neighbouring groups of points (sheets turned by their straight-out points)")
GROUPS = [("left 90", -95, -85), ("left 45", -47, -43), ("left 20-40 (3 Nov)", -40, -19), ("straight", -1, 1),
          ("right 10-25 (3 Nov)", 9, 25), ("right 25-40 (3 Nov)", 25, 41), ("right 45 (13 Nov, nudged)", 44, 47),
          ("right 52-64 (13 Nov)", 51, 65), ("right 90", 85, 95)]
gm = []
for name, lo_, hi_ in GROUPS:
    m = (ra >= lo_) & (ra <= hi_)
    gm.append((name, float(np.mean(500 - f[m] / PULSE_DEG)), float(np.mean(ra[m])), int(m.sum())))
for (n1, p1, r1, k1), (n2, p2, r2, k2) in zip(gm, gm[1:]):
    print(f"   {n1:26s} ({p1:5.1f}, {r1:+6.2f}) -> {n2:26s} ({p2:5.1f}, {r2:+6.2f}): {abs((r2 - r1) / (p2 - p1)):.4f} "
          f"deg a pulse over {abs(p2 - p1):5.1f} pulses")
