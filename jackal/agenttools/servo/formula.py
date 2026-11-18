#!/usr/bin/env python3
"""raw.py's true x, y on today's calibration points, the owner's way: the reach the same at every turn, the turn the
only thing that can be off; the turn right straight out (and at +-90, no points there). No arm.

A. About today's calibration's axis and turn (camera.json's anchor): the reach anchored on the points straight out,
   then every point's reach and turn against the formula's.
B. The reach held the same at every turn (raw.py's true reach exactly, or plus one amount): the axis that makes it so,
   then the turn there against TURN_DEG, by bearing, in pulses.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/servo/formula.py
"""
import json
import math
from pathlib import Path

import numpy as np

from applib.camera import store
from calibration.camera import anchor
from maxarm import raw

PULSE_DEG = 0.24
STRAIGHT_DEG = 5.0             # the points taken as straight out
rows = json.loads(Path("agenttools/servo/out/points.json").read_text())
saved = store.load_camera(Path("config"))
tie = anchor.Tie(tuple(saved.anchor["axis"]), float(saved.anchor["turn_deg"]), None, [])
seen = np.array([tie.to_arm(r["seen3"]) for r in rows])           # today's calibration: each cube in the arm's x, y
form = np.array([r["formula"] for r in rows])                     # raw.py's true x, y of each reading
fw = np.array([r["raw_bearing"] for r in rows])
F = np.linalg.norm(form, axis=1)
fb = np.degrees(np.arctan2(form[:, 0], -form[:, 1]))
table = fb - fw                                                   # TURN_DEG at each point: true minus firmware
straight = np.abs(fw) < STRAIGHT_DEG
order = np.argsort(fw)


def about(axis):
    rel = seen - axis
    return np.linalg.norm(rel, axis=1), np.degrees(np.arctan2(rel[:, 0], -rel[:, 1]))


def turn_misses(bearing):
    """Seen minus formula bearing, deg, the frame turned so that the points straight out average 0."""
    miss = (bearing - fb + 180) % 360 - 180
    return miss - miss[straight].mean()


def bands(values):
    out = []
    for lo, hi in ((-80, -55), (-55, -35), (-35, -10), (-10, 10), (10, 30), (30, 50)):
        m = (fw >= lo) & (fw < hi)
        out.append(f"{lo:+d}..{hi:+d} {values[m].mean():+.2f}")
    return "; ".join(out)


print(f"{len(rows)} points, firmware bearing {fw.min():+.0f} to {fw.max():+.0f}, reach {F.min():.0f}-{F.max():.0f} mm; "
      f"straight out: " + ", ".join(f"#{rows[i]['id']} ({fw[i]:+.1f}, {F[i]:.0f} mm)" for i in np.where(straight)[0]))

# --- A ---------------------------------------------------------------------------------------------------------
V, B = about(np.zeros(2))
c0 = (V - F)[straight].mean()
reach_miss = V - F - c0
print(f"\n=== A. about today's calibration's axis {tuple(round(v, 2) for v in tie.axis)}: straight out the cubes are "
      f"{c0:+.2f} mm further out than raw.py's reach ({', '.join(f'{v:+.2f}' for v in (V - F)[straight])}); that "
      f"taken as the reach everywhere, each point's reach miss, mm, by bearing:")
print("   " + bands(reach_miss))
print("   " + "  ".join(f"#{rows[i]['id']} {fw[i]:+.0f}: {reach_miss[i]:+.2f}" for i in order))

# --- B ---------------------------------------------------------------------------------------------------------


def radial(add_free):
    p = np.zeros(3)
    for _ in range(60):
        d = np.linalg.norm(seen - p[:2], axis=1)
        res = d - (F + p[2])
        J = np.column_stack([-(seen - p[:2]) / d[:, None]] + ([-np.ones(len(d))] if add_free else []))
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        p[:J.shape[1]] += step
        if np.abs(step).max() < 1e-10:
            break
    return p


def left_out(add_free):
    out = []
    global seen, F
    keep_seen, keep_F = seen, F
    for i in range(len(rows)):
        mask = np.arange(len(rows)) != i
        seen, F = keep_seen[mask], keep_F[mask]
        p = radial(add_free)
        seen, F = keep_seen, keep_F
        out.append(np.linalg.norm(seen[i] - p[:2]) - (F[i] + p[2]))
    return np.array(out)


for add_free in (False, True):
    p = radial(add_free)
    V, B = about(p[:2])
    miss = V - F - p[2]
    t = np.radians(fw)
    twice = np.linalg.lstsq(np.column_stack([np.cos(2 * t), np.sin(2 * t), np.ones(len(t))]), miss, rcond=None)[0]
    lo = left_out(add_free)
    print(f"\n=== B. the reach the same at every turn: raw.py's true reach" + (f" {p[2]:+.2f} mm" if add_free else "")
          + f"; the axis at ({p[0]:+.2f}, {p[1]:+.2f}), {np.linalg.norm(p[:2]):.2f} mm from today's calibration's")
    print(f"   reach misses {np.sqrt(np.mean(miss ** 2)):.2f} rms, worst {np.abs(miss).max():.2f}; each left out "
          f"{np.sqrt(np.mean(lo ** 2)):.2f} rms; a twice-a-turn wave in them {math.hypot(*twice[:2]):.2f} mm")
    print("   reach miss by bearing: " + bands(miss))
    turn = turn_misses(B)
    print("   turn: the base further round than TURN_DEG says, deg (pulses), by firmware bearing [TURN_DEG there]:")
    print("   " + "  ".join(f"#{rows[i]['id']} {fw[i]:+.0f}: {turn[i]:+.2f} ({turn[i] / PULSE_DEG:+.1f}) "
                            f"[{table[i]:+.2f}]" for i in order))
    print("   by bearing, deg: " + bands(turn))
    for row_fw, row_true in ((-46.80, 1.80), (47.15, -2.15)):
        near = np.abs(fw - row_fw) < 3
        if near.any():
            print(f"   at the 13 Nov row {row_fw:+.2f} (true {row_true:+.2f} further round): today "
                  + ", ".join(f"#{rows[i]['id']} ({fw[i]:+.1f}) {table[i] + turn[i]:+.2f}" for i in np.where(near)[0]))

# --- C. near against far at one bearing --------------------------------------------------------------------------
print("\n=== C. about B's axis (raw.py's true reach exactly): the turn misses as a smooth curve by bearing (cubic), and "
      "with a sideways part that shrinks with reach (s mm across: s / reach rad) -- a turn table by bearing cannot "
      "make near and far points at one bearing need different turns")
p = radial(False)
V, B = about(p[:2])
turn = turn_misses(B)


def curve_fit(idx, sideways):
    x = fw[idx] / 60.0
    cols = [np.ones(len(x)), x, x ** 2, x ** 3] + ([np.degrees(1.0 / F[idx])] if sideways else [])
    return np.column_stack(cols)


for sideways in (False, True):
    A = curve_fit(np.arange(len(rows)), sideways)
    coef = np.linalg.lstsq(A, turn, rcond=None)[0]
    res = turn - A @ coef
    left = []
    for i in range(len(rows)):
        keep = np.arange(len(rows)) != i
        c = np.linalg.lstsq(curve_fit(np.where(keep)[0], sideways), turn[keep], rcond=None)[0]
        left.append(turn[i] - curve_fit(np.array([i]), sideways) @ c)
    left = np.concatenate(left)
    mm = np.radians(res) * F
    print(f"  curve{' + sideways' if sideways else ''}: turn misses {np.sqrt(np.mean(res ** 2)):.2f} deg rms "
          f"({np.sqrt(np.mean(mm ** 2)):.2f} mm across), each left out {np.sqrt(np.mean(left ** 2)):.2f} deg"
          + (f"; sideways s = {coef[-1]:+.2f} mm" if sideways else ""))
    print("   left over, deg, near (< 170 mm) against far (> 220 mm) points: near "
          f"{res[F < 170].mean():+.2f}, far {res[F > 220].mean():+.2f}")

# --- D. the base's own curve, the sideways part taken out ---------------------------------------------------------
A = curve_fit(np.arange(len(rows)), True)
coef, *_ = np.linalg.lstsq(A, turn, rcond=None)
res = turn - A @ coef
cov = np.linalg.inv(A.T @ A) * np.sum(res ** 2) / (len(rows) - A.shape[1])
s, s_sure = coef[-1], math.sqrt(cov[-1, -1])
print(f"\n=== D. sideways s = {s:+.2f} +- {s_sure:.2f} mm; the base's own curve (0 straight out): deg the base is "
      f"further round than TURN_DEG says (pulses); and the table it makes, against TURN_DEG")
at = np.array([-75.0, -60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0])
x = at / 60.0
curve = coef[0] + coef[1] * x + coef[2] * x ** 2 + coef[3] * x ** 3
curve -= coef[0]
now = np.array([math.degrees(raw._further_round(math.radians(b))) for b in at])
print("   firmware bearing  " + "  ".join(f"{b:+6.0f}" for b in at))
print("   base off, deg     " + "  ".join(f"{v:+6.2f}" for v in curve))
print("   base off, pulses  " + "  ".join(f"{v / PULSE_DEG:+6.1f}" for v in curve))
print("   TURN_DEG          " + "  ".join(f"{v:+6.2f}" for v in now))
print("   wanted            " + "  ".join(f"{v:+6.2f}" for v in now + curve))
print(f"   the sideways part at 120 / 190 / 250 mm out: " + " / ".join(f"{math.degrees(s / r):.2f}" for r in (120, 190, 250))
      + " deg")

# --- E. the owner's 13 Nov rows (TURN_DEG, by hand at their own reach) against today's points -----------------
print("\n=== E. TURN_DEG's hand rows against what today's points say at the same firmware bearing and reach (straight out "
      "lined up at the 162 mm stack); beyond -76 / +47 the curve is not today's points' -- only what each row needs")


def base_curve(b):
    x = b / 60.0
    return coef[1] * x + coef[2] * x ** 2 + coef[3] * x ** 3


for row_fw, row_true, row_reach in ((-89.76, -0.24, 160.0), (-46.80, 1.80, 170.0), (47.15, -2.15, 226.0),
                                    (53.04, -0.99, 203.0), (58.97, -0.98, 189.0), (64.80, -1.37, 179.0),
                                    (92.64, -2.64, 200.0)):
    side = math.degrees(s / row_reach) - math.degrees(s / 162.0)
    inside = fw.min() - 1 <= row_fw <= fw.max() + 1
    if inside:
        today = row_true + base_curve(row_fw) + side
        print(f"   {row_fw:+7.2f} ({row_reach:.0f} mm): the row {row_true:+.2f}, today's points {today:+.2f} "
              f"({(today - row_true) / PULSE_DEG:+.1f} pulses)")
    else:
        print(f"   {row_fw:+7.2f} ({row_reach:.0f} mm): the row {row_true:+.2f}; for it to hold, the base's own curve there "
              f"{-side:+.2f} deg ({-side / PULSE_DEG:+.1f} pulses) off TURN_DEG")

# --- F. the axis from the turns alone, against the one from the reaches alone (B) ------------------------------------
print("\n=== F. the axis from the points' turns alone (no reaches), against B's from the reaches alone; mm in today's "
      "calibration's frame")
reach_axis = radial(False)[:2]


def turns_axis(curve_terms, sideways):
    """The axis, the frame's turn and the asked terms that best explain each point's seen bearing."""
    def model(q):
        rel = seen - q[:2]
        miss = (np.degrees(np.arctan2(rel[:, 0], -rel[:, 1])) - fb + 180) % 360 - 180
        x = fw / 60.0
        cols = [np.ones(len(x))] + [x ** k for k in range(1, curve_terms + 1)] \
            + ([np.degrees(1.0 / np.linalg.norm(rel, axis=1))] if sideways else [])
        Am = np.column_stack(cols)
        c = np.linalg.lstsq(Am, miss, rcond=None)[0]
        return miss - Am @ c, c
    q = np.zeros(2)
    for _ in range(50):
        r0, _ = model(q)
        J = np.column_stack([(model(q + step)[0] - r0) / 1e-4 for step in (np.array([1e-4, 0]), np.array([0, 1e-4]))])
        dq = np.linalg.lstsq(J, -r0, rcond=None)[0]
        q = q + dq
        if np.abs(dq).max() < 1e-7:
            break
    r0, c = model(q)
    dof = len(rows) - 2 - (1 + curve_terms + (1 if sideways else 0))
    J = np.column_stack([(model(q + step)[0] - r0) / 1e-4 for step in (np.array([1e-4, 0]), np.array([0, 1e-4]))])
    sure = np.sqrt(np.diag(np.linalg.pinv(J.T @ J) * np.sum(r0 ** 2) / dof))
    return q, sure, c, np.sqrt(np.mean(r0 ** 2))


print(f"  reaches alone (B): ({reach_axis[0]:+.2f}, {reach_axis[1]:+.2f})")
for name, terms, side in (("TURN_DEG as it is (the frame's turn only)", 0, False),
                          ("a base curve, no sideways", 3, False),
                          ("a base curve and the sideways", 3, True)):
    q, sure, c, rms = turns_axis(terms, side)
    d = q - reach_axis
    extra = f"; sideways {c[-1]:+.2f} mm" if side else ""
    print(f"  turns alone, {name:42s} ({q[0]:+.2f} +- {sure[0]:.2f}, {q[1]:+.2f} +- {sure[1]:.2f}); from B "
          f"{d[0]:+.2f} right, {d[1]:+.2f} back; turns left {rms:.2f} deg rms{extra}")

# --- G. the reach seen about today's calibration's axis, against what its place predicts ------------------------
d = -reach_axis                                # today's calibration's axis from the reaches' point
u = np.column_stack([np.sin(np.radians(fb)), -np.cos(np.radians(fb))])
V0 = np.linalg.norm(seen, axis=1)
seen_miss = V0 - F - (V0 - F)[straight].mean()
shift = -(u @ d)
print(f"\n=== G. today's calibration's axis is {d[0]:+.2f} right, {d[1]:+.2f} back of the reaches' point; the reach seen "
      f"about it, against what that place alone predicts (sides against straight out)")
print(f"   straight out beyond raw.py's reach: seen {(V0 - F)[straight].mean():+.2f}, predicted {shift[straight].mean():+.2f}")
print("   seen:      " + bands(seen_miss))
print("   predicted: " + bands(shift - shift[straight].mean()))
