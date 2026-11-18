#!/usr/bin/env python3
"""The landings of `landings.py` read: where each way settles against the target, by height, reach and bearing; how
it repeats; what aims would centre it. No arm.

    PYTHONPATH=src ../.venv/bin/python3 agenttools/servo/landfit.py [landings-....jsonl]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("agenttools/servo/out")
path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(OUT.glob("landings-*.jsonl"))[-1]
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
print(f"{path.name}: {len(rows)} landings")
KEYS = ("base", "side_mm", "s2", "s3", "z", "reach")


def stats(values):
    v = np.array(values, float)
    return f"{v.mean():+5.2f} sd {v.std():4.2f} [{v.min():+5.2f} {v.max():+5.2f}]"


def group(rs, key):
    out = defaultdict(list)
    for r in rs:
        out[key(r)].append(r)
    return dict(sorted(out.items()))


ways = group(rows, lambda r: (r["height"], r["way"], r["pass"]))
print("\n== each pass: off from the target (base, s2, s3 pulses; side, z, reach mm) ==")
for (height, way, index), rs in ways.items():
    side = np.abs([r["off"]["side_mm"] for r in rs])
    z = np.array([r["off"]["z"] for r in rs])
    print(f"desk {height:+.0f} {way:8s} pass {index}: {len(rs)} | base {stats([r['off']['base'] for r in rs])}"
          f" | side within 0.5 mm {np.mean(side <= 0.5):4.0%}, 1 mm {np.mean(side <= 1.0):4.0%}"
          f" | again {np.mean([r['library_lands_again'] for r in rs]):3.0%}")
    for key in ("s2", "s3", "z", "reach"):
        print(f"      {key:5s} {stats([r['off'][key] for r in rs])}")

print("\n== by reach: mean off (sd) ==")
for (height, way), rs in group(rows, lambda r: (r["height"], r["way"])).items():
    print(f"desk {height:+.0f} {way}")
    for reach, rr in group(rs, lambda r: r["reach"]).items():
        cells = "  ".join(f"{k} {np.mean([r['off'][k] for r in rr]):+5.2f} ({np.std([r['off'][k] for r in rr]):.2f})"
                          for k in ("base", "s2", "s3", "z", "reach"))
        print(f"   {reach:3.0f} n{len(rr):2d}  {cells}")

print("\n== by bearing: base off, mean (sd) ==")
for (height, way), rs in group(rows, lambda r: (r["height"], r["way"])).items():
    cells = "  ".join(f"{b:+.0f}: {np.mean([r['off']['base'] for r in rr]):+.2f} ({np.std([r['off']['base'] for r in rr]):.2f})"
                      for b, rr in group(rs, lambda r: r["bearing"]).items())
    print(f"desk {height:+.0f} {way:8s} {cells}")

# the swing before the approach: from which side and how far the base came to its approach point
print("\n== the base against the swing before (from the previous place to the approach point) ==")
for (height, way), rs in group(rows, lambda r: (r["height"], r["way"])).items():
    if way not in ("lib", "falling", "settle"):
        continue
    swing = np.array([r["steps"]["approach"][0] - r["from_base"] for r in rs if r["from_base"] is not None])
    off = np.array([r["off"]["base"] for r in rs if r["from_base"] is not None])
    same = np.sign(swing) == np.sign(np.array([r["steps"]["over"][0] - r["steps"]["approach"][0]
                                               for r in rs if r["from_base"] is not None]))
    print(f"desk {height:+.0f} {way:8s} swing the same way as the last turn: {stats(off[same]) if same.any() else '-'}"
          f" (n {same.sum()}); the other way: {stats(off[~same]) if (~same).any() else '-'} (n {(~same).sum()})")

# repeat: the two lib passes, place by place
print("\n== repeat: lib pass 0 against pass 4, the same place ==")
for height in sorted({r["height"] for r in rows}):
    a = {(r["bearing"], r["reach"]): r for r in rows if r["height"] == height and r["way"] == "lib" and r["pass"] == 0}
    b = {(r["bearing"], r["reach"]): r for r in rows if r["height"] == height and r["way"] == "lib" and r["pass"] == 4}
    both = sorted(set(a) & set(b))
    if not both:
        continue
    for key in KEYS:
        d = [b[p]["off"][key] - a[p]["off"][key] for p in both]
        print(f"desk {height:+.0f} {key:7s} n{len(both)}: pass 4 - pass 0 {stats(d)}; |diff| median {np.median(np.abs(d)):.2f}")

# the elbow way: what the last move alone did
els = [r for r in rows if r["way"] == "elbow"]
if els:
    print("\n== elbow way: the last move alone, the elbow's stop against its command; the shoulder held? ==")
    for height, rs in group(els, lambda r: r["height"]).items():
        for reach, rr in group(rs, lambda r: r["reach"]).items():
            short = [r["reads"]["final"][2] - r["steps"]["elbow"][2] for r in rr]
            pre2 = [r["reads"]["pre"][1] - r["steps"]["pre"][1] for r in rr]
            moved2 = [r["reads"]["final"][1] - r["reads"]["pre"][1] for r in rr]
            moved1 = [r["reads"]["final"][0] - r["reads"]["pre"][0] for r in rr]
            print(f"desk {height:+.0f} r{reach:3.0f}: elbow final - command {stats(short)} | shoulder at pre - command "
                  f"{stats(pre2)} | shoulder moved in the elbow's move {stats(moved2)} | base {stats(moved1)}")

# what each way could do with its aims set per band: each landing's miss less its band's mean (the band's other
# landings only), by height; side in mm at the cup, z and reach in mm
print("\n== each way with its aims set per reach band (band mean from the band's other landings), rms mm ==")
print(f"{'height way':18s} {'n':>4s} {'side':>6s} {'z':>6s} {'reach':>6s} | as it landed: {'side':>6s} {'z':>6s} {'reach':>6s}")
for (height, way), rs in group(rows, lambda r: (r["height"], r["way"])).items():
    res = {k: [] for k in ("side_mm", "z", "reach")}
    raw_ = {k: [r["off"][k] for r in rs] for k in res}
    for reach, rr in group(rs, lambda r: r["reach"]).items():
        if len(rr) < 3:
            continue
        for i, r in enumerate(rr):
            others = [o for j, o in enumerate(rr) if j != i]
            for k in res:
                res[k].append(r["off"][k] - np.mean([o["off"][k] for o in others]))
    rms = lambda v: math.sqrt(np.mean(np.square(v))) if len(v) else float("nan")
    print(f"desk {height:+.0f} {way:9s} {len(rs):4d} {rms(res['side_mm']):6.2f} {rms(res['z']):6.2f} "
          f"{rms(res['reach']):6.2f} | {'':13s} {rms(raw_['side_mm']):6.2f} {rms(raw_['z']):6.2f} {rms(raw_['reach']):6.2f}")

# the shoulder's and elbow's extras that would centre each reach band, from the ways that descend as the library
# does (lib, settle): the old extra less the band's mean miss
sys.path.insert(0, "src")
from maxarm import landing  # noqa: E402
print("\n== extras that centre the band (old extra less the mean miss), pulses: s2, s3 ==")
for height in sorted({r["height"] for r in rows}):
    for ways_used in (("lib",), ("settle",), ("lib", "settle")):
        rs = [r for r in rows if r["height"] == height and r["way"] in ways_used]
        if not rs:
            continue
        cells = []
        for reach, rr in group(rs, lambda r: r["reach"]).items():
            e2, e3 = landing.extra_pulses(reach)
            cells.append(f"{reach:.0f}: {e2 - np.mean([r['off']['s2'] for r in rr]):+.2f} "
                         f"{e3 - np.mean([r['off']['s3'] for r in rr]):+.2f}")
        print(f"desk {height:+.0f} {'+'.join(ways_used):11s} " + " | ".join(cells))

# the stop against the whole pulse each servo was given (the command is a whole pulse: round(exact + extra)), so the
# rounding at each reach does not hide in the band's mean; the descent's ways only
print("\n== where s2 and s3 stop against their whole command, mean (sd), pulses ==")
for height in sorted({r["height"] for r in rows}):
    for ways_used in (("lib", "settle", "falling"),):
        rs = [r for r in rows if r["height"] == height and r["way"] in ways_used]
        cells = []
        for reach, rr in group(rs, lambda r: r["reach"]).items():
            s2 = [r["reads"]["final"][1] - round(r["steps"]["down"][1]) for r in rr]
            s3 = [r["reads"]["final"][2] - round(r["steps"]["down"][2]) for r in rr]
            cells.append(f"{reach:.0f}: {np.mean(s2):+.2f} ({np.std(s2):.2f}) {np.mean(s3):+.2f} ({np.std(s3):.2f})")
        print(f"desk {height:+.0f} " + " | ".join(cells))
