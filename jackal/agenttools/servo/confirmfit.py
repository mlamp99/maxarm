#!/usr/bin/env python3
"""The library's landing after the 17 Nov change (`landings.py confirm`) against the library before it (the study's
lib passes), by height: at the study's places and at places between them. No arm.

    PYTHONPATH=src ../.venv/bin/python3 agenttools/servo/confirmfit.py [confirm-....jsonl]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path("agenttools/servo/out")
study = [json.loads(x) for x in sorted(OUT.glob("landings-*.jsonl"))[-1].read_text().splitlines() if x.strip()]
chosen = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(OUT.glob("confirm-*.jsonl"))[-1]
confirm = [json.loads(x) for x in chosen.read_text().splitlines() if x.strip()]
print(chosen.name)
places = {(r["bearing"], r["reach"]) for r in study}


def line(name, rs):
    if not rs:
        return
    side = np.array([r["off"]["side_mm"] for r in rs])
    z = np.array([r["off"]["z"] for r in rs])
    reach = np.array([r["off"]["reach"] for r in rs])
    nearest = np.array([abs(r["reads"]["final"][0] - round(r["exact"][0])) <= 0.25 for r in rs])
    rms = lambda v: math.sqrt(np.mean(v ** 2))
    turned = [any(again for _, again in r.get("turns", [])) for r in rs]
    secs = np.median([r["seconds"] for r in rs])
    print(f"   {name:34s} n{len(rs):3d} | side within 0.5 mm {np.mean(np.abs(side) <= 0.5):4.0%}, 1 mm "
          f"{np.mean(np.abs(side) <= 1.0):4.0%}; base on the nearest pulse {nearest.mean():4.0%} | rms side "
          f"{rms(side):.2f} z {rms(z):.2f} reach {rms(reach):.2f}; mean z {z.mean():+.2f} reach {reach.mean():+.2f}"
          f" | turned again {np.mean(turned):3.0%}, {secs:.1f} s")


for height in sorted({r["height"] for r in confirm}):
    print(f"desk {height:+.0f}")
    line("before (lib passes), study places", [r for r in study if r["height"] == height and r["way"] == "lib"])
    line("after, study places", [r for r in confirm if r["height"] == height
                                and (r["bearing"], r["reach"]) in places])
    line("after, between them", [r for r in confirm if r["height"] == height
                                 and (r["bearing"], r["reach"]) not in places])
    by = {}
    for r in confirm:
        if r["height"] == height:
            by.setdefault(r["reach"], []).append(r)
    print("      after, by reach: " + "  ".join(
        f"{k:.0f}: z {np.mean([r['off']['z'] for r in v]):+.2f} reach {np.mean([r['off']['reach'] for r in v]):+.2f}"
        for k, v in sorted(by.items())))
turns = [t for r in confirm for t in r.get("turns", [])]
again = [r for r in confirm if any(a for _, a in r.get("turns", []))]
print(f"\nturned again: {len(again)} of {len(confirm)}; the first turn's miss, then where the base ended: " +
      ", ".join(f"{r['turns'][0][0]:+.2f} -> {r['off']['base']:+.2f}" for r in again))
