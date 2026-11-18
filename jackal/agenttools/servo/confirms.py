#!/usr/bin/env python3
"""The three checks through `down_to()` (17 Nov evening, all with the arm warm), side by side, by height: the extras
as tables by reach and height; the 6 Nov table plus the mean miss by height; the 6 Nov table. No arm.

    PYTHONPATH=src ../.venv/bin/python3 agenttools/servo/confirms.py
"""
import json
import math
from pathlib import Path

import numpy as np

NAMES = ("tables by reach and height", "6 Nov table + mean by height", "6 Nov table (kept)")
files = sorted(Path("agenttools/servo/out").glob("confirm-*.jsonl"))[-3:]
rms = lambda v: math.sqrt(np.mean(np.square(v)))
for height in (41.0, 5.0):
    print(f"desk {height:+.0f}")
    for name, path in zip(NAMES, files):
        rs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        rs = [r for r in rs if r["height"] == height]
        side = np.array([r["off"]["side_mm"] for r in rs])
        z = np.array([r["off"]["z"] for r in rs])
        reach = np.array([r["off"]["reach"] for r in rs])
        nearest = np.mean([abs(r["final"][0] - round(r["exact"][0])) <= 0.25 for r in rs])
        print(f"   {name:30s} n{len(rs):3d} side within 0.5 / 1 mm {np.mean(np.abs(side) <= 0.5):4.0%} "
              f"{np.mean(np.abs(side) <= 1.0):4.0%}, nearest pulse {nearest:4.0%} | rms side {rms(side):.2f} z "
              f"{rms(z):.2f} reach {rms(reach):.2f} | mean z {z.mean():+.2f} reach {reach.mean():+.2f} | worst side "
              f"{np.abs(side).max():.2f} z {np.abs(z).max():.2f} reach {np.abs(reach).max():.2f}")
