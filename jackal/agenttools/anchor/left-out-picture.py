#!/usr/bin/env python3
"""The Fine Tune page's rings and dots, drawn on CALIBRATE's frame: ring = the cube as the camera sees it, dot = the
arm's reading as the other points' tie and correction put it (the point left out). With a legend.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/anchor/left-out-picture.py out.jpg
"""
import sys
from pathlib import Path

import cv2

from applib.camera import store
from calibration.camera import anchor
from calibration.tests import refine as page

CONFIG = Path("config")
fit = store.load_camera(CONFIG)
points = store.load_refine_points(CONFIG)
frame = store.load_frame(CONFIG)
found = anchor.base_of(fit, frame)
rows = page.describe(fit, points, found)["points"]
h, w = frame.shape[:2]
draw = frame.copy()
for row, point in zip(rows, points):
    view = (int(row["view"][0] * w), int(row["view"][1] * h))
    cv2.circle(draw, view, 22, (255, 255, 255), 4)
    if row["put"]:
        put = (int(row["put"][0] * w), int(row["put"][1] * h))
        cv2.circle(draw, put, 9, (0, 0, 255), -1)
        cv2.line(draw, view, put, (0, 0, 255), 2)
    cv2.putText(draw, f"#{row['id']} {row['cube'] or ''} left out {row['off']['after']:.1f} mm", (view[0] + 40, view[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
cv2.rectangle(draw, (0, h - 60), (w, h), (0, 0, 0), -1)
cv2.putText(draw, "white ring: cube seen by the camera (edges)   red dot: its reading as the OTHER points' calibration "
            "puts it (page's dot); CALIBRATE 16 Nov 10:11 frame", (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
cv2.imwrite(sys.argv[1], cv2.resize(draw, None, fx=0.5, fy=0.5))
