#!/usr/bin/env python3
"""Where CALIBRATE's recorded cubes project in a snapshot through the saved camera.json, and where each cube found in
the snapshot stands on the desk (its top's centre at 40 mm): a picture with the two marked.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/anchor/where-cubes.py snapshot.png out.jpg
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from applib.camera import cubes
from applib.camera.fit import CameraFit
from applib.camera.model import project

CONFIG = Path("config")
fit = CameraFit.from_json(json.loads((CONFIG / "camera.json").read_text()))
recorded = json.loads((CONFIG / "camera-points.json").read_text())["cubes"]
frame = cv2.imread(sys.argv[1])
matrix = fit.desk_to_pixel
draw = frame.copy()
colours = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 0, 0)}
print("recorded (desk, top at 40) and its pixel through camera.json:")
for name, point in recorded.items():
    pixel = project(matrix, np.array([[point["x"], point["y"], 40.0]]))[0]
    print(f"  {name}: ({point['x']:7.2f}, {point['y']:7.2f}) -> pixel ({pixel[0]:6.0f}, {pixel[1]:6.0f})")
    cv2.drawMarker(draw, tuple(int(v) for v in pixel), colours[name], cv2.MARKER_CROSS, 60, 6)
    cv2.putText(draw, f"recorded {name}", (int(pixel[0]) + 30, int(pixel[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                colours[name], 4)
# the pixel of desk height 40 -> desk x, y: the plane z = 40 is a homography
plane = matrix[:, [0, 1, 3]].copy()
plane[:, 2] += matrix[:, 2] * 40.0
inverse = np.linalg.inv(plane)
print("cubes found in the snapshot, the top face's middle on the desk at 40 mm:")
for blob in cubes.find_blobs(frame):
    if blob.area < 5000:
        continue
    top = blob.contour.reshape(-1, 2).astype(float)
    centre = top.mean(axis=0)
    desk = inverse @ np.array([centre[0], centre[1], 1.0])
    desk = desk[:2] / desk[2]
    print(f"  {'warm' if cubes.is_warm(blob) else 'cool'} blob, area {blob.area:.0f}: pixel ({centre[0]:6.0f}, "
          f"{centre[1]:6.0f}) -> desk about ({desk[0]:7.1f}, {desk[1]:7.1f})")
    cv2.circle(draw, tuple(int(v) for v in centre), 18, (255, 255, 255), -1)
    cv2.putText(draw, f"seen ({desk[0]:.0f}, {desk[1]:.0f})", (int(centre[0]) + 30, int(centre[1]) + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
cv2.putText(draw, "cross: where the arm recorded each cube (camera.json of 13 Nov)   white dot: a cube in this snapshot "
            "(blob middle, rough)", (20, 1440), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
cv2.imwrite(sys.argv[2], cv2.resize(draw, None, fx=0.5, fy=0.5))
