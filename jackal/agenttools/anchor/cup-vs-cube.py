#!/usr/bin/env python3
"""A snapshot with the cup over the red cube: the cube's top centre as the saved camera finds it, and marks every 5 mm
out from it along the arm's reach (away from the base) and round, at the top's height, to read the cup's miss by eye.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/anchor/cup-vs-cube.py snapshot.png out.jpg
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from applib.camera import store
from applib.camera.locate import find_red_cube
from applib.camera.model import project

fit = store.load_camera(Path("config"))
frame = cv2.imread(sys.argv[1])
cube = find_red_cube(frame, fit)
x, y = cube.x, cube.y
out = np.array([x, y]) / math.hypot(x, y)               # desk axis at 0: away from the base
round_ = np.array([-out[1], out[0]])
draw = frame.copy()
top = project(fit.desk_to_pixel, np.array([[x, y, 40.0]]))[0]
for step in range(-4, 5):
    for direction, colour in ((out, (0, 255, 255)), (round_, (255, 0, 255))):
        p = np.array([x, y]) + 5 * step * direction
        pixel = project(fit.desk_to_pixel, np.array([[p[0], p[1], 40.0]]))[0]
        cv2.circle(draw, (int(pixel[0]), int(pixel[1])), 6 if step % 2 else 9, colour, -1)
cv2.drawMarker(draw, (int(top[0]), int(top[1])), (255, 255, 255), cv2.MARKER_CROSS, 50, 4)
x0, y0 = int(top[0]) - 300, int(top[1]) - 300
crop = draw[max(y0, 0):y0 + 600, max(x0, 0):x0 + 600].copy()
cv2.rectangle(crop, (0, 545), (600, 600), (0, 0, 0), -1)
cv2.putText(crop, "white +: top centre (saved camera); dots every 5 mm at 40 mm high:", (8, 565),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(crop, "yellow = out (away from base), magenta = round", (8, 588), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1)
cv2.imwrite(sys.argv[2], crop)
print(f"red cube top centre ({x:.1f}, {y:.1f}) at pixel ({top[0]:.0f}, {top[1]:.0f})")
