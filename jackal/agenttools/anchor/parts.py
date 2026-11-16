#!/usr/bin/env python3
"""Each base part as the frame shows it: a crop round where the whole base puts it, the part drawn where the whole
base puts it (cyan) and where it was found alone (green, if found), and its edge-cost surface (dark: low).

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/anchor/parts.py [frame.png]
Writes agenttools/anchor/out/parts.jpg.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from applib.camera import base
from applib.camera.fit import CameraFit
from applib.camera.model import project

CONFIG, OUT = Path("config"), Path("agenttools/anchor/out")
frame = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else str(CONFIG / "camera-frame.png"))
matrix = CameraFit.from_json(json.loads((CONFIG / "camera.json").read_text())).cube_fit().desk_to_pixel
found = base.find_base(frame, matrix)
print(found.describe())
pose = (found.x, found.y, np.radians(found.turn_deg))
gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (base.BLUR_PX, base.BLUR_PX), 0)
edges = cv2.Canny(gray, *base.CANNY)  # the whole base's edges; parts use their own
tiles = []
for (name, corner, points), part in zip(base.part_shapes(), found.parts):
    placed = base.moved(points, pose)
    pix = project(matrix, placed)
    centre = pix.mean(axis=0)
    half = 90 if name.startswith("nut") else 200
    x0, y0 = int(centre[0] - half), int(centre[1] - half)
    crop = np.zeros((2 * half, 2 * half, 3), np.uint8)
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + 2 * half, frame.shape[1]), min(y0 + 2 * half, frame.shape[0])
    if xe > xs and ye > ys:
        crop[ys - y0:ye - y0, xs - x0:xe - x0] = frame[ys:ye, xs:xe]
        crop[ys - y0:ye - y0, xs - x0:xe - x0][edges[ys:ye, xs:xe] > 0] = (0, 0, 255)
    for p in pix:
        cv2.circle(crop, (int(p[0] - x0), int(p[1] - y0)), 2, (255, 255, 0), -1)
    if part.found:
        shift = (part.x - (placed[:, 0].mean()), part.y - (placed[:, 1].mean()))
        for p in project(matrix, placed + (*shift, 0)):
            cv2.circle(crop, (int(p[0] - x0), int(p[1] - y0)), 2, (0, 255, 0), -1)
    crop = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_NEAREST if half < 100 else cv2.INTER_AREA)
    cv2.putText(crop, f"{name} {'FOUND' if part.found else '-'} c{part.contrast:.2f}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(crop)
rows = [np.hstack(tiles[k:k + 4]) for k in range(0, 8, 4)]
OUT.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT / "parts.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("out/parts.jpg: red = Canny edges, cyan = the part where the whole base puts it, green = found alone")
