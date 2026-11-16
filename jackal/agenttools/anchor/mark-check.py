#!/usr/bin/env python3
"""CALIBRATE's check picture for a cube that does not fit, as camera/api.py draws it, no arm: (1) a refusal -- a
snapshot with readings that do not lay on the cubes; (2) saved, a cube left out -- a config copy's frame and cubes.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/anchor/mark-check.py
"""
from pathlib import Path

import cv2

from applib.camera import store
from applib.camera.base import find_base
from applib.camera.fit import CubesDoNotFit, fit_camera
from calibration.camera import anchor, check
from maxarm import Space, Spaces, raw

OUT = Path("agenttools/anchor/out")

frame = cv2.imread(str(OUT / "snap-1046.png"))
cubes = store.load_points(Path("config"))
try:
    fit_camera(frame, {colour: (point.x, point.y) for colour, point in cubes.items()})
except CubesDoNotFit as reason:
    worst, (mm, there) = reason.worst, reason.apart[reason.worst]
    image = check.mark(check.draw(frame, reason.fit, []), reason.fit, worst,
                       f"{worst.upper()}: {mm:.0f} mm off where the arm recorded it", there,
                       [f"CALIBRATE failed: the {worst} cube is {mm:.0f} mm off where the arm recorded it"])
    cv2.imwrite(str(OUT / "mark-refusal.jpg"), cv2.resize(image, None, fx=0.5, fy=0.5))
    print("refusal:", reason)

config = OUT / "config-1011"
frame = store.load_frame(config)
cubes = store.load_points(config)
spaces = Spaces(config)
fit = fit_camera(frame, {colour: spaces.convert(raw.formula_true(tuple(point.raw)), Space.ARM, Space.DESK)[:2]
                         for colour, point in cubes.items()})
cube_points = store.cube_refine_points(fit, cubes, "mark check")
tie = anchor.fit_tie(find_base(frame, fit.desk_to_pixel), anchor.tie_points(fit, cube_points))
anchored = anchor.anchored(fit, tie)
image = check.draw(frame, anchored, [])
colours = {point.id: point.cube for point in cube_points}
for point, miss in tie.misfits.items():
    image = check.mark(image, anchored, colours[point], f"{colours[point].upper()}: does not fit, {miss:.0f} mm, left out",
                       anchor.to_anchored(tie.left[point]))
cv2.imwrite(str(OUT / "mark-misfit.jpg"), cv2.resize(image, None, fx=0.5, fy=0.5))
print("misfits:", tie.misfits)
