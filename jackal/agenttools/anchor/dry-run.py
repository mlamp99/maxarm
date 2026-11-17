#!/usr/bin/env python3
"""CALIBRATE and SAVE as the app now does them, on a copy of config/ (the real one untouched): CALIBRATE's frame and
cubes' points, then SAVE with every refine point on file. Prints what each writes; the copy is left in
agenttools/anchor/out/config-dry/ to look at, its check images among it. The source is config/, or a copy of one
given (out/config-13nov: CALIBRATE and the points of 13 Nov).

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/anchor/dry-run.py [source]
"""
import json
import shutil
import sys
import time
from pathlib import Path

import cv2

from maxarm import Space, Spaces, raw

from applib.camera import store
from applib.camera.base import find_base
from applib.camera.fit import fit_camera
from calibration.camera import anchor, check
from calibration.tests import refine as page

DRY = Path("agenttools/anchor/out/config-dry")
shutil.rmtree(DRY, ignore_errors=True)
shutil.copytree(sys.argv[1] if len(sys.argv) > 1 else "config", DRY, ignore=shutil.ignore_patterns("*.pem", "iotcDeviceConfig.json", "*-previous*"))

# CALIBRATE, as camera/api.py's _calibrate() does it, from the frame it took then.
frame = store.load_frame(DRY)
cubes = store.load_points(DRY)
started = time.time()
spaces = Spaces(DRY)
fit = fit_camera(frame, {colour: spaces.convert(raw.formula_true(tuple(point.raw)), Space.ARM, Space.DESK)[:2]
                         for colour, point in cubes.items()})
cube_points = store.cube_refine_points(fit, cubes, "dry run")
found = find_base(frame, fit.desk_to_pixel)
tie = anchor.fit_tie(found, anchor.tie_points(fit, cube_points))
anchored = anchor.anchored(fit, tie)
store.save_camera(DRY, anchored)
store.save_correction(DRY, tie.mesh("CALIBRATE dry run, the 3 cubes"))
cv2.imwrite(str(DRY / "check-calibrate.jpg"), check.draw(frame, anchored, []))
print(f"CALIBRATE ({time.time() - started:.0f} s): {store.describe_fit(anchored)}")
print(f"  {found.describe()}")
print(f"  report: {json.dumps(tie.report)}")
print(f"  does not fit: {tie.misfits or 'none'}")

# SAVE with the points on file (the three cubes as CALIBRATE made them now, and points 4, 5 as recorded).
saved = store.load_camera(DRY)
points = cube_points + [point for point in store.load_refine_points(DRY) if point.cube is None]
described = page.describe(saved, points, anchor.base_of(saved, frame))
for row in described["points"]:
    print(f"  #{row['id']}: off now {row['off']['now']:.2f}, without it {row['off']['without']}, "
          f"Z now {row['height']['now']}, fits {row['fits']}")
print(f"  SAVE would: {described['fit'] and {key: described['fit'][key] for key in ('off_mm', 'does_not_fit', 'pattern', 'change_mm')}}"
      f"{' -- ' + described['problem'] if described['problem'] else ''}")
refined, tie = page.refined(saved, points, anchor.base_of(saved, frame))
store.save_camera(DRY, refined)
store.save_correction(DRY, tie.mesh("SAVE dry run"))
cv2.imwrite(str(DRY / "check-save.jpg"), check.draw(frame, refined, []))
print(f"SAVE: {store.describe_fit(refined)}")
print(f"  report: {json.dumps(tie.report)}")
