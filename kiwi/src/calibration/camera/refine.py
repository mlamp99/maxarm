"""SAVE CALIBRATION: the arm's correction made again with every point, on the frame CALIBRATE laid.

    refined, tie, put = refine_and_leave_out(fit, points, found)   # points: RefinePoints; found: anchor.base_of()
    off = off_by(refined, point.sighting)    # how far its reading is from its cube as seen; how tall it looks

A point is a cube set down in place after place, centred under the cup, and seen:
the firmware's position over it, and the edges found round it in the snapshot
(`sighting.py`). The camera stays as CALIBRATE made it from the cubes' edges --
reality, not bent onto the arm (the owner, 13 Nov; `cubes_alone`) -- and so does the
arm's frame on it, its axis and turn (camera.json's `anchor`): SAVE does not move
them. What is left at each point is a row of the arm's correction, the mesh
(`anchor.py`): SAVE writes `arm-true.json`, which the library reads on the fly
(`maxarm/converter.py`), and camera.json with its report.

A point's miss without it is its reading through the mesh of the other points,
against its cube as the camera sees it: how far the arm would land from the cube,
had the point not been taken. A point its neighbours put further than
`anchor.DOES_NOT_FIT_MM` from its cube does not fit: it is left out of the mesh and
named. At the points in the mesh the correction is exact.

A point's height (Z) is the camera's alone: how far from 40 mm tall its cube
looks, its base on the desk. One far off while the others are not says its edges
were not all the cube's: a bad snapshot.

With no points, SAVE puts back the three cubes' own calibration: the arm laid on
them by their readings, and no correction.
"""

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from applib.camera.base import FoundBase
from applib.camera.fit import CameraFit
from applib.camera.model import CUBE_MM
from applib.camera.sighting import Sighting, in_frame, measure, on_desk, place, start

from . import anchor


class PointsDoNotFit(Exception):
    """The points do not lay the arm on the camera."""


@dataclass(frozen=True)
class Off:
    seen: np.ndarray                 # desk x, y of the cube's centre, as the calibration sees it
    across: float                    # mm from where the arm says the cup was, across the desk
    height: Optional[float]          # mm the cube's top is from 40: + taller; None for a point without edges


def off_by(fit: CameraFit, point: Sighting) -> Off:
    """Where the calibration sees the point's cube, a 40 mm one, how far that is from where the arm says the cup was,
    and how tall the cube looks."""
    matrix = in_frame(fit.true_to_pixel, fit.frame_size, point)
    centre, height = _seen(matrix, point), None
    if point.edges is not None:
        height = float(measure(matrix, point, start(matrix, point))[3] - CUBE_MM)
    seen = np.linalg.solve(fit.desk_to_true, centre)
    return Off(seen, float(math.dist(seen, (point.x, point.y))), height)


def refine(fit: CameraFit, points, found: Optional[FoundBase]) -> Tuple[CameraFit, Optional[anchor.Tie]]:
    """camera.json laid by the points, and the tie whose mesh is the correction; with no points, the three cubes'
    own calibration and no tie. Raises PointsDoNotFit if the points do not lay the arm."""
    refined, tie, _ = refine_and_leave_out(fit, points, found)
    return refined, tie


def refine_and_leave_out(fit: CameraFit, points, found: Optional[FoundBase]
                         ) -> Tuple[CameraFit, Optional[anchor.Tie], List[Optional[np.ndarray]]]:
    """`refine()`, and each point's reading through the mesh of the other points: true x, y as the refined
    calibration has them (the board's mirror in them), None with no others."""
    alone = fit.cube_fit()
    if not points:
        return alone, None, []
    tie_points = anchor.tie_points(alone, points)
    try:
        tie = (anchor.kept_tie(fit.anchor, found, tie_points) if fit.anchor
               else anchor.fit_tie(found, tie_points))
    except (ValueError, np.linalg.LinAlgError) as reason:
        raise PointsDoNotFit(f"the points do not lay the arm on the camera: {reason}") from None
    refined = anchor.anchored(alone, tie)
    put = [None if tie.left.get(point.id) is None else anchor.to_anchored(tie.left[point.id]) for point in points]
    offs = [off_by(refined, point.sighting) for point in points]
    across = [float(math.dist(others, refined.desk_to_true @ off.seen))
              for point, off, others in zip(points, offs, put) if others is not None and point.id not in tie.misfits]
    heights = [abs(off.height) for off in offs if off.height is not None]
    summary = {"points": len(points)}
    if across:
        summary["off_mm"] = [round(float(np.median(across)), 2), round(max(across), 2)]
    if heights:
        summary["height_mm"] = [round(float(np.median(heights)), 2), round(max(heights), 2)]
    if tie.misfits:
        summary["does_not_fit_mm"] = {str(key): value for key, value in tie.misfits.items()}
    return replace(refined, refined=summary), tie, put


def _seen(matrix: np.ndarray, point: Sighting) -> np.ndarray:
    """True x, y under the point's cube through a camera for its frame: the 40 mm cube on its edges, or the pixel."""
    if point.edges is None:
        return on_desk(matrix, point)
    return place(matrix, point, start(matrix, point)[:3])[:2]
