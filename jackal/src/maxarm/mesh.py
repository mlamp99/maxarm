#!/usr/bin/env python3
"""The arm's error measured at a few points, and between them by a mesh: the camera calibration's correction.

    correction = Mesh([Row(bearing, reach, further_out, further_round), ...])
    dx, dy, dz = correction.at(raw, formula_true)

A row is a point where the camera saw the cube under the cup: where the formula
(`raw.py`) put the cup -- its bearing, degrees, 0 straight out, + toward +x, and
its reach from the base axis, mm -- and how much further out (mm) and further round
(degrees) the camera saw it. Between the points both are straight across the
triangles the points make (Delaunay), in the plane of bearing and reach with a
degree counted as an arc at REACH_SCALE_MM out, so the triangles have about the
shape they have on the desk; beyond the mesh each holds the value at its nearest
edge. At a point the correction is that point's exactly. One point holds
everywhere; two, or points in a line, join straight along it.

Round is kept in degrees: the base's turn is off by an angle, the same at any
reach. Out is kept in mm. The rows are taken at a cube's top; the correction is the
same at every height, the formula carrying what changes with height. Nothing
changes z.
"""

import itertools
import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import Position

REACH_SCALE_MM = 200.0      # a degree of bearing counts as this far round in the mesh's plane: mid-reach
SAME_MM = 0.5               # rows closer than this in the plane are one, their values averaged
FLAT = 1e-6                 # a triangle thinner than this share of its longest side squared is no triangle


@dataclass(frozen=True)
class Row:
    bearing_deg: float
    reach_mm: float
    further_out_mm: float
    further_round_deg: float
    point: Optional[int] = None         # the calibration point it came from, for the record


class Mesh:
    def __init__(self, rows: Sequence[Row], made: str = "") -> None:
        self.rows = list(rows)
        self.made = made
        plane, values = _merged(self.rows)
        self.plane, self.values = plane, values
        self.triangles = _delaunay(plane)
        self.edges = _edges(plane, self.triangles)

    @classmethod
    def from_json(cls, stored: dict) -> "Mesh":
        return cls([Row(**row) for row in stored["rows"]], stored.get("made", ""))

    def to_json(self) -> dict:
        return {"kind": "mesh", "made": self.made, "rows": [asdict(row) for row in self.rows]}

    def to_page(self) -> dict:
        """What a page needs to correct as `at()` does: the plane's points and their values, the triangles, the
        edges, and how far round a degree counts."""
        return {"scale": REACH_SCALE_MM, "plane": self.plane.round(4).tolist(), "values": self.values.round(4).tolist(),
                "triangles": [list(map(int, corners)) for corners in self.triangles],
                "edges": [list(map(int, pair)) for pair in self.edges]}

    def describe(self) -> str:
        return f"a mesh of {len(self.rows)} point{'' if len(self.rows) == 1 else 's'}" + (
            f" ({self.made})" if self.made else "")

    def at(self, raw_position: Position, formula_true: Position) -> Tuple[float, float, float]:
        """How far the cup is from where the formula puts it: true mm."""
        x, y, _ = formula_true
        reach = math.hypot(x, y)
        if reach < 1.0 or not self.rows:
            return (0.0, 0.0, 0.0)
        bearing = math.atan2(x, -y)
        out, round_deg = self.value(math.degrees(bearing), reach)
        reach, bearing = reach + out, bearing + math.radians(round_deg)
        return (reach * math.sin(bearing) - x, -reach * math.cos(bearing) - y, 0.0)

    def value(self, bearing_deg: float, reach_mm: float) -> Tuple[float, float]:
        """Further out (mm) and further round (degrees) at a formula bearing and reach."""
        if len(self.values) == 1:
            return tuple(float(value) for value in self.values[0])
        here = np.array([math.radians(bearing_deg) * REACH_SCALE_MM, reach_mm])
        for corners in self.triangles:
            shares = _barycentric(self.plane[list(corners)], here)
            if shares.min() >= -1e-9:
                return tuple(float(value) for value in shares @ self.values[list(corners)])
        best = None
        for start, end in self.edges:                     # beyond the mesh: its nearest edge
            a, b = self.plane[start], self.plane[end]
            share = float(np.clip((here - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0.0, 1.0))
            gap = float(np.linalg.norm(a + share * (b - a) - here))
            if best is None or gap < best[0]:
                best = (gap, (1 - share) * self.values[start] + share * self.values[end])
        return tuple(float(value) for value in best[1])


def _merged(rows: Sequence[Row]) -> Tuple[np.ndarray, np.ndarray]:
    """The rows as points of the plane and their values, rows at one place made one."""
    plane, values, counts = [], [], []
    for row in rows:
        place = np.array([math.radians(row.bearing_deg) * REACH_SCALE_MM, row.reach_mm])
        value = np.array([row.further_out_mm, row.further_round_deg])
        for index, there in enumerate(plane):
            if np.linalg.norm(there - place) < SAME_MM:
                values[index] = (values[index] * counts[index] + value) / (counts[index] + 1)
                counts[index] += 1
                break
        else:
            plane.append(place)
            values.append(value)
            counts.append(1)
    return np.array(plane).reshape(-1, 2), np.array(values).reshape(-1, 2)


def _delaunay(plane: np.ndarray) -> List[Tuple[int, int, int]]:
    """Every triangle of the points with no other point inside its circle: few points, so tried one by one."""
    found = []
    for corners in itertools.combinations(range(len(plane)), 3):
        a, b, c = plane[list(corners)]
        twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        longest = max(np.sum((b - a) ** 2), np.sum((c - b) ** 2), np.sum((a - c) ** 2))
        if abs(twice_area) <= FLAT * longest:
            continue
        d = 2 * twice_area
        centre = np.array([
            (a @ a * (b[1] - c[1]) + b @ b * (c[1] - a[1]) + c @ c * (a[1] - b[1])) / d,
            (a @ a * (c[0] - b[0]) + b @ b * (a[0] - c[0]) + c @ c * (b[0] - a[0])) / d])
        radius2 = np.sum((a - centre) ** 2)
        others = np.delete(plane, list(corners), axis=0)
        if not len(others) or np.min(np.sum((others - centre) ** 2, axis=1)) >= radius2 * (1 - 1e-9):
            found.append(corners)
    return found


def _edges(plane: np.ndarray, triangles: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    """The mesh's sides; with no triangle, the points in a line joined in order along it."""
    if triangles:
        return sorted({tuple(sorted(pair)) for corners in triangles for pair in itertools.combinations(corners, 2)})
    if len(plane) < 2:
        return []
    direction = plane[np.argmax(np.linalg.norm(plane - plane[0], axis=1))] - plane[0]
    order = np.argsort((plane - plane[0]) @ direction)
    return list(zip(order[:-1].tolist(), order[1:].tolist()))


def _barycentric(corners: np.ndarray, here: np.ndarray) -> np.ndarray:
    a, b, c = corners
    matrix = np.column_stack((b - a, c - a))
    s, t = np.linalg.solve(matrix, here - a)
    return np.array([1 - s - t, s, t])
