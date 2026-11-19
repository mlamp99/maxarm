#!/usr/bin/env python3
"""The arm's error measured at a few points: one pattern over the whole desk, and between the points a mesh of what
the pattern leaves at each. The camera calibration's correction.

    correction = Mesh([Row(bearing, reach, further_out, further_round), ...])
    dx, dy, dz = correction.at(raw, formula_true)

A row is a point where the camera saw the cube under the cup: where the formula
(`raw.py`) put the cup -- its bearing, degrees, 0 straight out, + toward +x, and
its reach from the base axis, mm -- and how much further out (mm) and further round
(degrees) the camera saw it.

**The pattern** is what the whole desk shares, fitted to the points (least squares,
out and across in mm): the reach off by one amount and the turn by one angle; and,
from PATTERN_FROM_POINTS points on, the arm's axis off its place (further out on one
side than on the other) and the turn wider or narrower to the sides (degrees per
PER_DEG of bearing), these two held toward none by PATTERN_HOLD so that a few points
cannot swing them. Fewer points cannot tell them from their own scatter. The turn's
widening grows across the points' bearings and WIDENING_BEYOND_DEG past them, then
holds.

**The mesh** is what the pattern leaves at each point: straight across the
triangles the points make (Delaunay), in the plane of bearing and reach with a
degree counted as an arc at REACH_SCALE_MM out, so the triangles have about the
shape they have on the desk; beyond them its nearest edge's, fading to nothing over
FADE_MM: far from the points, the pattern alone. At a point the correction is that
point's exactly. One point: the pattern is it, everywhere.

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
PATTERN_FROM_POINTS = 5     # from this many points the pattern has the axis off its place and the turn widening too
PATTERN_HOLD = 0.3          # how hard those are held toward none, per mm (axis) and per degree (widening) squared
PER_DEG = 60.0              # the turn's widening is counted per this much bearing
WIDENING_BEYOND_DEG = 20.0  # the widening grows this far past the points' bearings, then holds
FADE_MM = 80.0              # beyond the mesh, what the pattern leaves fades to nothing over this far in its plane
PATTERN = ("out_mm", "round_deg", "axis_x_mm", "axis_y_mm", "widening_deg")   # the pattern's numbers, in order


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
        places = [(math.degrees(across / REACH_SCALE_MM), reach) for across, reach in plane]
        self.span = (min(b for b, _ in places) - WIDENING_BEYOND_DEG,
                     max(b for b, _ in places) + WIDENING_BEYOND_DEG) if places else (0.0, 0.0)
        self.pattern = _pattern(places, values, self.span)
        self.plane = plane
        self.left = np.array([value - _pattern_value(self.pattern, self.span, *place)
                              for place, value in zip(places, values)]).reshape(-1, 2)
        self.triangles = _delaunay(plane)
        self.edges = _edges(plane, self.triangles)

    @classmethod
    def from_json(cls, stored: dict) -> "Mesh":
        return cls([Row(**row) for row in stored["rows"]], stored.get("made", ""))

    def to_json(self) -> dict:
        return {"kind": "mesh", "made": self.made, "rows": [asdict(row) for row in self.rows]}

    def to_page(self) -> dict:
        """What a page needs to correct as `at()` does: the pattern's numbers (PATTERN), the bearings the widening
        grows across, the plane's points and what the pattern leaves at each, the triangles, the edges; how
        far round a degree counts, the widening's degrees, the fade."""
        return {"scale": REACH_SCALE_MM, "per": PER_DEG, "fade": FADE_MM, "pattern": self.pattern.round(5).tolist(),
                "span": [round(self.span[0], 4), round(self.span[1], 4)],
                "plane": self.plane.round(4).tolist(), "left": self.left.round(4).tolist(),
                "triangles": [list(map(int, corners)) for corners in self.triangles],
                "edges": [list(map(int, pair)) for pair in self.edges]}

    def describe(self) -> str:
        return f"the pattern and mesh of {len(self.rows)} point{'' if len(self.rows) == 1 else 's'}" + (
            f" ({self.made})" if self.made else "")

    def describe_pattern(self) -> str:
        """The pattern in words."""
        out, round_deg, axis_x, axis_y, widening = self.pattern
        text = f"reach {out:+.1f} mm, turn {round_deg:+.2f} deg"
        if len(self.left) >= PATTERN_FROM_POINTS:
            text += (f", the axis {math.hypot(axis_x, axis_y):.1f} mm off its place ({axis_x:+.1f}, {axis_y:+.1f}),"
                     f" the turn {widening:+.2f} deg per {PER_DEG:.0f} deg to the sides")
        return text

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
        if not len(self.left):
            return (0.0, 0.0)
        pattern = _pattern_value(self.pattern, self.span, bearing_deg, reach_mm)
        return tuple(float(value) for value in pattern + self._left_at(bearing_deg, reach_mm))

    def _left_at(self, bearing_deg: float, reach_mm: float) -> np.ndarray:
        """What the pattern leaves: straight across the triangles; beyond them the nearest edge's, fading out."""
        here = np.array([math.radians(bearing_deg) * REACH_SCALE_MM, reach_mm])
        for corners in self.triangles:
            shares = _barycentric(self.plane[list(corners)], here)
            if shares.min() >= -1e-9:
                return shares @ self.left[list(corners)]
        best = (float(np.linalg.norm(self.plane[0] - here)), self.left[0])     # one point: its own
        for start, end in self.edges:
            a, b = self.plane[start], self.plane[end]
            share = float(np.clip((here - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0.0, 1.0))
            gap = float(np.linalg.norm(a + share * (b - a) - here))
            if gap < best[0]:
                best = (gap, (1 - share) * self.left[start] + share * self.left[end])
        return max(0.0, 1.0 - best[0] / FADE_MM) * best[1]


def _terms(span: Tuple[float, float], bearing_deg: float, reach_mm: float) -> np.ndarray:
    """Further out and further across (mm) at a place, per one of each of the pattern's numbers (PATTERN); the
    widening by the bearing held within `span`."""
    bearing = math.radians(bearing_deg)
    arc = math.radians(reach_mm)                        # mm across per degree round
    widening = arc * min(max(bearing_deg, span[0]), span[1]) / PER_DEG
    return np.array([[1.0, 0.0, math.sin(bearing), -math.cos(bearing), 0.0],
                     [0.0, arc, math.cos(bearing), math.sin(bearing), widening]])


def _pattern(places: Sequence[Tuple[float, float]], values: np.ndarray, span: Tuple[float, float]) -> np.ndarray:
    """The pattern's numbers (PATTERN) fitted to the points; the axis and the widening none below
    PATTERN_FROM_POINTS."""
    numbers = np.zeros(len(PATTERN))
    if not len(places):
        return numbers
    used = len(PATTERN) if len(places) >= PATTERN_FROM_POINTS else 2
    terms = np.vstack([_terms(span, *place)[:, :used] for place in places])
    seen = np.concatenate([[out, math.radians(reach) * round_deg] for (_, reach), (out, round_deg) in zip(places, values)])
    if used > 2:
        hold = math.sqrt(PATTERN_HOLD) * np.diag([0.0, 0.0, 1.0, 1.0, 1.0])
        terms, seen = np.vstack([terms, hold]), np.concatenate([seen, np.zeros(len(PATTERN))])
    numbers[:used] = np.linalg.lstsq(terms, seen, rcond=None)[0]
    return numbers


def _pattern_value(numbers: np.ndarray, span: Tuple[float, float], bearing_deg: float, reach_mm: float
                   ) -> np.ndarray:
    """Further out (mm) and further round (degrees) by the pattern."""
    out, across = _terms(span, bearing_deg, reach_mm) @ numbers
    return np.array([out, across / math.radians(reach_mm)])


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
