"""The ink on a tile face: whether there is any, which way it is, and the dot that marks a letter's bottom right.

A letter is darker than its face -- a marker's, on a light tile -- or lighter: white print on a dark one. Ink is
measured each way, against the face round it, not against a level: each pixel's darkness under the brightest of
the face within `PATCH_MM` (a morphological closing), or its lightness over the darkest (an opening), as a share of
the lighter of the two (Weber contrast), so a shadow or light falling off across the face cancels. A pixel is ink
at half the face's own strongest contrast.

On the owner's tiles the marker's strongest contrast is 0.56 to 0.77 and the bare face's about 0 (STATUS-hyena).
A face whose strongest is under `MIN_INK_CONTRAST` holds no dark letter. White print stands out less the brighter
its face is lit, the camera's white being as light as it shows the print: on a brown tile 0.17 to 0.49, 0.25 and
under where the camera has lightened a dark desk, against 0.06 at most for the bare face. Its least is lower,
`MIN_LIGHT_CONTRAST`.

Measured so, most faces have ink both ways: the face a letter closes in or between its strokes -- inside an A,
between an E's arms -- stands out from those strokes as the letter does from the face. It is told from a letter
by its tone, which is the bare face's: a letter's ink is, at its middle pixel, at least faint ink against the
face where there is no ink either way as well. On the owner's tiles and the brown printed ones the letter's is
0.36 to 0.65 against that face, the face it closes in 0.03 at most. Where both ways pass -- a marker's dot or a
stain beside a printed letter, a glint beside a marker's -- the letter is the way with more ink.

The dot is a small, round, separate blob of ink in a corner of the letter -- off its middle both ways -- and that
corner is the letter's bottom right. A faint stroke breaks into pieces at half the strongest ink, and a piece can
look like a dot: so the dot must be no longer than twice its width, and still apart from the letter in the faint
ink too. Beside a light letter the dot may be dark as well, a marker's on a printed tile: it is looked for first,
on the face with the letter taken out of it, so that the face the letter closes in is not taken for ink.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

WINDOW_MM = 23.0            # inside the face's rounded, darker rim
PX_PER_MM = 8.0
# Wider than a marker's stroke, so a closing fills the stroke with the face's own lightness.
PATCH_MM = 5.0
MIN_INK_CONTRAST = 0.25
MIN_LIGHT_CONTRAST = 0.12
# The strongest ink: a high percentile, not the maximum, so that a speck does not set it.
STRONGEST_PERCENTILE = 99.8
# Where a letter must have some ink: the face's middle.
MIDDLE_MM = 17.0
# A dot: this much ink, no larger across, round enough; this far off the letter's middle each way, as a share of
# the letter's half width and half height.
DOT_AREA_MM2 = (0.3, 9.0)
DOT_SIZE_MM = 3.5
DOT_MIN_FILL = 0.45
DOT_MAX_ELONGATION = 2.0
DOT_OFF_MIDDLE = 0.2
# Faint ink, a quarter of the face's strongest: what joins a piece of a stroke back to its letter, and the least a
# letter's ink is against the bare face.
FAINT_SHARE = 0.25


@dataclass(frozen=True)
class Ink:
    is_blank: bool
    contrast: float                             # the face's strongest ink, against the face round it
    dot: Optional[Tuple[float, float]]          # the dot, quarter turn 0's window, mm right and down of the centre
    is_light: bool = False                      # the letter is lighter than the face, not darker


@dataclass(frozen=True)
class _Way:
    """The face's ink one way: darker than the face round it, or lighter."""

    is_light: bool
    contrast: np.ndarray                        # each pixel's, against the face round it
    bare: np.ndarray                            # the face with this ink off it: under a stroke, what is round it
    strongest: float

    @property
    def least(self) -> float:
        """The least a face's strongest ink is, this way, if it holds a letter."""
        return MIN_LIGHT_CONTRAST if self.is_light else MIN_INK_CONTRAST

    @property
    def ink(self) -> np.ndarray:
        return self.contrast >= max(self.strongest, self.least) / 2

    @property
    def faint(self) -> np.ndarray:
        return self.contrast >= FAINT_SHARE * self.strongest


def size_px() -> int:
    return round(WINDOW_MM * PX_PER_MM)


def measure(grey: np.ndarray) -> Ink:
    """`grey`: the face at quarter turn 0, `WINDOW_MM` square at `PX_PER_MM`."""
    image = grey.astype(np.float32)
    ways = [_measure_way(image, is_light) for is_light in (False, True)]
    size = image.shape[0]
    margin = round((WINDOW_MM - MIDDLE_MM) / 2 * PX_PER_MM)
    middle = (slice(margin, size - margin), slice(margin, size - margin))
    is_bare = ~(ways[0].faint | ways[1].faint)
    face_tone = float(np.median(image[is_bare] if is_bare.any() else image))
    letters = [way for way in ways if _is_letter(way, image[middle], way.ink[middle], face_tone)]
    if not letters:
        return Ink(True, max(way.strongest for way in ways), None)
    letter = max(letters, key=lambda way: int(way.ink[middle].sum()))     # the dark way first: it wins a tie
    dot = _dark_dot(np.where(letter.faint, letter.bare, image), letter.ink) if letter.is_light else None
    return Ink(False, letter.strongest, dot or _dot(letter.ink, letter.faint), letter.is_light)


def _measure_way(image: np.ndarray, is_light: bool) -> _Way:
    patch = _odd(PATCH_MM * PX_PER_MM)
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (patch, patch))
    if is_light:
        # The window is carried on past its edges first. Without that, the face between a dark dot and an edge near
        # it has nowhere to find its own tone as the darkest round it, only the dot's, and counts as light ink.
        reach = patch // 2
        wider = cv2.copyMakeBorder(image, reach, reach, reach, reach, cv2.BORDER_REPLICATE)
        bare = cv2.morphologyEx(wider, cv2.MORPH_OPEN, shape)[reach:-reach, reach:-reach]
        contrast = (image - bare) / np.maximum(image, 1.0)
    else:
        bare = cv2.morphologyEx(image, cv2.MORPH_CLOSE, shape)
        contrast = (bare - image) / np.maximum(bare, 1.0)
    return _Way(is_light, contrast, bare, float(np.percentile(contrast, STRONGEST_PERCENTILE)))


def _is_letter(way: _Way, middle: np.ndarray, ink: np.ndarray, face_tone: float) -> bool:
    """Whether `way`'s ink can be a letter: strong enough, some of it in the face's `middle`, and there -- its `ink`
    -- of another tone than the bare face: at its middle pixel, at least faint ink against `face_tone` as well."""
    if way.strongest < way.least or not ink.any():
        return False
    tone = float(np.median(middle[ink]))
    lighter, darker = (tone, face_tone) if way.is_light else (face_tone, tone)
    return (lighter - darker) / max(lighter, 1.0) >= FAINT_SHARE * way.strongest


def _dark_dot(bare: np.ndarray, letter: np.ndarray) -> Optional[Tuple[float, float]]:
    """A dark dot in a corner of the light `letter`. `bare`: the face with that letter taken out of it. The
    strongest ink is what the smallest dot's worth of pixels reaches: there may be nothing dark here but the dot."""
    dark = _measure_way(bare, False)
    smallest = max(1, round(DOT_AREA_MM2[0] * PX_PER_MM ** 2))
    strongest = float(np.partition(dark.contrast, -smallest, axis=None)[-smallest])
    if strongest < MIN_INK_CONTRAST:
        return None
    return _dot(dark.contrast >= strongest / 2, letter=letter)


def _dot(ink: np.ndarray, faint: Optional[np.ndarray] = None,
         letter: Optional[np.ndarray] = None) -> Optional[Tuple[float, float]]:
    """The one small round blob of `ink`, if it is in a corner of the letter. The letter is the rest of `ink`, and the
    blob must be apart from it in the `faint` ink too; or it is `letter`, ink of the other tone, which the blob's
    own tone tells it from -- the face between a dark dot and the tile's edge is light ink as well, and touches it."""
    count, labels, stats, centres = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    mm2 = PX_PER_MM ** 2
    size = ink.shape[0]
    dots = []
    for index in range(1, count):
        left, top, width, height, area = stats[index]
        touches_edge = left == 0 or top == 0 or left + width == size or top + height == size
        is_small = DOT_AREA_MM2[0] * mm2 <= area <= DOT_AREA_MM2[1] * mm2 \
            and max(width, height) <= DOT_SIZE_MM * PX_PER_MM
        is_round = area >= DOT_MIN_FILL * width * height \
            and max(width, height) <= DOT_MAX_ELONGATION * min(width, height)
        if not touches_edge and is_small and is_round:
            dots.append(index)
    if len(dots) != 1:
        return None
    if letter is None:
        letter = (labels > 0) & (labels != dots[0])
        _, faint_labels = cv2.connectedComponents(faint.astype(np.uint8), connectivity=8)
        if (letter & (faint_labels == faint_labels[labels == dots[0]][0])).any():
            return None
    if not letter.any():
        return None
    rows, columns = np.nonzero(letter)
    middle = np.array([(columns.min() + columns.max()) / 2, (rows.min() + rows.max()) / 2])
    half = np.array([max(columns.max() - columns.min(), 1) / 2, max(rows.max() - rows.min(), 1) / 2])
    off = (centres[dots[0]] - middle) / half
    if min(abs(off)) < DOT_OFF_MIDDLE:
        return None
    centre = (size - 1) / 2
    return (float((centres[dots[0]][0] - centre) / PX_PER_MM), float((centres[dots[0]][1] - centre) / PX_PER_MM))


def _odd(pixels: float) -> int:
    return int(pixels) // 2 * 2 + 1
