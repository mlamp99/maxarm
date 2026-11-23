"""What the agent's tools do with the tiles, as each app does it: the /IOTCONNECT app through its own commands
(`app/app.py`), the calibration app through the arm's one thread (`calibration/chat/api.py`).

    desk = Desk(seen=work.letters_seen, detect=..., check=word_check(work, settings), place=...)

`detect` and `place` block until the arm is done -- `place` for minutes -- and raise, saying why, when they cannot;
the agent runs `place` on a thread of its own. `check` and `seen` move nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..motion import Refused
from ..settings import Settings
from ..tilework import TileWork
from ..word import WordRefused


@dataclass(frozen=True)
class Desk:
    seen: Callable[[], Optional[dict]]      # `TileWork.letters_seen()`: the letters by where they lie; None before any
    detect: Callable[[], str]               # find the tiles again; the answer
    check: Callable[[str], str]             # why the word cannot be set down from the last detection; "" if it can
    place: Callable[[str], str]             # set the word down; the answer


def word_check(work: TileWork, settings: Settings) -> Callable[[str], str]:
    """`Desk.check` for both apps: the word planned on the last detection, as PLACE would plan it, nothing moved."""
    def check(word: str) -> str:
        try:
            work.plan(word, settings.get("tile_offset_mm"), settings.get("reader"))
        except (Refused, WordRefused) as reason:
            return str(reason)
        return ""
    return check
