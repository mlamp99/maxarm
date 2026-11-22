#!/usr/bin/env python3
"""Sentences put to the language model on the Ara240 through the app's own `AgentService`, with nothing moved: the
tools answer from a fixed set of letters, the arm and the camera are not touched. On the board, connector/run.sh
running and the model loaded:

    venv/bin/python agenttools/agent-check.py                        # the sentences below
    venv/bin/python agenttools/agent-check.py "a 5-letter word"      # these

Each tool call is printed as the app prints it. `detect_tiles` and `place_word` say what they would have done.
"""

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app import agent                                   # noqa: E402

LETTERS = "AEEHMORST"
DROPOFF = "CAT"
SENTENCES = ["Which letters are on the desk?", "Can you make the word HORSE?", "Give me a 5-letter word and place it."]


def describe_letters() -> str:
    return (f"The tiles as found at 12:00:00. Ready to pick up: {LETTERS}. Already in the dropoff area, left to "
            f"right: {DROPOFF}. Not usable: none ('?' not read, '-' blank).")


def check_word(text: str) -> str:
    wanted = text.strip().upper()
    short = Counter(wanted) - Counter(LETTERS)
    if short:
        return f"{wanted} cannot be set down: not on the tiles ready to pick up: {' '.join(sorted(short))}."
    return f"{wanted} can be set down from the tiles ready to pick up."


def run_command(command) -> str:
    if command.verb == "detect":
        return f"{len(LETTERS) + len(DROPOFF)} tiles (agent-check: nothing moved)."
    return f"agent-check: {command.verb} {command.argument} not done, nothing moved."


def main() -> int:
    service = agent.AgentService(run_command, describe_letters, check_word)
    while not service.is_ready:
        time.sleep(1)
    for sentence in sys.argv[1:] or SENTENCES:
        print(f"\n> {sentence}", flush=True)
        started = time.monotonic()
        print(f"{service.ask(sentence)}   [{time.monotonic() - started:.1f} s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
