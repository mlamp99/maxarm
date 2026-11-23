#!/usr/bin/env python3
"""Requests put to the language model on the Ara240 through the apps' own `AgentService`, with nothing moved: the
desk is a fixed set of letters, the arm and the camera are not touched. On the board, the connector ready:

    venv/bin/python agenttools/agent-check.py                               # the requests below, in turn
    venv/bin/python agenttools/agent-check.py "a 5-letter word" "explain"   # these
    venv/bin/python agenttools/agent-check.py --letters DVQEENMAOTSTZWN go

One service for all of them, so each request is told what happened before it, as on the Chat page. A word "set
down" takes PLACE_S and leaves the letters as they were: the tiles put back. Each tool call is printed as the apps
print it.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from applib.agent import AgentService, AraConnector, Desk        # noqa: E402
from applib.agent.words import missing_letters                   # noqa: E402

REQUESTS = ["go", "7-letter word", "A large herbivore domestic animal", "Elaborate",
            "Locate tiles again and give me a 7-letter word", "Give me a word, but don't place tiles"]
PLACE_S = 3.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--letters", default="AEEHMORST", help="the letters on the tiles ready to pick up")
    parser.add_argument("requests", nargs="*", default=REQUESTS)
    args = parser.parse_args()
    letters = "".join(sorted(args.letters.upper()))

    def seen() -> dict:
        return {"taken": "12:00:00", "on_offer": letters, "dropoff": "", "unusable": ""}

    def check(word: str) -> str:
        lacking = missing_letters(word, letters)
        return f"not on the tiles ready to pick up: {' '.join(lacking)}" if lacking else ""

    def place(word: str) -> str:
        time.sleep(PLACE_S)
        return f"{word} set down (agent-check: nothing moved)"

    desk = Desk(seen=seen, detect=lambda: f"{len(letters)} tiles (agent-check: nothing moved)", check=check,
                place=place)
    source = AraConnector()
    agent = AgentService(source, desk)
    while source.status()["state"] != "ready":
        time.sleep(1)
    for request in args.requests:
        print(f"\n> {request}", flush=True)
        started = time.monotonic()
        print(f"{agent.ask(request)}   [{time.monotonic() - started:.1f} s]", flush=True)
        while agent.state()["placing"]:
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
