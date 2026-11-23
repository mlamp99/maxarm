#!/usr/bin/env python3
"""One request put to the connector as the agent puts it -- its system prompt, its tools, the message a request
becomes -- several times, and the connector's raw reply printed: the text, the tool calls, the tokens. Nothing run.
For finding out how the model and the connector behave. On the board, the connector ready:

    venv/bin/python agenttools/agent-raw.py go                        # 3 times, the agent's sampling
    venv/bin/python agenttools/agent-raw.py -n 5 -t 0.1 "7-letter word"
    venv/bin/python agenttools/agent-raw.py --letters DVQEENMAOTSTZWN --repeat 1.1 go
    venv/bin/python agenttools/agent-raw.py --system /tmp/prompt.txt go     # a system prompt to try

--repeat asks for a repetition penalty, which connector 2.2 does not pass on to the Ara240 unless patched for it.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from applib.agent import AgentService, Desk, connector, prompt   # noqa: E402
from applib.agent.words import spaced                            # noqa: E402


class Unwatched:
    """A model source that is never started: the service only wants to give it its tools."""
    name, lock, tools = "raw", Lock(), []

    def start(self) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request")
    parser.add_argument("--letters", default="AEEHMORST")
    parser.add_argument("-n", type=int, default=3, help="how many times")
    parser.add_argument("-t", "--temperature", type=float, default=connector.TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=connector.TOP_K)
    parser.add_argument("--top-p", type=float, default=connector.TOP_P)
    parser.add_argument("--repeat", type=float, default=0.0, help="repeat_penalty; 0 leaves it out")
    parser.add_argument("--system", type=Path, help="a system prompt to try, from a file, in place of the agent's")
    parser.add_argument("--recap", default="", help="the line of what happened before, as the chat would give it")
    args = parser.parse_args()

    letters = "".join(sorted(args.letters.upper()))
    source = Unwatched()
    AgentService(source, Desk(seen=lambda: None, detect=str, check=str, place=str))
    message = prompt.compose(args.request, f"Tiles you can use ({len(letters)}): {spaced(letters)}", args.recap)
    body = {"model": connector.ARA_MODEL, "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
            "messages": [{"role": "system", "content": args.system.read_text() if args.system else prompt.SYSTEM_PROMPT},
                         {"role": "user", "content": message}],
            "tools": source.tools}
    if args.repeat:
        body["repeat_penalty"] = args.repeat
    print(message, flush=True)
    for _ in range(args.n):
        started = time.monotonic()
        request = urllib.request.Request(f"{connector.ARA_URL}/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=connector.REQUEST_TIMEOUT_S) as reply:
            answer = json.load(reply)
        choice = answer["choices"][0]
        calls = [(call["function"]["name"], call["function"]["arguments"])
                 for call in choice["message"].get("tool_calls") or []]
        print(f"--- {time.monotonic() - started:.1f} s, {answer['usage']['prompt_tokens']} read, "
              f"{answer['usage']['completion_tokens']} written, {choice.get('finish_reason')}", flush=True)
        print(f"text  {choice['message'].get('content')!r}\ncalls {calls}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
