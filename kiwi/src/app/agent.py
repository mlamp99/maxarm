"""`agent`: a sentence answered by Qwen2.5-7B-Instruct on the Ara240, with the app's commands as its tools.

The property-alarm demo's design (`work/refs/iotc-eiq-imx95-property-alarm`, `app/agent.py`), its first pass here:
the four tools and a placeholder prompt; what the model is asked to do with them comes next (PILOT-kiwi).

    agent = AgentService(app.on_command, app.describe_letters, app.check_word)
    app.set_agent(agent)
    agent.ask("which words can the tiles make?")      # tens of seconds; raises CommandError

**The model is in another process.** `connector/` runs NXP's eIQ AAF Connector on the board, its own venv, the
model held on the Ara240 behind an OpenAI-style endpoint on port 3000: its torch must not meet the image's
numpy, cv2 and tflite_runtime that this app runs on. Here, strands-agents and openai, nothing more. The connector
is optional: without it the app runs as before and `agent` says the model is not answering.

**The tools are the app's.** `detect_tiles` and `place_word` are the `detect` and `place` commands, run through
`app.on_command()` with their locks and refusals; a refusal goes back to the model as text, to say in its words.
`get_letters` and `check_word` move nothing. Checking a word is code, not the model: models this size cannot
count letters. Every tool call is printed with what came back: the model's last sentence says what it meant to
do, not what it did.

The model reads 4096 tokens at most, prompt and answer together, and every tool's description is in every prompt:
hence short descriptions, short results, and a new `Agent` for each sentence -- no history. Answers are not
streamed (`IS_STREAMING`): property-alarm's connector lost tool calls streaming a sentence that needed two, and
nothing here needs an answer before it is whole. A sentence takes tens of seconds, on a command thread, one at a
time (`_lock`): the Ara240 answers one at a time anyway.
"""

from __future__ import annotations

import logging
from threading import Lock, Thread
from time import perf_counter, sleep
from typing import Callable

from applib import commands
from applib.commands import Command, CommandError

logger = logging.getLogger(__name__)

try:
    from strands import Agent, tool
    from strands.models.openai import OpenAIModel
    IS_STRANDS_AVAILABLE = True
except ImportError:                     # the app runs without it: `agent` says it is off
    IS_STRANDS_AVAILABLE = False

ARA_URL = "http://127.0.0.1:3000/v1"    # connector/run.sh, on the board, beside the app
ARA_MODEL = "Qwen2.5-7B-Instruct"
MAX_TOKENS = 256                        # an answer's at most
TEMPERATURE = 0.7
IS_STREAMING = False

# Ready once the model has answered something: a throwaway question until it does, and again after a sentence fails.
PROBE_QUESTION = "Say ready."
PROBE_SYSTEM_PROMPT = "Answer with a single word."
PROBE_INTERVAL_S = 30.0

SYSTEM_PROMPT = (
    "You are the assistant of a robot arm demo on an NXP i.MX95 board. The arm picks up letter tiles from a desk "
    "and sets words down with them in the dropoff area. Use a tool whenever the answer depends on the tiles or the "
    "user asks you to do something; never guess which letters there are. Check a word with check_word before "
    "placing it. Call one tool at a time, waiting for its result, and answer only once nothing is left to do, in "
    "one or two short sentences."
)


class AgentService:
    def __init__(self, run_command: Callable[[Command], str], describe_letters: Callable[[], str],
                 check_word: Callable[[str], str]) -> None:
        self.run_command = run_command              # app.on_command: one command, its answer
        self.describe_letters = describe_letters    # app.describe_letters: the letters as last seen
        self.check_word = check_word                # app.check_word: whether a word can be set down, and why not
        self._model = OpenAIModel(client_args={"api_key": "unused", "base_url": ARA_URL}, model_id=ARA_MODEL,
                                  params={"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                                          "stream": IS_STREAMING})
        self._tools = self._build_tools()
        self._lock = Lock()
        self.is_ready = False
        Thread(target=self._watch_the_model, name="agent", daemon=True).start()

    def ask(self, sentence: str) -> str:
        """The model's answer to `sentence`, after whatever tools it calls; CommandError if it cannot answer."""
        if not self.is_ready:
            raise CommandError("The language model on the Ara240 is not answering yet: after connector/run.sh it takes "
                               "minutes to load.")
        with self._lock:
            started = perf_counter()
            try:
                # callback_handler=None: strands would print the answer as it comes, among the other threads' lines.
                agent = Agent(model=self._model, tools=self._tools, system_prompt=SYSTEM_PROMPT,
                              callback_handler=None)
                answer = str(agent(sentence)).strip()
            except Exception as error:
                logger.debug("the agent failed", exc_info=True)
                print(f"[agent] {type(error).__name__}: {error}", flush=True)
                self.is_ready = False
                raise CommandError(f"The language model on the Ara240 did not answer ({type(error).__name__}): is "
                                   f"the connector running? connector/run.sh") from error
            print(f"[agent] {perf_counter() - started:.1f} s: {sentence!r} -> {answer!r}", flush=True)
        return answer or "The language model gave no answer."

    def _watch_the_model(self) -> None:
        """Ask the throwaway question until the model answers it; again whenever a sentence has failed."""
        is_first_failure = True
        while True:
            if self.is_ready:
                sleep(PROBE_INTERVAL_S)
                continue
            try:
                with self._lock:
                    Agent(model=self._model, system_prompt=PROBE_SYSTEM_PROMPT, callback_handler=None)(PROBE_QUESTION)
            except Exception as error:
                if is_first_failure:
                    print(f"agent      : the language model on the Ara240 is not answering ({type(error).__name__}); "
                          f"tried again every {PROBE_INTERVAL_S:.0f} s", flush=True)
                    is_first_failure = False
                sleep(PROBE_INTERVAL_S)
                continue
            self.is_ready = True
            print("agent      : the language model on the Ara240 answers", flush=True)

    def _build_tools(self) -> list:
        """The tools. Their docstrings are what the model reads, in every prompt: one line each."""
        run, describe_letters, check = self._run, self.describe_letters, self.check_word

        @tool
        def get_letters() -> str:
            """Return the letters on the desk as last seen: ready to pick up, already set down, and not usable."""
            return _logged("get_letters", "", describe_letters)

        @tool
        def detect_tiles() -> str:
            """Look at the desk again: the arm moves aside and the camera reads the tiles. Returns the letters."""
            return _logged("detect_tiles", "", lambda: f"{run(commands.DETECT)} {describe_letters()}")

        @tool
        def check_word(word: str) -> str:
            """Check whether a word can be set down from the letters ready to pick up.

            Args:
                word: the word, letters A to Z.
            """
            return _logged("check_word", word, lambda: check(word))

        @tool
        def place_word(word: str) -> str:
            """Set a word down: the arm carries a tile for each letter to the dropoff area. Takes minutes.

            Args:
                word: the word, letters A to Z.
            """
            return _logged("place_word", word, lambda: run(commands.PLACE, word))

        return [get_letters, detect_tiles, check_word, place_word]

    def _run(self, verb: str, argument: str = "") -> str:
        """One of the app's commands, for the model: its answer, or why it was refused, which is an answer too."""
        try:
            return self.run_command(Command(verb, argument, f"agent: {verb} {argument}".strip()))
        except CommandError as error:
            return str(error)


def _logged(name: str, argument: str, call: Callable[[], str]) -> str:
    """`call()`'s result, the call printed before it -- a command takes time -- and what came back after it."""
    print(f"    [agent] {name}({argument!r})" if argument else f"    [agent] {name}()", flush=True)
    result = call()
    print(f"    [agent] {name} -> {result!r}", flush=True)
    return result
