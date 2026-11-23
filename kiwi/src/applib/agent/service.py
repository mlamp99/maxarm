"""`AgentService`: a request answered by the language model, the tiles its tools -- the /IOTCONNECT app's `agent`
command and the calibration app's Chat page, the same way.

    agent = AgentService(AraConnector(), desk, on_say=lambda text, is_ok: print(text))
    agent.ask("a 7-letter word")            # the answer, after tens of seconds; CommandError if it cannot be asked
    agent.submit("go", source="page")       # the same on a thread of its own: the page follows `state()`
    agent.stop()                            # the request being answered, cut short
    agent.forget()                          # a new visitor: the conversation starts again

The property-alarm demo's design (`app/agent.py` there): the model in another process (`connector.py`), strands-
agents asking it, a new strands `Agent` for every request, nothing streamed, every tool call printed -- the model's
last sentence says what it meant to do, not what it did. On top of it:

**The model guesses, code counts, the model judges.** `find_word` takes six guesses at once and finds which the tiles
hold (`words.py`), which the arm can set down (the snapshot's plan, `desk.check`), and which are words at all (the
judge, `judge.py`), the model being a better judge of a word than a maker of one (STATUS-kiwi §6). One model call
for many guesses, where one guess a call ran out of time (STATUS-kiwi §5). Of those that pass, the longest -- of the
request's length, if it asks for one, read from it in code -- and not set down before this visit, is set down.

**A word is set down while the model talks.** `find_word` starts the arm and returns: the model's last words -- the
word and what it means -- come while the arm works, minutes. When it is done, the chat and `on_say` are told, which
is the cloud's `answer`; the model is not asked again. Meanwhile another request may be put to it: nothing more can
be set down or looked at, but words can be checked against the tiles left.

**Each request starts afresh**, told the letters and what happened before in a line (`chat.py`), never its history:
the window is 4096 tokens, and every model call of a request reads the whole of it again.

**Bounded, not queued.** One request at a time: another, while one is answered, is refused, and so is a request
before the model is ready -- with how far it has got. At most MAX_STEPS tool calls a request, the rest refused with
"answer now"; cut short after REQUEST_LIMIT_S. A tool call the model writes out as text is made all the same
(`repair.py`).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, replace
from threading import Lock, Thread, Timer
from typing import Any, Callable, List, Optional, Set, Tuple

from ..commands import CommandError
from . import prompt
from .chat import NOT_PLACED, PLACED, PLACING, PROPOSED, Chat, Turn
from .connector import FAILED, LOADING, NOT_INSTALLED, READY, STOPPED, WAITING, ModelSource
from .desk import Desk
from .judge import WordJudge
from .repair import written_call
from .words import check_words, describe_fits, spaced, take_letters, wanted_length

logger = logging.getLogger(__name__)

try:
    from strands import Agent, tool
    from strands.hooks import BeforeModelCallEvent, HookProvider
    IS_STRANDS_AVAILABLE = True
except ImportError:                         # the apps run without it: no agent
    HookProvider = object
    IS_STRANDS_AVAILABLE = False

MAX_STEPS = 4                               # a look, two finds, and one to spare
REQUEST_LIMIT_S = 180.0
STEP_DOING = {"find_word": "checking the guesses against the tiles",
              "detect_tiles": "looking at the tiles: the arm moves aside, the camera looks"}


@dataclass
class _Request:
    text: str
    source: str
    turn: Optional[Turn]
    started: float = field(default_factory=time.monotonic)
    agent: Any = None                       # strands' Agent, while it runs: to cut it short
    doing: str = "reading the request"
    calls: int = 0                          # model calls so far
    steps: int = 0                          # tool calls so far
    checked: Set[str] = field(default_factory=set)
    fitting: List[str] = field(default_factory=list)   # the words checked that can be set down, in order
    placed: str = ""                        # the word it started setting down


class AgentService:
    def __init__(self, source: ModelSource, desk: Desk, on_say: Optional[Callable[[str, bool], None]] = None) -> None:
        self.source = source
        self.desk = desk
        # What is said once the arm is done, and whether it set the word down: the cloud's `answer`.
        self.on_say = on_say or (lambda text, is_ok: None)
        self.chat = Chat()
        self.judge = WordJudge(source)                  # whether a guess is a word
        self._lock = Lock()                             # one request at a time
        self._request: Optional[_Request] = None        # the one being answered
        self._placing = ""                              # the word the arm is setting down
        self.source.tools = [_openai_spec(each) for each in self._build_tools(_Request("", "probe", None))]
        self.source.start()

    # --- asking -----------------------------------------------------------------------------------------------

    def ask(self, text: str, source: str = "console") -> str:
        """The model's answer to `text`, after whatever tools it calls; CommandError if it cannot be asked now."""
        self._take(text)
        try:
            return self._answer(text, source)
        finally:
            self._lock.release()

    def submit(self, text: str, source: str = "page") -> None:
        """`ask()` on a thread of its own: the answer goes into the chat. CommandError, at once, as `ask()`."""
        self._take(text)

        def answer() -> None:
            try:
                self._answer(text, source)
            except CommandError:
                pass                                    # in the chat already
            except Exception as error:                  # a bug here must not pass unsaid
                logger.exception("the agent's request failed")
                self.chat.add("assistant", f"Something went wrong: {type(error).__name__}: {error}", is_failed=True)
            finally:
                self._lock.release()
        Thread(target=answer, name="agent", daemon=True).start()

    def stop(self) -> bool:
        """Cut the request being answered short, at the model's next safe point; whether there was one."""
        request = self._request
        if request is None or request.agent is None:
            return False
        request.doing = "stopping"
        request.agent.cancel()
        return True

    def forget(self) -> None:
        self.chat.forget()

    def state(self) -> dict:
        """For the Chat page: the model, the conversation, the request being answered and the word being set down."""
        request = self._request
        # Seconds, not the time it started: the page's clock need not be the board's.
        thinking = None if request is None else {"text": request.text, "doing": request.doing, "calls": request.calls,
                                                 "steps": request.steps, "seconds": int(time.monotonic() - request.started)}
        return {"model": self.source.status(), "messages": self.chat.messages(), "thinking": thinking,
                "placing": self._placing}

    def _take(self, text: str) -> None:
        if not text.strip():
            raise CommandError("Ask what? Like: a 7-letter word.")
        status = self.source.status()
        if status["state"] != READY:
            raise CommandError(describe_unready(status))
        if not self._lock.acquire(blocking=False):
            raise CommandError("Still answering the last request: ask again when it is done.")

    def _answer(self, text: str, source: str) -> str:
        turn = self.chat.begin_turn(text)
        self.chat.add("user", text, source=source)
        request = _Request(text, source, turn)
        message = prompt.compose(text, self._describe_desk(), self.chat.recap())
        print(f"[agent] {source}: {text!r}", flush=True)
        self._request = request
        try:
            with self.source.lock:
                request.agent = Agent(model=self.source.model(), tools=self._build_tools(request),
                                      system_prompt=prompt.SYSTEM_PROMPT, callback_handler=None,
                                      hooks=[_Progress(request)])
                limit = Timer(REQUEST_LIMIT_S, request.agent.cancel)
                limit.daemon = True                     # not to hold the app up as it stops
                limit.start()
                try:
                    result = request.agent(message)
                finally:
                    limit.cancel()
        except Exception as error:
            logger.debug("the agent failed", exc_info=True)
            print(f"[agent] {type(error).__name__}: {error}", flush=True)
            self.source.failed(error)
            said = f"The language model did not answer ({type(error).__name__}): {error}"
            self.chat.add("assistant", said, is_failed=True, seconds=round(time.monotonic() - request.started, 1))
            self._request = None
            raise CommandError(said) from error
        answer = self._finish(request, result)
        self._request = None                            # once the answer is in the chat: the page shows one or the other
        print(f"[agent] {answer!r}", flush=True)
        return answer

    def _finish(self, request: _Request, result: Any) -> str:
        """The answer as the user is given it, its figures in the chat, and the word it chose remembered."""
        answer = str(result).strip()
        if written_call(answer, self.source.tools):
            answer = ""                             # a tool call written out where none could be made: nothing to say
        is_cut = result.stop_reason == "cancelled"
        if not answer:
            answer = (f"{', '.join(request.fitting[:3])} would fit." if request.fitting
                      else "I could not find a word in time." if is_cut else "I have no answer to that.")
        answer = _sentence_case(answer, request.fitting)
        usage = result.metrics.accumulated_usage
        figures = {"seconds": round(time.monotonic() - request.started, 1), "calls": request.calls,
                   "tokens_in": usage.get("inputTokens", 0), "tokens_out": usage.get("outputTokens", 0),
                   "is_cut": is_cut}
        self.chat.add("assistant", answer, **figures)
        print(f"[agent] {figures['seconds']} s, {request.calls} model calls, {figures['tokens_in']} tokens read, "
              f"{figures['tokens_out']} written{', cut short' if is_cut else ''}", flush=True)
        return answer

    # --- the tools ----------------------------------------------------------------------------------------------

    def _build_tools(self, request: _Request) -> list:
        """The tools, one set per request. Their docstrings are what the model reads, in every call: short."""

        @tool
        def find_word(guesses: str, place: bool = True) -> str:
            """Check guessed words against the tiles and have the arm set down the longest real word that fits.

            Args:
                guesses: 6 different English words, longest first, separated by spaces.
                place: false if the user said not to set it down, or only asked whether a word fits.
            """
            # One string, not a list of words: STATUS-kiwi §6.
            argument = guesses + ("" if place else ", place=false")
            return self._step(request, "find_word", argument, lambda: self._find(request, guesses, place))

        @tool
        def detect_tiles() -> str:
            """Look at the tiles again: the arm moves aside and the camera reads every letter."""
            return self._step(request, "detect_tiles", "", lambda: self._detect())

        return [find_word, detect_tiles]

    def _step(self, request: _Request, name: str, argument: str, call: Callable[[], Tuple[str, dict]]) -> str:
        """One tool call: in the chat and on the console before it runs -- it takes time -- and with what came back."""
        request.steps += 1
        if request.steps > MAX_STEPS:
            if request.steps > MAX_STEPS + 2:
                request.agent.cancel()
            return "No more tool calls: answer the user now, with what you have."
        print(f"    [agent] {name}({argument!r})" if argument else f"    [agent] {name}()", flush=True)
        step = self.chat.add("step", "", tool=name, argument=argument, is_done=False)
        request.doing = STEP_DOING[name]
        started = time.monotonic()
        try:
            result, data = call()
        except Exception as error:              # refused or failed: an answer for the model too, to pass on
            why = (str(error) or repr(error)).rstrip(".")
            result, data = f"{name} could not be done: {why}. Tell the user so.", {"is_failed": True}
        self.chat.update(step, result, is_done=True, seconds=round(time.monotonic() - started, 1), **data)
        request.doing = "the model reads what came back"
        print(f"    [agent] {name} -> {result!r}", flush=True)
        return result

    def _find(self, request: _Request, guesses: str, place: bool) -> Tuple[str, dict]:
        """The guesses checked -- their letters, then where they would go, then whether each is a word -- and the
        longest that fits, of the request's length if it asks for one, set down."""
        tiles, why_none = self._letters_now()
        if tiles is None:
            return why_none, {}
        letters = wanted_length(request.text)
        fits = check_words(guesses, tiles, request.checked)
        if not self._placing:
            fits = [replace(fit, problem=self.desk.check(fit.word)) if not fit.missing else fit for fit in fits]
        fits = [replace(fit, is_word=self.judge.is_word(fit.word)) if not fit.missing and not fit.problem else fit
                for fit in fits]
        wanted = [fit.word for fit in fits if fit.is_fit and (not letters or len(fit.word) == letters)]
        request.fitting += [fit.word for fit in fits if fit.is_fit and fit.word not in request.fitting]
        data = {"fits": [{"word": fit.word, "is_fit": fit.is_fit, "missing": fit.missing, "problem": fit.problem,
                          "is_word": fit.is_word} for fit in fits], "place": place}
        said = describe_fits(fits)
        if not wanted:
            some = f"of {letters} letters" if letters else "shorter ones"
            if letters and any(fit.is_fit for fit in fits):
                said += f" None of them has {letters} letters."
            return f"{said} Call find_word again with 6 other real words, {some}, of only {spaced(tiles)}.", data
        # The first of the longest -- the model's own order -- and not one set down already this visit, if any other.
        used = self.chat.words_used()
        best = max([word for word in wanted if word not in used] or wanted, key=len)
        data["word"] = best
        if not place:
            self.chat.set_word(request.turn, best, PROPOSED)
            return f"{said} Best: {best}; not set down, as asked. Tell the user.", data
        busy = self._placing or request.placed
        if busy:
            self.chat.set_word(request.turn, best, PROPOSED)
            return f"{said} Best: {best}; not set down: the arm is setting down {busy}. Tell the user.", data
        request.placed = best
        self.chat.set_word(request.turn, best, PLACING)
        self._start_placing(best, request.turn)
        data["is_placed"] = True
        return f"{said} The arm has started setting down {best}. Tell the user the word and what it means.", data

    def _detect(self) -> Tuple[str, dict]:
        if self._placing:
            return f"The arm is setting down {self._placing}: the tiles can be looked at once it is done.", {}
        self.desk.detect()
        return f"Looked again. {self._describe_desk()}. If the user wants a word, call find_word now.", {}

    def _start_placing(self, word: str, turn: Optional[Turn]) -> None:
        self._placing = word

        def place() -> None:
            try:
                self.desk.place(word)
                said, status = f"{word} is set down.", PLACED
            except Exception as error:
                said, status = f"{word} was not set down: {str(error) or repr(error)}", NOT_PLACED
            finally:
                self._placing = ""
            self.chat.set_word(turn, word, status)
            self.chat.add("event", said, word=word, is_ok=status == PLACED)
            print(f"[agent] {said}", flush=True)
            try:
                self.on_say(said, status == PLACED)
            except Exception:
                logger.exception("on_say raised")
        Thread(target=place, name="agent-place", daemon=True).start()

    # --- what the model is told -------------------------------------------------------------------------------

    def _letters_now(self) -> Tuple[Optional[str], str]:
        """The letters a word can take now -- those left once the word being set down has taken its own -- or None
        and why there are none."""
        seen = self.desk.seen()
        if seen is None:
            return None, "No tiles looked at yet: call detect_tiles first."
        letters = take_letters(seen["on_offer"], self._placing) if self._placing else seen["on_offer"]
        if not letters:
            return None, "No tiles left to use: every tile is set down, out of reach or not read."
        return letters, ""

    def _describe_desk(self) -> str:
        """The tiles, as the model is told them at the start of each request and after it looks again."""
        seen = self.desk.seen()
        if seen is None:
            return "No tiles looked at yet: detect_tiles finds them."
        letters, why_none = self._letters_now()
        said = f"Tiles you can use ({len(letters)}): {spaced(letters)}" if letters else why_none
        if self._placing:
            said = (f"The arm is setting down {self._placing} now: nothing else can be set down or looked at until "
                    f"it is done. Left after it: {said[0].lower()}{said[1:]}")
        if seen["dropoff"] and not self.chat.words_used():
            said += f"\nIn the dropoff area, left to right from behind the arm: {seen['dropoff']}"
        return said


class _Progress(HookProvider):
    """Counts the model's calls, for the page."""

    def __init__(self, request: _Request) -> None:
        self.request = request

    def register_hooks(self, registry, **kwargs) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before)

    def _before(self, event) -> None:
        self.request.calls += 1
        self.request.doing = f"the model reads and writes (call {self.request.calls})"


def _sentence_case(answer: str, keep: List[str]) -> str:
    """An answer the model wrote all in capitals -- told to write the word in capitals, it sometimes writes all of it
    so -- in lower case but the sentences' first letters, 'I', and the words in `keep`."""
    letters = [character for character in answer if character.isalpha()]
    if len(letters) < 12 or sum(character.isupper() for character in letters) < 0.8 * len(letters):
        return answer
    lowered = re.sub(r"[A-Za-z]+", lambda found: found.group(0) if found.group(0) in keep
                     else "I" if found.group(0) == "I" else found.group(0).lower(), answer)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda found: found.group(1) + found.group(2).upper(), lowered)


def describe_unready(status: dict) -> str:
    """Why a request cannot be put to the model now, in a sentence."""
    state, detail = status["state"], status["detail"]
    if state == NOT_INSTALLED:
        return "The language model is not installed on this board: connector/install.sh, with an Ara240."
    if state == STOPPED:
        return "The language model is installed but not running: connector/run.sh, or connector/install-service.sh."
    if state == WAITING:
        return f"The language model's service is {detail}: it loads the model after that, minutes."
    if state == LOADING:
        left = max(0, int(status["typical_s"] - (time.time() - status["since"]))) if status["since"] else 0
        return ("The language model is loading onto the Ara240" +
                (f": about {left // 60 + 1} min left." if left else ", nearly done."))
    if state == FAILED:
        return f"The language model is not answering: {detail}."
    return "The language model is being looked for: ask again in a moment."


def _openai_spec(decorated) -> dict:
    """A strands tool as the connector is sent it, for its probe: the same guide built for the same tools."""
    spec = decorated.tool_spec
    return {"type": "function", "function": {"name": spec["name"], "description": spec["description"],
                                             "parameters": spec["inputSchema"]["json"]}}
