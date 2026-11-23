"""The word a request asks for, found in code: the user's own, the longest the tiles spell, one of a length, or one
that fits a clue. The words are the dictionary's (`dictionary.py`); where more than one would do, the choice among a
few is the model's -- it knows what a toad is, and cannot count letters (STATUS-kiwi §7, §8).

    finder = WordFinder(Dictionary(), source)
    found = finder.find("AEEHMORST", clue="a large farm animal", given="", length=0, used=[], check=desk.check)
    found.word          # 'HORSE'; "" when there is none, and `found.tell` says why

- **Given** -- "spell AvnetDemo": the user's word as it is, a name or not, if the tiles hold its letters; never another
  in its place.
- **The longest** -- "go": the longest word the tiles spell, of a length the commonest; one the arm cannot set down
  gives way to the next.
- **A length** -- "a 5-letter word": the commonest words of the length but the little ones -- THERE, ONE, WAS, Zipf
  LITTLE or more -- SHORTLIST of them put to the model, which chooses the most vivid.
- **A clue** -- "a small swamp animal": the words the tiles spell, the nearest the clue in meaning first, SHORTLIST of
  them put to the model, which names the one the clue fits, or NONE. Ten clues so: eight right, 1.5 s each; held to
  the list by the connector, the model chose worse, and never NONE (STATUS-kiwi §7, §8).

A word the model names that is not among those put to it -- the clue's own, mostly, FLOWER for "a flower" -- is taken
for the first of them. A word set down before in this visit (`used`) is chosen again only when no other is there.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .dictionary import Dictionary
from .words import describe_missing, missing_letters

GIVEN, LONGEST, LENGTH, CLUE = "given", "longest", "length", "clue"
SHORTLIST = 12              # the words put to the model, as tried (STATUS-kiwi §7)
SHOWN = 8                   # the longest words the page shows beside the one chosen
TRIED = 12                  # the longest words planned at most, for one the arm can set down
LITTLE = 6.0                # Zipf: the commonest words of English, little ones mostly (STATUS-kiwi §8)
NONE = "NONE"
CLUE_SYSTEM = "You match words to descriptions. Answer with one word."
CLUE_QUESTION = ('Words: {words}.\nDescription: "{clue}"\nWhich ONE word of the list does the description fit best? '
                 "Judge by meaning alone, not by length. Answer with that word only, or NONE if none fits.")
LENGTH_SYSTEM = "You pick words. Answer with one word of the list."
LENGTH_QUESTION = ("List: {words}.\nPick the ONE word of the list that names a thing or an action, the most vivid one. "
                   "Answer with that word only.")


@dataclass
class Found:
    kind: str                   # GIVEN, LONGEST, LENGTH or CLUE
    word: str = ""              # the word to set down; "" for none
    about: str = ""             # the word, for the model: 'TOAD (T-O-A-D, 4 letters) fits "a small swamp animal"'
    tell: str = ""              # for the model, why there is none
    shown: List[str] = field(default_factory=list)    # the words weighed, for the page
    said: str = ""              # what the model named: one of `shown`, NONE, or another
    missing: str = ""           # what the tiles lack for a given word
    problem: str = ""           # why the arm cannot set the word down
    asked: str = ""             # what the model was asked, and its answer
    seconds: float = 0.0        # the model's time on it


class WordFinder:
    def __init__(self, dictionary: Dictionary, source) -> None:
        self.dictionary = dictionary
        self.source = source            # a `connector.ModelSource`: `ask_plain()`

    def find(self, tiles: str, clue: str, given: str, length: int, used: Sequence[str],
             check: Optional[Callable[[str], str]]) -> Found:
        """The word for a request. `tiles`: the letters to use; `clue`, `given`: the model's reading of the request;
        `length`: the letters asked for, 0 for any; `check(word)`: why the arm cannot set it down, "" if it can --
        None when nothing can be planned, the arm busy."""
        check = check or (lambda word: "")
        given = "".join(re.findall(r"[A-Za-z]+", given)).upper()
        if given:
            return self._given(given, tiles, check)
        kind = CLUE if clue.strip() else LENGTH if length else LONGEST
        try:
            words = self.dictionary.spelled(tiles, length)
        except LookupError as reason:
            return Found(kind, tell=f"The dictionary is not there: {reason}. Tell the user.")
        if not words:
            some = f"of {length} letters " if length else ""
            return Found(kind, tell=f"The tiles spell no word {some}in the dictionary. Tell the user.")
        fresh = [word for word in words if word not in used] or words
        if kind == CLUE:
            return self._clue(" ".join(clue.split()), fresh, check)
        if kind == LENGTH:
            return self._of_length(fresh, length, check)
        return self._longest(fresh, check)

    def _given(self, word: str, tiles: str, check: Callable[[str], str]) -> Found:
        missing = missing_letters(word, tiles)
        if missing:
            return Found(GIVEN, shown=[word], missing=missing,
                         tell=f"{word} cannot be set down: the tiles have {describe_missing(missing, tiles)}. Tell "
                              "the user so, and choose no other word.")
        problem = check(word)
        if problem:
            return Found(GIVEN, shown=[word], problem=f"{word}, {problem}",
                         tell=f"The arm cannot set {word} down: {problem}. Tell the user.")
        return Found(GIVEN, word=word, shown=[word], about=f"{_spelled(word)}, as the user asked")

    def _longest(self, words: List[str], check: Callable[[str], str]) -> Found:
        problems = []
        for word in words[:TRIED]:
            problem = check(word)
            if not problem:
                return Found(LONGEST, word=word, shown=words[:SHOWN],
                             about=f"{_spelled(word)} is the longest word the tiles spell")
            problems.append(f"{word}, {problem}")
        return Found(LONGEST, shown=words[:SHOWN], problem=problems[0],
                     tell=f"The arm cannot set down the longest words: {problems[0]}. Tell the user.")

    def _of_length(self, words: List[str], length: int, check: Callable[[str], str]) -> Found:
        everyday = [word for word in words if self.dictionary.zipf[word] < LITTLE] or words
        found = self._ask(LENGTH, everyday[:SHORTLIST], LENGTH_SYSTEM, LENGTH_QUESTION, clue="")
        return self._settle(found, check, f"is a {length}-letter word the tiles spell, the model's choice")

    def _clue(self, clue: str, words: List[str], check: Callable[[str], str]) -> Found:
        try:
            nearest = self.dictionary.nearest(clue, words)[:SHORTLIST]
        except LookupError as reason:
            return Found(CLUE, tell=f"Words cannot be found by meaning: {reason}. Tell the user.")
        found = self._ask(CLUE, nearest, CLUE_SYSTEM, CLUE_QUESTION, clue)
        if found.said == NONE:          # the nearest not named: the model took one and said so (STATUS-kiwi §8)
            found.tell = f'None of the words the tiles spell is "{clue}": nothing is set down. Tell the user so.'
            return found
        return self._settle(found, check, f'fits "{clue}"')

    def _ask(self, kind: str, words: List[str], system: str, question: str, clue: str) -> Found:
        """The model's choice among `words`: one of them, NONE, or another word it named."""
        question = question.format(words=", ".join(words), clue=clue)
        started = time.monotonic()
        answer = self.source.ask_plain(system, question)
        return Found(kind, shown=words, said=_named(answer, words), asked=f"{system}\n{question}\n-> {answer.strip()}",
                     seconds=round(time.monotonic() - started, 1))

    def _settle(self, found: Found, check: Callable[[str], str], why: str) -> Found:
        """`found` with the word the model chose -- or the first put to it, if it named another -- if the arm can set
        it down."""
        word = found.said if found.said in found.shown else found.shown[0]
        problem = check(word)
        if problem:
            found.problem = f"{word}, {problem}"
            found.tell = f"{word} fits, but the arm cannot set it down: {problem}. Tell the user."
            return found
        found.word, found.about = word, f"{_spelled(word)} {why}"
        return found


def _named(answer: str, words: Sequence[str]) -> str:
    """The word the model's answer names: the first of `words` in it, or NONE, or else its first word."""
    named = re.findall(r"[A-Za-z]+", answer.upper())
    return next((word for word in named if word in words or word == NONE), named[0] if named else "")


def _spelled(word: str) -> str:
    """'TOAD (T-O-A-D, 4 letters)': spelled for the model, which copies it rather than spell it wrong (MEANT as
    M-E-A-N-M-E-N-T, STATUS-kiwi §8)."""
    return f"{word} ({'-'.join(word)}, {len(word)} letters)"
