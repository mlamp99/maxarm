"""Words checked against the tiles' letters, in code: a model this size cannot count letters (STATUS-kiwi §5).

    fits = check_words("THEOREM harmony, MOTHER", "AEEHMORST", already=set())
    describe_fits(fits)     # 'Fit: THEOREM (7 letters), MOTHER (6). Do not fit: HARMONY (no N, Y).'

A word fits when the tiles hold every letter it has, as often as it has it, and it is a word. The letters are
counted here; whether it is a word, and where the arm can set it down, are the caller's to add -- the judge's verdict
(`judge.py`, there is no dictionary yet) and the snapshot's plan.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

# A check's words at most: a list longer than this is the model rambling, and writing is the slow part of its work.
MAX_WORDS = 12
# Shorter than this is not a word to set down.
MIN_LETTERS = 2


@dataclass(frozen=True)
class Fit:
    word: str
    missing: str              # what the tiles lack for it, a letter for each one short, sorted; "" when none
    letters: str              # the tiles' letters it was checked against
    problem: str = ""         # why it cannot be set down although the letters are there; "" when it can
    is_repeat: bool = False   # checked before in the same request
    is_word: Optional[bool] = None    # whether it is a word, as the judge has it (`judge.py`); None: not asked

    @property
    def is_fit(self) -> bool:
        return not self.missing and not self.problem and self.is_word is not False


def split_words(text: str) -> List[str]:
    """The words in what the model wrote -- spaces, commas, quotes, a list, whatever -- upper case, each once."""
    seen: List[str] = []
    for word in re.findall(r"[A-Za-z]+", text):
        word = word.upper()
        if len(word) >= MIN_LETTERS and word not in seen:
            seen.append(word)
    return seen


def missing_letters(word: str, letters: str) -> str:
    """The letters `word` needs that `letters` does not hold, one for each one short, sorted."""
    return "".join(sorted((Counter(word) - Counter(letters)).elements()))


def take_letters(letters: str, word: str) -> str:
    """`letters` once `word` has taken its own: the tiles left over."""
    return "".join(sorted((Counter(letters) - Counter(word)).elements()))


def check_words(text: str, letters: str, already: Set[str]) -> List[Fit]:
    """Each word in `text`, MAX_WORDS at most, checked against `letters`; `already`, the words checked before in the
    same request, is added to."""
    fits = []
    for word in split_words(text)[:MAX_WORDS]:
        fits.append(Fit(word, missing_letters(word, letters), letters, is_repeat=word in already))
        already.add(word)
    return fits


def describe_fits(fits: Iterable[Fit]) -> str:
    """For the model, short: the words that fit, longest first, with their letter counts -- the count is what it cannot
    do -- then those whose letters are there but cannot be set down, and those that do not fit, with what they lack."""
    fits = list(fits)
    parts = []
    good = sorted((fit for fit in fits if fit.is_fit), key=lambda fit: -len(fit.word))
    if good:
        parts.append("Fit: " + ", ".join(f"{fit.word} ({len(fit.word)} letters)" for fit in good) + ".")
    blocked = [fit for fit in fits if not fit.missing and fit.problem]
    made_up = [fit.word for fit in fits if not fit.missing and not fit.problem and fit.is_word is False]
    if made_up:
        parts.append(f"Not English words: {', '.join(made_up)}.")
    for problem in dict.fromkeys(fit.problem for fit in blocked):
        words = ", ".join(f"{fit.word} ({len(fit.word)} letters)" for fit in blocked if fit.problem == problem)
        parts.append(f"The letters are there for {words}, but the arm cannot set it down: {problem}.")
    glued = [fit.word for fit in fits if len(fit.word) > len(fit.letters)]
    bad = [fit for fit in fits if fit.missing and fit.word not in glued]
    if bad:
        parts.append("Do not fit: " + ", ".join(f"{fit.word} ({_lacks(fit)})" for fit in bad) + ".")
    if glued:
        parts.append(f"Longer than the {len(fits[0].letters)} tiles: {', '.join(glued)}; put a space between words.")
    repeats = [fit.word for fit in fits if fit.is_repeat]
    if repeats:
        parts.append(f"Checked before: {', '.join(repeats)}; do not try them again.")
    if not good and not blocked and fits:
        parts.append(f"No English word fits. The letters are {spaced(fits[0].letters)}.")
    return " ".join(parts) or "No words given: write the words, separated by spaces."


NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12}


def wanted_length(request: str) -> int:
    """How many letters the request asks the word to have -- '7-letter word', 'seven letters', 'of 5 letters' -- or 0:
    read here, as the model passed on the wrong number (STATUS-kiwi §6)."""
    found = re.search(r"\b(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")[\s-]*letters?\b", request.lower())
    if not found:
        return 0
    number = found.group(1)
    return int(number) if number.isdigit() else NUMBER_WORDS[number]


def spaced(letters: str) -> str:
    """'A E E H M O R S T': a letter apart, so that the model sees each one (its tokens are pieces of words)."""
    return " ".join(letters)


def _lacks(fit: Fit) -> str:
    counts = Counter(fit.missing)
    absent = [letter for letter in sorted(counts) if letter not in fit.letters]
    short = [letter for letter in sorted(counts) if letter in fit.letters]
    said = [f"no {', '.join(absent)}"] if absent else []
    said += [f"too few {', '.join(short)}"] if short else []
    return "; ".join(said)
