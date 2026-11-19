"""A question answered in one word, one the letter tiles can spell if there is one.

    answer = answer_question(model, vocabulary, "What colour is an apple?", "AEEGINORT", library, most_letters=14)
    answer.said          # "RED": what the model answers
    answer.word          # "GREEN": what to set down; "" if the tiles spell no answer
    answer.detail        # "the tiles have no D; GREEN is right too", for whoever asked

The model is asked for one word and left to answer: a few tokens, to where it ends. A number it writes in digits,
whatever it is told, is put in letters here (`NUMBERS`). That word is the answer if the tiles spell it. If they do
not, or it gave several words, the model has more to say: its scores for the answer's first token rank every word of
the library the tiles do spell (`library.py`) as an answer too. The best few of those are put to it, one at a time
-- is this a correct answer to the question? -- and the first it says yes to is the answer. A second-best word is
not right for being second: FOUR's runner-up, asked two and two, is THREE.

A prompt takes the model about 2 s: an answer the tiles spell that long, one they do not up to 2 s more for each
word put to it. The model's answers are its own: asked for a father's sister, this one says SISTER.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .ara import SCORE_SCALE, AraModel
from .library import UNSUITABLE, Library
from .vocabulary import Vocabulary

SYSTEM = ("You answer every question with one common English word, written in letters: no digits, no punctuation, "
          "no explanation.")
# An answer is cut short after this many tokens: one word is one to three.
MOST_TOKENS = 8
NUMBERS = {str(number): name for number, name in enumerate(
    "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN FIFTEEN SIXTEEN SEVENTEEN "
    "EIGHTEEN NINETEEN TWENTY".split())}
NUMBERS.update({"30": "THIRTY", "40": "FORTY", "50": "FIFTY", "60": "SIXTY", "70": "SEVENTY", "80": "EIGHTY",
                "90": "NINETY", "100": "HUNDRED", "1000": "THOUSAND", "1000000": "MILLION"})
CHECK_SYSTEM = "You check answers to quiz questions. Reply with yes or no, nothing else."
CHECK = "Question: {question}\nProposed answer: {word}\nIs the proposed answer a correct answer to the question?"
# The library's words put to the model when the tiles do not spell its answer: the best-scored, this many, and
# none scored further behind its own answer than this, natural logarithms of odds.
MOST_PUT = 3
MOST_BEHIND = 9.0


@dataclass(frozen=True)
class Answer:
    question: str
    said: str                # the model's own answer, capitals: a word, or several with spaces between
    word: str                # the word to set down: `said` or another right answer the tiles spell; "" for none
    detail: str              # why `word` is not `said`; "" if it is

    def to_json(self) -> dict:
        return {"question": self.question, "said": self.said, "word": self.word, "detail": self.detail}


def answer_question(model: AraModel, vocabulary: Vocabulary, question: str, letters: str, library: Optional[Library],
                    most_letters: int, is_stopped: Optional[Callable[[], bool]] = None) -> Answer:
    """`letters`: the tiles to spell the answer with. `most_letters`: the longest word that can be set down."""
    first = model.begin(vocabulary.chat(SYSTEM, question))
    scores, tokens = first, []
    while len(tokens) < MOST_TOKENS and int(np.argmax(scores)) not in model.end_tokens:
        tokens.append(int(np.argmax(scores)))
        scores = model.extend(tokens[-1])
    said = " ".join(NUMBERS.get(part, part.upper()) for part in re.findall(r"[A-Za-z]+|\d+", vocabulary.text(tokens)))
    problem = spelling_problem(said, letters, most_letters)
    if not problem:
        return Answer(question, said, said, "")
    for word in _others(vocabulary, first, library.spelled_with(letters) if library else [], most_letters):
        if is_stopped and is_stopped():
            break
        if _is_right(model, vocabulary, question, word):
            return Answer(question, said, word, f"{problem}; {word} is right too")
    return Answer(question, said, "", problem)


def spelling_problem(word: str, letters: str, most_letters: int) -> str:
    """Why the tiles `letters` cannot spell `word`, or it cannot be set down; "" if they can and it can."""
    if len(word) < 2:
        return "not a whole word"                       # the model stops in a word now and then: T, for TEN
    if not word.isalpha():
        return "not one word of letters"
    if word in UNSUITABLE:
        return "not a word this demo spells"
    if len(word) > most_letters:
        return f"{len(word)} letters, and {most_letters} at most fit along the arc"
    need, have = Counter(word), Counter(letters.upper())
    missing = [f"no {letter}" if not have[letter] else f"only {have[letter]} {letter}"
               for letter in sorted(need) if need[letter] > have[letter]]
    return f"the tiles have {', '.join(missing)}" if missing else ""


def _others(vocabulary: Vocabulary, first: np.ndarray, words: List[str], most_letters: int) -> List[str]:
    """Of `words`, the ones the model scored best as its answer, the best first."""
    best = float(first.max())
    scored = [(max(float(first[token]) for token in vocabulary.spells[word]), word)
              for word in words if len(word) <= most_letters]
    close = [(score, word) for score, word in scored if best - score <= MOST_BEHIND * SCORE_SCALE]
    return [word for _, word in sorted(close, reverse=True)[:MOST_PUT]]


def _is_right(model: AraModel, vocabulary: Vocabulary, question: str, word: str) -> bool:
    scores = model.begin(vocabulary.chat(CHECK_SYSTEM, CHECK.format(question=question, word=word.lower())))
    yes, no = (max(float(scores[token]) for said in variants for token in vocabulary.tokens_of(said)[:1])
               for variants in (("Yes", "yes"), ("No", "no")))
    return yes > no
