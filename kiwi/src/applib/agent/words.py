"""Words checked against the tiles' letters, in code: a model this size cannot count letters (STATUS-kiwi §5).

    missing_letters("AVNETDEMO", "AEEHMORST")      # 'DNV': the letters the tiles lack, one for each short
    describe_missing("DNV", "AEEHMORST")            # 'no D, N, V'
    take_letters("AEEHMORST", "HORSE")              # 'AEMT': the tiles left
    wanted_length("a 7-letter word")                # 7
    meaningful_clue("a 7-letter word")              # '': no clue, a length
    is_placing_refused("a word, don't place it")    # True

Which words there are is the dictionary's (`dictionary.py`).
"""

from __future__ import annotations

import re
from collections import Counter


def missing_letters(word: str, letters: str) -> str:
    """The letters `word` needs that `letters` does not hold, one for each one short, sorted."""
    return "".join(sorted((Counter(word) - Counter(letters)).elements()))


def take_letters(letters: str, word: str) -> str:
    """`letters` once `word` has taken its own: the tiles left over."""
    return "".join(sorted((Counter(letters) - Counter(word)).elements()))


def describe_missing(missing: str, letters: str) -> str:
    """What the tiles `letters` lack, `missing` being what `missing_letters()` said: 'no V, N; too few E'."""
    counts = Counter(missing)
    absent = [letter for letter in sorted(counts) if letter not in letters]
    short = [letter for letter in sorted(counts) if letter in letters]
    said = [f"no {', '.join(absent)}"] if absent else []
    said += [f"too few {', '.join(short)}"] if short else []
    return "; ".join(said)


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


GIVING = re.compile(r"\b(?:spell|write|place|put|set|lay|make|show)\b", re.IGNORECASE)


def is_word_given(request: str) -> bool:
    """Whether the request asks to spell or set down a word of its own -- "spell AvnetDemo", "place HORSE" -- and not
    for one: the model took "a fruit" for the word FRUIT (STATUS-kiwi §8)."""
    return GIVING.search(request) is not None


ANOTHER = re.compile(r"\b(?:another|one more|again|else|next one)\b", re.IGNORECASE)


def is_another(request: str) -> bool:
    """Whether the request asks for one more like the last: 'another one', 'one more', 'again'."""
    return ANOTHER.search(request) is not None


REFUSED = re.compile(r"\b(?:don'?t|do\s+not|never|without|no\s+need\s+to|not)\s+(?:\w+\s+){0,2}?"
                     r"(?:plac|set|put|lay|mov)\w*", re.IGNORECASE)


def is_placing_refused(request: str) -> bool:
    """Whether the request says not to set the word down -- "don't place it", "without placing", "do not set it
    down": read here, as the model passed place=false on "spell HAMSTER" (STATUS-kiwi §8)."""
    return REFUSED.search(request) is not None


# What a clue may say beside its meaning: the model passes on "a 7-letter word" and "the longest word" as clues.
NOT_MEANING = {"a", "an", "the", "any", "one", "some", "word", "words", "longest", "long", "letter", "letters", "of",
               "with", "please", "me", "give", "find", "make", "real", "english"}


def meaningful_clue(clue: str) -> str:
    """`clue` without how many letters it asks for, or "" when it says nothing else: 'a 4-letter animal' is
    'a animal', 'a 7-letter word' nothing."""
    without = re.sub(r"\b(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")[\s-]*letters?\b", " ", clue.lower())
    is_meant = any(word not in NOT_MEANING for word in re.findall(r"[a-z]+", without))
    return " ".join(without.split()) if is_meant else ""


def spaced(letters: str) -> str:
    """'A E E H M O R S T': a letter apart, so that the model sees each one (its tokens are pieces of words)."""
    return " ".join(letters)
