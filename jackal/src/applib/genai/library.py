"""The library of words: the everyday English words the language model knows whole, and those of them that a set of
letter tiles spells.

    library = Library.marked(model, vocabulary)        # 40 s, once: the model goes through its own vocabulary
    library.save(path)                                 # kept: Library.load(path, library.source) next time
    library.spelled_with("AABEEGHIINOORRTU")           # ["NEIGHBOR", "ANOTHER", ..., "BEAR", ..., "TEA"]

A language model does not count letters: asked for words of given tiles, this one names words the tiles do not hold,
and joins fragments into words that are none (GOOEI, of 20 tiles, 19 Nov). What it does know is which words are
everyday English ones. So the letters are counted here, and the model is asked only that:

1. the candidates are the model's own tokens that are a whole word start -- a space and small letters, 3 to 9 of
   them (`vocabulary.py`): twenty thousand, most of them no words but fragments, names, abbreviations and other
   languages' words;
2. for each length the model is shown the start of a list of everyday English words of that length, and its scores
   for what comes next are read, once: a score for every candidate, in the time one answer takes. Three lists a
   length, LISTS: of things, of the small words between them, and of what is done and how things are -- a list of
   things alone scores ABOUT and GREAT low;
3. a candidate is kept if in any of the lists it scores within MOST_BEHIND of the model's first choice -- natural
   logarithms of odds. That far down, on Qwen2.5-7B, the tokens are English words; in the list of things the first
   that is none is a few tenths further (`sep`, `cray`, `lifes`, `pupper`, `matcher`, `meilleur`, `verdienen`, by
   length; 19 Nov). Some everyday words lie further down still, among those, and are left out: WOMAN and HEAT are
   not in the library, as it stands;
4. UNSUITABLE words are left out: a demo spells its words in front of people. So are the NOT_WORDS found among
   those kept.

What the tiles spell is then arithmetic, and immediate. The marks do not depend on the tiles, so they are made once
and kept in a file, with what they were made by; the library is there before the model has loaded, and without it.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .ara import SCORE_SCALE, AraModel
from .vocabulary import Vocabulary

LIBRARY_FILE = "words.json"
NAMES = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
SYSTEM = "You are an English dictionary for school children."
ASK = ("List everyday English words of exactly {count} letters that every English speaker knows: no names, no "
       "abbreviations, no words of other languages. Small letters, separated by commas.")
# How the model's list begins, by what it lists and by length: the candidates are scored as what comes next.
LISTS = {
    "things": {
        3: "sun, dog, cup, red, map, key, ice, sky, fox, pen, leg, jam, owl, win, mud",
        4: "time, home, blue, fish, door, milk, jump, wind, cake, lamp, snow, yard, wolf, desk, pink",
        5: "water, house, green, plant, smile, chair, music, lemon, cloud, zebra, piano, dance, wheel, juicy, flame",
        6: "garden, yellow, window, people, simple, market, flower, castle, silver, pencil, monkey, bridge, cheese",
        7: "morning, kitchen, picture, weather, chicken, blanket, journey, library, diamond, penguin, thunder",
        8: "mountain, children, question, together, remember, sandwich, elephant, daughter, umbrella, hospital",
        9: "beautiful, different, important, yesterday, chocolate, adventure, butterfly, vegetable, knowledge",
    },
    "between": {
        3: "the, and, but, for, not, you, all, any, can, her, was, one, our, out, how, why, too, yes, far, big",
        4: "that, with, have, this, will, your, from, they, know, want, been, good, much, some, very, when, come, here",
        5: "about, there, which, would, could, after, think, never, every, again, still, where, being, might, today",
        6: "before, should, around, always, really, little, enough, either, rather, almost, behind, toward, during",
        7: "because, through, another, without, between, nothing, already, against, however, perhaps, usually",
        8: "although, together, anything, yourself, actually, probably, whenever, tomorrow, somebody, wherever",
        9: "something, different, sometimes, everybody, important, certainly, otherwise, therefore, somewhere",
    },
    "doing": {
        3: "run, eat, sit, get, put, say, see, buy, cry, dig, fly, hug, mix, pay, try, old, new, sad, hot, wet",
        4: "walk, jump, sing, read, cook, help, play, open, swim, wash, kind, slow, tall, warm, soft, rich, loud",
        5: "bring, carry, laugh, sleep, write, dance, clean, drive, happy, quick, brave, small, sweet, young, tired",
        6: "listen, travel, answer, decide, follow, gather, borrow, repair, pretty, strong, gentle, bright, clever",
        7: "believe, explain, forgive, prepare, improve, collect, careful, healthy, curious, crowded, helpful",
        8: "remember, complete, discover, continue, exercise, practise, friendly, peaceful, generous, terrible",
        9: "celebrate, recognize, introduce, apologize, encourage, wonderful, beautiful, dangerous, delicious",
    },
}
# A word is kept if it scores this close to the model's first choice in any of the lists, by length: three tenths
# short of the best-scored token that is no word, in the list of things.
MOST_BEHIND = {3: 5.2, 4: 4.3, 5: 4.6, 6: 5.0, 7: 6.1, 8: 5.0, 9: 6.3}
# The file keeps the words this close, so that MOST_BEHIND can be set again without asking the model again.
KEPT_BEHIND = 8.0
UNSUITABLE = frozenset("""
    ANAL ANUS ARSE ASS BITCH BOOB BOOBS BUTT COCAINE COCK CRAP CUM CUNT DAMN DICK DILDO EROTIC FAG FUCK HELL HENTAI
    HEROIN HOMO HORNY KILL LIBIDO MURDER NAKED NAZI NEGRO NIGGER NUDE ORGASM PENIS PISS PORN PORNO PUSSY RAPE RAPIST
    SEX SEXY SHIT SLAVE SLAVERY SLUT SUICIDE TIT TITS VAGINA WHORE""".split())
NOT_WORDS = frozenset({"DIFFICILE", "GONNA"})


class Library:
    def __init__(self, behind: Dict[str, float], source: str) -> None:
        """`behind`: each candidate word, capitals, and how far it scored behind the model's first choice, in the
        list it did best in. `source`: what marked them, the model and the questions."""
        self.behind = behind
        self.source = source
        self.words = sorted((word for word in behind if word in self), key=lambda word: (-len(word), behind[word]))
        self._spelled: Dict[str, List[str]] = {}

    @classmethod
    def marked(cls, model: AraModel, vocabulary: Vocabulary,
               is_stopped: Optional[Callable[[], bool]] = None) -> Optional["Library"]:
        """The model's vocabulary gone through, a list and a length at a time; None if `is_stopped()` says so on the
        way."""
        behind: Dict[str, float] = {}
        for examples in LISTS.values():
            for length in sorted(NAMES, reverse=True):
                if is_stopped and is_stopped():
                    return None
                scores = model.begin(vocabulary.chat(SYSTEM, ASK.format(count=NAMES[length]), examples[length] + ","))
                candidates = np.flatnonzero(vocabulary.starts & vocabulary.is_lower & (vocabulary.lengths == length))
                gaps = (float(scores.max()) - scores[candidates]) / SCORE_SCALE
                for token, gap in zip(candidates, gaps):
                    word = vocabulary.letters_of(int(token))
                    if gap <= min(KEPT_BEHIND, behind.get(word, KEPT_BEHIND)):
                        behind[word] = round(float(gap), 2)
        return cls(behind, source_of(model))

    @classmethod
    def load(cls, path: Path, source: str) -> Optional["Library"]:
        """The library kept in `path`, if it was marked by `source`; None if there is none, or by something else."""
        try:
            kept = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None
        if kept.get("source") != source or not isinstance(kept.get("behind"), dict):
            return None
        return cls({str(word): float(gap) for word, gap in kept["behind"].items()}, source)

    def save(self, path: Path) -> None:
        partial = Path(path).with_name(Path(path).name + ".partial")
        partial.write_text(json.dumps({"source": self.source, "behind": self.behind}, indent=0) + "\n")
        partial.replace(path)

    def spelled_with(self, letters: str) -> List[str]:
        """The library's words the tiles `letters` spell, each tile once in a word: the longest first, and of a
        length, those the model scored best."""
        tiles = "".join(sorted(letters.upper()))
        if tiles not in self._spelled:
            held = Counter(tiles)
            self._spelled[tiles] = [word for word in self.words if not Counter(word) - held]
        return self._spelled[tiles]

    def __contains__(self, word: str) -> bool:
        return (self.behind.get(word, KEPT_BEHIND + 1.0) <= MOST_BEHIND.get(len(word), -1.0)
                and word not in UNSUITABLE and word not in NOT_WORDS)


def source_of(model: AraModel) -> str:
    """What a library is marked by: the model's file, as its own description has it, and what it is asked."""
    try:
        described = json.loads((model.model_dir / "model.dvm.json").read_text()).get("md5sum", "")
    except (OSError, ValueError):
        described = ""
    asked = json.dumps([SYSTEM, ASK, LISTS], sort_keys=True)
    return f"{model.model_dir.name} {described} asked {zlib.crc32(asked.encode()):08x}"
