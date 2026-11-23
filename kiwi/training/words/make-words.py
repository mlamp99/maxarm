#!/usr/bin/env python3
"""Make the agent's word list, files/words-en.tsv: everyday English words and how common each is.

    ../.venv/bin/python training/words/make-words.py      # -> files/words-en.tsv, seconds

with `training/words/requirements.txt` installed; WordNet's data is fetched by nltk the first time.

How common: wordfreq's Zipf frequency -- 6 is "the", 4 "garden", 3 "toad" -- of film subtitles, Wikipedia, news,
books and the web, a snapshot of about 2021 that is no longer updated. Which are words: the PC's Hunspell en_US
dictionary, made from SCOWL (Fedora's hunspell-en-US, Debian's hunspell-en-us), asked in lower case, so that names
drop out; and WordNet, words with a meaning of their own -- inflections taken -- so that THE, WITH, BECAUSE and
GONNA drop out too, the commonest words of a length otherwise (STATUS-kiwi §7). Kept: 3 to 15 letters, A to Z,
Zipf 3 or more, but NOT_WORDS.

Host-side only; the board reads the file (`src/applib/agent/dictionary.py`). wordfreq's data is CC BY-SA 4.0, with
credits to the sources it was made from (https://github.com/rspeer/wordfreq); SCOWL's terms are MIT-like, and
WordNet's (Princeton) BSD-like.
"""

import sys
from pathlib import Path

import nltk
from nltk.corpus import wordnet
from spylls.hunspell import Dictionary
from wordfreq import iter_wordlist, zipf_frequency

OUT = Path(__file__).resolve().parents[2] / "files" / "words-en.tsv"
HUNSPELL = "/usr/share/hunspell/en_US"
MIN_ZIPF = 3.0
LETTERS = range(3, 16)
# Abbreviations and Roman numerals that Hunspell and WordNet both take: no words to set down.
NOT_WORDS = set("""
    ACC ALT APR ASAP ATM AUG CORP DEC DEPT ESP EST ETC FEB FRI GOV III INC JAN JUN LTD MISC MON NOV OCT SEC SEP SEPT
    THU THUR THURS TUE TUES VAR VOL XII XIII XIV XVI XVII XIX XXX""".split())


def main() -> int:
    nltk.download("wordnet", quiet=True)
    hunspell = Dictionary.from_files(HUNSPELL)
    kept = []
    for word in iter_wordlist("en", wordlist="large"):          # commonest first
        zipf = zipf_frequency(word, "en", wordlist="large")
        if zipf < MIN_ZIPF:
            break
        if (word.isascii() and word.isalpha() and len(word) in LETTERS and word.upper() not in NOT_WORDS
                and hunspell.lookup(word) and wordnet.synsets(word)):
            kept.append((word.upper(), zipf))
    lines = ["# Everyday English words and their Zipf frequency, made by training/words/make-words.py from",
             "# wordfreq's English list (data CC BY-SA 4.0, https://github.com/rspeer/wordfreq), Hunspell en_US",
             f"# (SCOWL) and WordNet: {len(kept)} words of {LETTERS.start} to {LETTERS.stop - 1} letters, "
             f"Zipf {MIN_ZIPF:g} or more.",
             *(f"{word}\t{zipf:.2f}" for word, zipf in kept)]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{OUT}: {len(kept)} words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
