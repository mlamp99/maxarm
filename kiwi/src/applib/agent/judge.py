"""Whether a word is a word: asked of the model alone, a word at a time, nothing else in the prompt -- PILOT-kiwi's
judge, until a dictionary takes its place behind the same call.

    judge = WordJudge(source)
    judge.is_word("THEOREM")        # True: asked once, then remembered; None if the model could not be asked

The model is a poor maker of words from letters -- its guesses that use every tile are mostly not words, and they
fit -- and a better judge of a word put to it alone, a word at a time, a second or so each; the question is the one
that judged best (STATUS-kiwi §6). Rare words may be judged not words: a demo can do without them.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SYSTEM = "You are a strict English dictionary. Answer only yes or no."
QUESTION = 'Would most people know "{word}" as an English word?'
MAX_KEPT = 5000             # verdicts remembered: a day at a booth keeps far fewer


class WordJudge:
    def __init__(self, source) -> None:
        self.source = source                # a `connector.ModelSource`: `ask_plain()`
        self._verdicts: Dict[str, bool] = {}
        self._lock = Lock()

    def is_word(self, word: str) -> Optional[bool]:
        """Whether `word` is an English word, as the model judges it; None if it could not be asked."""
        word = word.upper()
        with self._lock:
            if word in self._verdicts:
                return self._verdicts[word]
        try:
            answer = self.source.ask_plain(SYSTEM, QUESTION.format(word=word))
        except Exception as error:
            logger.debug("the judge could not ask", exc_info=True)
            print(f"    [agent] {word}: not judged ({type(error).__name__})", flush=True)
            return None
        verdict = answer.strip().lower().startswith("yes")
        with self._lock:
            if len(self._verdicts) >= MAX_KEPT:
                self._verdicts.clear()
            self._verdicts[word] = verdict
        return verdict
