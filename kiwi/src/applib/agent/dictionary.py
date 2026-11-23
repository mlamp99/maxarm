"""The agent's dictionary: which words there are, which of them the tiles spell, and which of those are nearest a
clue in meaning. The words the arm sets down come from here, not from the model (STATUS-kiwi §7).

    words = Dictionary()                                    # loads on a thread of its own: seconds
    words.spelled("AEEHMORST", length=7)                   # ['MOTHERS', 'THEOREM', ...]: commonest first
    words.nearest("a small swamp animal", ["VENOM", "TOAD", "DOVE"])      # ['TOAD', 'VENOM', 'DOVE']

The words are `files/words-en.tsv`: everyday English, commonest first, made by `training/words/make-words.py`. The
tiles spell a word when they hold every letter it has, as often as it has it, counted here.

Meaning is told by a small embedding model on the CPU, potion-base-8M: a table of vectors for the pieces of words, a
text's vector their average. Each word's vector is made once, at the start -- about a second on the board -- and a
clue's when it is asked, a millisecond; the nearest words are those whose vectors point most the same way. Ten
clues so, on the board's model: the right word among the first ten for eight (STATUS-kiwi §7). `scripts/install.sh`
puts the model in Hugging Face's cache at REVISION; it is read from there, never fetched.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List, Sequence

import numpy as np

WORDS_FILE = Path(__file__).resolve().parents[3] / "files" / "words-en.tsv"
MODEL = "minishlab/potion-base-8M"                          # as scripts/install.sh fetches it
REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json")
LOAD_WAIT_S = 30.0          # a request in the app's first seconds waits for the dictionary this long at most


class Dictionary:
    def __init__(self, path: Path = WORDS_FILE) -> None:
        self.path = path
        self.words: List[str] = []                  # longest first, of a length the commonest first
        self.zipf: Dict[str, float] = {}            # how common each is: 6 "the", 3 "toad"
        self.problem = "still loading"              # why `spelled()` cannot answer; "" once it can
        self.meaning_problem = "still loading"      # why `nearest()` cannot; "" once it can
        self._counts = np.zeros((0, 26), np.uint8)  # each word's count of each letter, A to Z
        self._lengths = np.zeros(0, np.int64)
        self._vectors = np.zeros((0, 0), np.float32)
        self._rows: Dict[str, int] = {}
        self._model = None
        self._listed, self._loaded = Event(), Event()
        Thread(target=self._load, name="dictionary", daemon=True).start()

    def spelled(self, letters: str, length: int = 0) -> List[str]:
        """The words the tiles' `letters` spell, each tile once -- of `length` letters unless 0 -- longest first, of
        a length the commonest first; LookupError if there is no word list."""
        self._listed.wait(LOAD_WAIT_S)
        if self.problem:
            raise LookupError(self.problem)
        fits = (self._counts <= _counted(letters)).all(axis=1)
        if length:
            fits &= self._lengths == length
        return [self.words[index] for index in np.flatnonzero(fits)]

    def nearest(self, clue: str, words: Sequence[str]) -> List[str]:
        """`words`, the dictionary's own, the nearest `clue` in meaning first; LookupError if meaning cannot be told."""
        self._loaded.wait(LOAD_WAIT_S)
        if self.meaning_problem:
            raise LookupError(self.meaning_problem)
        query = self._model.encode([clue], use_multiprocessing=False)[0]
        scores = self._vectors[[self._rows[word] for word in words]] @ (query / np.linalg.norm(query))
        return [words[index] for index in np.argsort(-scores, kind="stable")]

    # --- loading -------------------------------------------------------------------------------------------------

    def _load(self) -> None:
        started = time.monotonic()
        try:
            self._read_words()
            self.problem = ""
        except (OSError, ValueError) as error:
            self.problem = f"the word list could not be read: {error}"
        finally:
            self._listed.set()
        try:
            self.meaning_problem = self.problem or self._read_meaning()
        except Exception as error:                  # a bad download, a model2vec that changed: said, not raised
            self.meaning_problem = f"{MODEL} could not be loaded: {type(error).__name__}: {error}"
        finally:
            self._loaded.set()
        if self.problem:
            said = f"off, {self.problem}"
        elif self.meaning_problem:
            said = f"{len(self.words)} words; not by meaning: {self.meaning_problem}"
        else:
            said = f"{len(self.words)} words, by meaning too ({time.monotonic() - started:.1f} s)"
        print(f"dictionary : {said}", flush=True)

    def _read_words(self) -> None:
        listed = [line.split("\t") for line in self.path.read_text().splitlines() if line and line[0] != "#"]
        zipf = {word: float(score) for word, score in listed}
        if not zipf or not all(word.isascii() and word.isalpha() and word.isupper() for word in zipf):
            raise ValueError(f"{self.path.name} holds no words, or more than capitals")
        words = sorted(zipf, key=len, reverse=True)     # stable: of a length, the file's order, commonest first
        self._counts = np.array([_counted(word) for word in words], np.uint8)
        self._lengths = np.array([len(word) for word in words])
        self.words, self.zipf = words, zipf

    def _read_meaning(self) -> str:
        """The model and every word's vector; "" once loaded, or why not."""
        try:
            from huggingface_hub import hf_hub_download
            from model2vec import StaticModel
        except ImportError:
            return "model2vec is not installed (src/requirements.txt)"
        try:
            paths = [hf_hub_download(MODEL, name, revision=REVISION, local_files_only=True) for name in MODEL_FILES]
        except Exception:                           # huggingface_hub's LocalEntryNotFoundError, and the like
            return f"{MODEL} is not downloaded (scripts/install.sh)"
        model = StaticModel.from_pretrained(Path(paths[0]).parent)
        vectors = model.encode([word.lower() for word in self.words], use_multiprocessing=False)
        self._vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self._rows = {word: row for row, word in enumerate(self.words)}
        self._model = model
        return ""


def _counted(letters: str) -> np.ndarray:
    """How many of each letter, A to Z, `letters` has; anything else is not counted."""
    codes = np.frombuffer(letters.upper().encode("ascii", "ignore"), np.uint8).astype(np.int64) - ord("A")
    return np.bincount(codes[(codes >= 0) & (codes < 26)], minlength=26)
