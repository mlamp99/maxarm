"""The language model at work for the tiles: loaded in the background, its library of words, its answers.

    assistant = Assistant(CONFIG_DIR)
    assistant.start()                           # a thread: the model loads, 4 minutes; the library is marked, once
    assistant.state()                           # for a page: how far that is, and what it is asked
    assistant.words("AABEEGHIINOORRTU")         # the library's words those tiles spell, the longest first
    assistant.ask("What colour is an apple?", "AABEEGHIINOORRTU", most_letters=14)   # an Answer; raises NotReady
    assistant.close()

The library is kept in `config/words.json` (`library.py`): once marked it is there at once, before the model has
loaded and on a machine with no Ara240 at all, and so are the words the tiles spell. Questions need the model.
Nothing here is in the way of the arm or the camera: a model that does not load is said in `state()`, and that is
all. One question at a time; a second, meanwhile, is refused.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

from .answer import Answer, answer_question
from .ara import MODEL_DIR, AraModel, ModelUnavailable, ProxyNotReady
from .library import LIBRARY_FILE, Library, source_of
from .vocabulary import Vocabulary

# How long the model takes to load onto the Ara240, for a page to count towards: 8 GB, measured 19 Nov.
LOAD_SECONDS = 220
MARK_SECONDS = 45
CLOSE_WAIT_S = 5.0
# After a boot the Ara240's proxy comes up a minute after this app: it is asked for again this often, this long.
PROXY_RETRY_S = 5.0
PROXY_WAIT_S = 600.0


class NotReady(RuntimeError):
    """The model cannot answer now; the message says why."""


class Assistant:
    def __init__(self, config_dir: Path, model_dir: Path = MODEL_DIR) -> None:
        self.library_file = Path(config_dir) / LIBRARY_FILE
        self.model = AraModel(model_dir)
        self.vocabulary: Optional[Vocabulary] = None
        self.library = Library.load(self.library_file, source_of(self.model))
        self.lock = threading.Lock()            # held by whoever is using the model
        self.status = "idle"                    # idle, waiting, loading, marking, ready, failed
        self.detail = ""
        self.since = time.monotonic()
        self.asking: Optional[str] = None
        self.is_closing = False
        self.thread = threading.Thread(target=self._prepare, name="genai", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def words(self, letters: str) -> List[str]:
        library = self.library
        return library.spelled_with(letters) if library and letters else []

    def ask(self, question: str, letters: str, most_letters: int) -> Answer:
        if self.status != "ready":
            raise NotReady(self._why_not())
        if not self.lock.acquire(blocking=False):
            raise NotReady("the model is answering another question")
        try:
            self.asking = question
            return answer_question(self.model, self.vocabulary, question, letters, self.library, most_letters,
                                   lambda: self.is_closing)
        except ModelUnavailable as reason:
            self._set("failed", str(reason))
            raise NotReady(self._why_not()) from None
        finally:
            self.asking = None
            self.lock.release()

    def state(self) -> dict:
        library = self.library
        typical = {"loading": LOAD_SECONDS, "marking": MARK_SECONDS}.get(self.status)
        return {"status": self.status, "detail": self.detail, "seconds": round(time.monotonic() - self.since),
                "typical": typical, "name": self.model.model_dir.name, "asking": self.asking,
                "library": len(library.words) if library else 0}

    def close(self) -> None:
        """The model taken off the device, if it is not in the middle of loading: that cannot be cut short."""
        self.is_closing = True
        if self.lock.acquire(timeout=CLOSE_WAIT_S):
            try:
                self.model.close()
            finally:
                self.lock.release()

    def _prepare(self) -> None:
        try:
            with self.lock:
                try:
                    import tokenizers               # noqa: F401 -- missed now, not after minutes of loading
                except ImportError:
                    raise ModelUnavailable("the tokenizers package is not installed: pip install -r "
                                           "src/requirements.txt") from None
                self._load()
                self.vocabulary = Vocabulary(self.model.tokenizer_file, self.model.vocabulary_size)
                if self.library is None and not self.is_closing:
                    self._set("marking")
                    library = Library.marked(self.model, self.vocabulary, lambda: self.is_closing)
                    if library is not None:
                        library.save(self.library_file)
                        self.library = library
            self._set("ready")
        except ModelUnavailable as reason:
            self._set("failed", str(reason))
        except Exception as error:                  # said on the page; the console keeps the trace
            traceback.print_exc(file=sys.stderr)
            self._set("failed", str(error) or repr(error))

    def _load(self) -> None:
        """The model loaded, once the Ara240's proxy is there to load it through."""
        started = time.monotonic()
        while True:
            self._set("loading")
            try:
                return self.model.load()
            except ProxyNotReady as reason:
                if self.is_closing or time.monotonic() - started > PROXY_WAIT_S:
                    raise
                self._set("waiting", str(reason))
                time.sleep(PROXY_RETRY_S)

    def _set(self, status: str, detail: str = "") -> None:
        self.status, self.detail, self.since = status, detail, time.monotonic()

    def _why_not(self) -> str:
        if self.status == "failed":
            return f"the language model is not available: {self.detail}"
        if self.status == "loading":
            waited = round(time.monotonic() - self.since)
            return f"the language model is still loading onto the Ara240: {waited} s of about {LOAD_SECONDS}"
        if self.status == "waiting":
            return f"the language model is not loaded yet: {self.detail}"
        return "the language model is not ready yet"
