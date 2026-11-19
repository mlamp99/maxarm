"""The GenAI page's requests: the words the tiles can spell, one of them set down, and a question answered in one.

The language model is `applib/genai`: on the board's Ara240, loaded in the background when the app starts. The
tiles are the Tiles page's -- the same detection, the same PLACE, the same settings for the grip and the reader
(`tiles/api.py`) -- so LOCATE TILES on either page serves both.

| request                  | body        | does                                                                       |
| ------------------------ | ----------- | -------------------------------------------------------------------------- |
| `POST /api/genai/build`  | word        | that word set down, as PLACE does                                          |
|                          | pick        | a word of the library set down: `longest`, or a number of letters, 3 to 9; |
|                          |             | of several, one at random, and not the one built last                      |
| `POST /api/genai/ask`    | question    | the model's answer in one word, set down if the tiles spell it; if they do |
|                          |             | not, another word of the library the model holds right too, if there is one|

The letters a word is made of are those on the tiles ready to pick up, their letters read: `tiles.found.on_offer`.
The state's `genai` has the model's status, those letters, the library's words they spell, the longest first, and
the last question with its answer.
"""

import random
from typing import List, Optional

from applib import areas
from applib.genai import Assistant, NotReady
from common.server import BadRequest, Server
from tiles.api import TilesSession

MOST_QUESTION = 200
LONGEST = "longest"


class GenaiSession:
    def __init__(self, tiles: TilesSession, assistant: Assistant) -> None:
        self.tiles = tiles
        self.assistant = assistant
        self.answer: Optional[dict] = None          # the last question, what the model said, what was set down
        self.built = ""                             # the word a pick chose last

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.post("/api/genai/build", self.build)
        server.post("/api/genai/ask", self.ask)

    # --- the requests -----------------------------------------------------

    def build(self, body: dict) -> dict:
        if "word" in body:
            word = "".join(str(body["word"]).split()).upper()
        else:
            word = self._picked(str(body.get("pick", "")))
            if not word:
                return {"status": "refused", "detail": self._why_no_pick(str(body.get("pick", "")))}
            self.built = word
        return {**self.tiles.place_word(word), "word": word}

    def ask(self, body: dict) -> dict:
        question = " ".join(str(body.get("question", "")).split())
        if not question:
            raise BadRequest("no question given")
        if len(question) > MOST_QUESTION:
            raise BadRequest(f"a question is {MOST_QUESTION} characters at most, not {len(question)}")
        letters = self._letters()
        if not letters:
            return {"status": "refused", "detail": "no letters on tiles ready to pick up: LOCATE TILES first"}
        try:
            answer = self.assistant.ask(question, letters, areas.WORD_MAX_LETTERS)
        except NotReady as reason:
            return {"status": "refused", "detail": str(reason)}
        self.answer = {**answer.to_json(), "letters": letters, "placed": None}
        if not answer.word:
            return {"status": "done", "detail": f"the model says {answer.said or 'nothing'}: {answer.detail}; "
                                                f"nothing set down"}
        placed = self.tiles.place_word(answer.word)
        self.answer = {**self.answer, "placed": {"status": placed["status"], "detail": placed["detail"]}}
        return {**placed, "word": answer.word}

    # --- for the page -----------------------------------------------------

    def state(self) -> dict:
        letters = self._letters()
        problem = self.tiles.camera_file.get()[1]
        return {"genai": {"model": self.assistant.state(), "letters": letters,
                          "words": self.assistant.words(letters), "answer": self.answer},
                "unavailable": {"genai": problem} if problem else {}}

    def _letters(self) -> str:
        """The letters on the tiles ready to pick up, as the last detection has them; "" before one."""
        found = self.tiles.work.scene()["found"]
        return found["on_offer"] if found else ""

    def _picked(self, pick: str) -> str:
        choices = self._choices(pick)
        fresh = [word for word in choices if word != self.built] or choices
        return random.choice(fresh) if fresh else ""

    def _choices(self, pick: str) -> List[str]:
        words = self.assistant.words(self._letters())
        if pick == LONGEST:
            return [word for word in words if len(word) == len(words[0])]       # the longest come first
        return [word for word in words if str(len(word)) == pick]

    def _why_no_pick(self, pick: str) -> str:
        if not self._letters():
            return "no letters on tiles ready to pick up: LOCATE TILES first"
        if not self.assistant.state()["library"]:
            return "the library of words is not there yet: the language model marks it once it has loaded"
        wanted = "no word" if pick == LONGEST else f"no word of {pick} letters"
        return f"the library has {wanted} that the tiles ready to pick up spell"
