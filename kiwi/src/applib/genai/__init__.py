"""The language model on the board's Ara240, at work for the letter tiles: the library of words they can spell, and
questions answered in one of them. See `assistant.py`.

| module       | what it owns                                                                          |
| ------------ | ------------------------------------------------------------------------------------- |
| `assistant`  | `Assistant`: the model loaded in the background, the library kept, a question asked   |
| `library`    | the everyday English words the model knows whole, and those of them some tiles spell  |
| `answer`     | a question answered in one word, one the tiles spell if there is one                  |
| `vocabulary` | the model's tokens as letters: which start a word, which go on with one               |
| `ara`        | the model on the Ara240, through NXP's runtime: tokens in, a score for each next out  |

Needs, on the board: NXP's rt-sdk-ara2 with its proxy running, a model compiled for the Ara240 in
`/usr/share/llm/Qwen2.5-7B-Instruct`, and `tokenizers` from pip. Without them the assistant says what is missing,
and everything else runs as before.
"""

from .answer import Answer
from .assistant import Assistant, NotReady

__all__ = ["Answer", "Assistant", "NotReady"]
