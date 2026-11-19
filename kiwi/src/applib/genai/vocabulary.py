"""The language model's vocabulary, as letters: which tokens start a word, which go on with one, and what each spells.

    vocabulary = Vocabulary(model.tokenizer_file, model.vocabulary_size)
    tokens = vocabulary.chat("You answer in one word.", "What colour is an apple?")     # the prompt, as tokens
    vocabulary.starts[token], vocabulary.letters_of(token)       # True, "GREEN": the token " green"
    vocabulary.spells["GREEN"]                                   # the tokens that are that word: " green", "Green", ...

The model writes in pieces, tokens: a common word is one, a space before it (" green"); a rarer one is several, the
first with the space, the rest without (" gre", "en"); and the first word of an answer has no space ("Green"). So a
token of a space and letters starts a word, and one of letters alone goes on with the word before it, or begins an
answer. Only those are of use for tiles; every other token -- digits, punctuation, other scripts -- neither starts
nor goes on with a word here.

The tokenizer is Hugging Face's `tokenizers` (pip), read from the model's own `tokenizer.json`; the prompt is laid out
as Qwen's chat template has it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

# How the tokenizer writes a space at a token's start: U+0120.
SPACE = chr(0x120)
CHAT = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


class Vocabulary:
    def __init__(self, tokenizer_file: Path, size: int) -> None:
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.size = size
        self.spelled: List[str] = [""] * size           # a token's letters, capitals; "" if it is not a word's
        self.starts = np.zeros(size, dtype=bool)        # a space, then letters
        self.goes_on = np.zeros(size, dtype=bool)       # letters alone
        self.is_lower = np.zeros(size, dtype=bool)      # its letters all small ones
        self.lengths = np.zeros(size, dtype=np.int16)
        self.spells: Dict[str, List[int]] = {}          # a word, capitals: the tokens that are it whole
        for token in range(size):
            written = self.tokenizer.id_to_token(token)
            if not written:
                continue
            is_start = written.startswith(SPACE)
            letters = written[1:] if is_start else written
            if not letters or not letters.isascii() or not letters.isalpha():
                continue
            self.spelled[token] = letters.upper()
            self.starts[token], self.goes_on[token] = is_start, not is_start
            self.is_lower[token] = letters.islower()
            self.lengths[token] = len(letters)
            self.spells.setdefault(letters.upper(), []).append(token)

    def chat(self, system: str, user: str, begun: str = "") -> List[int]:
        """The prompt's tokens: what the model is told, what it is asked, and how its answer begins."""
        return self.tokens_of(CHAT.format(system=system, user=user) + begun)

    def letters_of(self, token: int) -> str:
        return self.spelled[token]

    def tokens_of(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def text(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
