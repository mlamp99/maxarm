"""What the language model is told: the system prompt, and the message each request becomes.

    prompt.compose("a 7-letter word", desk="Tiles you can use (9): A E E H M O R S T", recap="")

The system prompt is the same for every request; the message carries what changes -- the letters, a line of what
happened before (`chat.py`), the request. Both are read again on every model call of a request, and what the model
writes costs far more than what it reads (STATUS-kiwi §6). Hence the way to a word it is given: many guesses in one
check, which code answers, rather than one at a time.

The wording is tried, not chosen: whether the model makes a tool call or writes it out as text turns on it, down to
the last sentence, and an example it is shown it copies, labels and all (STATUS-kiwi §6).
`agenttools/agent-raw.py --system FILE` tries another wording on the board.

The model is Qwen2.5-7B-Instruct: it cannot see the letters in a word -- its tokens are pieces of words -- nor
count them, so the letters are written a space apart, and every count is code's (`words.py`).
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are the voice of a robot arm at an NXP i.MX95 demo. The arm sets words down with letter tiles.
Every request asks for a word made of the tiles, unless it asks to explain or elaborate. "go", "run", "start" or \
"execute" mean: the longest word you can find.
For a word, first call find_word with 6 different English words you think the tiles make, longest first. For a clue \
or a topic, like an animal, guess words that match it. If the user says not to place the word, or only asks whether \
it fits, set place to false.
Example: tiles A C E R S T, the user says "go": find_word(guesses="CRATES TRACES CARTS CREST STARE CARE").
find_word sets down the longest real word that fits. Then tell the user the word and, in one sentence, what it \
means. If none fit, call find_word once more with shorter words.
To explain or elaborate, call no tool: spell the last word letter by letter, say what it means, and use it in a \
sentence, as at a spelling bee.
Call detect_tiles only when the user asks to look at the tiles again, or no tiles are known.
Answer in one or two short sentences. Write the word itself in capitals."""


def compose(request: str, desk: str, recap: str) -> str:
    """The message the model reads for a request: the desk, what happened before, then the request itself."""
    lines = [desk] + ([recap] if recap else []) + [f'The user says: "{" ".join(request.split())}"']
    return "\n".join(lines)
