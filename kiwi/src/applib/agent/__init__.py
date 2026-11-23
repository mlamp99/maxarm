"""The language model on the Ara240, and what it may do with the tiles: both apps' -- the /IOTCONNECT app's `agent`
command and the calibration app's Chat page.

    source = AraConnector(on_change=...)
    agent = AgentService(source, Desk(seen=..., detect=..., check=word_check(work, settings), place=...))

| module      | what it owns                                                                            |
| ----------- | --------------------------------------------------------------------------------------- |
| `service`   | `AgentService`: a request answered by the model, the tiles its tools                    |
| `connector` | where the model is -- the Ara240's connector -- how far it has got, and its sampling    |
| `prompt`    | what the model is told: the system prompt, and the message each request becomes        |
| `words`     | words checked against the tiles' letters, in code                                       |
| `judge`     | whether a guess is a word, asked of the model a word at a time; a dictionary, later     |
| `repair`    | a tool call the model wrote out as text, made the call it meant                         |
| `chat`      | the conversation: in full for the pages, a line of it for the model                     |
| `desk`      | what the tools do with the tiles, as each app does it                                   |

strands-agents is needed for `AgentService` alone (`IS_STRANDS_AVAILABLE`); without it the apps run, no agent.
"""

from .connector import AraConnector, describe_status
from .desk import Desk, word_check
from .service import IS_STRANDS_AVAILABLE, AgentService, describe_unready

__all__ = ["AgentService", "AraConnector", "Desk", "IS_STRANDS_AVAILABLE", "describe_status", "describe_unready",
           "word_check"]
