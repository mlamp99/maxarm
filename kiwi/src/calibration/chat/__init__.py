"""The Chat page: a request to the language model on the Ara240, which finds a word of the tiles and sets it down.

| module | what it owns                                                                             |
| ------ | ---------------------------------------------------------------------------------------- |
| `api`  | the page's requests: a request to the model, stopping it, a new visitor; its state       |

The model, its tools and the conversation are `applib/agent`, the /IOTCONNECT app's `agent` command too; the tiles
are the Tiles page's (`tiles/api.py`). The page is in `web/chat/`.
"""
