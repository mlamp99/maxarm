"""The language model on the board's Ara240, through NXP's runtime: tokens in, a score for every next token out.

    model = AraModel(MODEL_DIR)
    model.load()                            # minutes: the model's 8 GB go into the Ara240's own memory
    scores = model.begin(prompt_tokens)     # the prompt read; a score per token of the vocabulary for what follows
    scores = model.extend(token)            # that token taken as said; the scores for the one after it
    model.close()

The runtime, rt-sdk-ara2, comes with NXP's image: its proxy holds the device and listens on PROXY_SOCKET, and
`dvapi.py`, its Python binding, lies in SDK_INCLUDE. A model compiled for the Ara240 is a directory, `model.dvm`
with its `generation_config.json` and `tokenizer/` (fetch_models, or NXP's on Hugging Face). None of this is needed
to import the module: `load()` needs it, and says what is missing.

The device keeps what it has read, so a token more costs one step, about 0.16 s, and a prompt about 2 s; it cannot
go back, though: a token read stays read until the next `begin()`. The scores come back whole -- the model's logits
in fixed point, SCORE_SCALE to one -- so that the caller, not the device, chooses each token, and can read from one
look what the model makes of every word it knows (`library.py`). One caller at a time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

MODEL_DIR = Path("/usr/share/llm/Qwen2.5-7B-Instruct")
PROXY_SOCKET = "/var/run/proxy.sock"
SDK_INCLUDE = Path("/usr/share/rt-sdk-ara240/include")
SCORE_SCALE = 2.0 ** 14
# A prompt at the model's full length takes a minute; a token a fraction of a second.
PROMPT_TIMEOUT_MS = 90_000
TOKEN_TIMEOUT_MS = 20_000


class ModelUnavailable(RuntimeError):
    """The model cannot be loaded or has stopped answering; the message says why."""


class ProxyNotReady(ModelUnavailable):
    """The Ara240's proxy is not there to load the model through: after a boot it takes a minute to come up."""


class AraModel:
    def __init__(self, model_dir: Path = MODEL_DIR, proxy_socket: str = PROXY_SOCKET) -> None:
        self.model_dir = Path(model_dir)
        self.proxy_socket = proxy_socket
        self.vocabulary_size = 0
        self.context = 0                    # the most tokens the model reads, prompt and answer together
        self.end_tokens: List[int] = []
        self.read = 0                       # the tokens it has read since `begin()`
        self._dvapi = None
        self._session = None
        self._endpoint = None
        self._model = None
        self._pad = 0

    @property
    def tokenizer_file(self) -> Path:
        return self.model_dir / "tokenizer" / "tokenizer.json"

    def load(self) -> None:
        """Connect to the proxy and load the model onto the device; raises `ModelUnavailable`, or `ProxyNotReady`
        while the proxy is not there to connect to."""
        model_file = self.model_dir / "model.dvm"
        for needed in (model_file, self.tokenizer_file, SDK_INCLUDE / "dvapi.py"):
            if not needed.exists():
                raise ModelUnavailable(f"{needed} is not there")
        if not Path(self.proxy_socket).exists():
            raise ProxyNotReady(f"the Ara240's proxy is not listening on {self.proxy_socket}: is "
                                f"rt-sdk-ara2.service running?")
        if str(SDK_INCLUDE) not in sys.path:
            sys.path.append(str(SDK_INCLUDE))
        try:
            import dvapi
        except (ImportError, OSError) as reason:
            raise ModelUnavailable(f"the Ara240's Python binding does not load: {reason}") from None
        self._dvapi = dvapi
        try:
            self._session = self._checked(dvapi.DVSession.create_via_unix_socket(self.proxy_socket), "the proxy")
            endpoints = self._checked(self._session.get_endpoint_list(), "the Ara240")
        except ModelUnavailable as reason:
            self.close()
            raise ProxyNotReady(str(reason)) from None
        if not endpoints:
            self.close()
            raise ProxyNotReady("the proxy has no Ara240 yet")
        self._endpoint = endpoints[0]
        options = dvapi.DVModelLoadOptions.model_load_option(
            model_name="llm_model", priority=dvapi.dv_model_priority_level.DV_MODEL_PRIORITY_LEVEL_DEFAULT,
            cache=False, model_async=False, model_type=dvapi.dv_model_type.DV_MODEL_TYPE_ARA2_LLM_DYN_V2,
            send_model_filepath_to_proxy=True)
        self._model = self._checked(
            self._session.load_model_from_file_with_options(self._endpoint, str(model_file), options=options),
            f"loading {model_file}")
        parameters = self._model._model.llm_params.contents
        self.vocabulary_size, self.context = int(parameters.vocab_size), int(parameters.max_num_tokens)
        self._pad = int(parameters.pad_token_id)
        generation = json.loads((self.model_dir / "generation_config.json").read_text())
        ends = generation.get("eos_token_id", [])
        self.end_tokens = sorted({int(parameters.eos_token_id), *([ends] if isinstance(ends, int) else ends)})
        # The embeddings looked up on the device (pre), the scores returned whole, not a token chosen there (post).
        chosen_here = dvapi.DVLLMModelCfgUpdateParams.llm_model_cfg_options(
            top_k=1, top_p=1.0, temperature=1.0, repetition_penalty=1.0, target_token_post_mcp=0,
            target_token_pre_mcp=1, target_prompt_post_mcp=0, target_prompt_pre_mcp=1)
        status = dvapi.dv_model_set_llm_cfg_params(
            session=self._session._session, endpoint=self._endpoint._endpoint, model=self._model._model,
            llm_cfg_update=chosen_here._llm_config_options)
        self._checked((status, True), "setting the model up")

    def begin(self, tokens: Sequence[int]) -> np.ndarray:
        """A new prompt, read from its first token; the scores for the token after it."""
        if not 0 < len(tokens) < self.context:
            raise ValueError(f"a prompt is 1 to {self.context - 1} tokens, not {len(tokens)}")
        padded = np.full(self.context, self._pad, dtype=np.int32)
        padded[:len(tokens)] = tokens
        kind = self._dvapi.dv_infer_type.DV_INFER_TYPE_LLM_PROMPT_PROCESSING
        scores = self._infer(padded, kind, len(tokens), len(tokens), PROMPT_TIMEOUT_MS)
        self.read = len(tokens)
        return scores

    def extend(self, token: int) -> np.ndarray:
        """`token` read after what has been; the scores for the token after it."""
        if self.read + 1 >= self.context:
            raise ValueError(f"the model reads {self.context} tokens at most")
        kind = self._dvapi.dv_infer_type.DV_INFER_TYPE_LLM_TOKEN_GENERATION
        scores = self._infer(np.array([token], dtype=np.int32), kind, self.read, 1, TOKEN_TIMEOUT_MS)
        self.read += 1
        return scores

    def close(self) -> None:
        if self._model is not None:
            self._dvapi.dv_model_unload(self._model._model)
        if self._session is not None:
            self._session.close()
        self._model = self._session = self._endpoint = None

    def _infer(self, tokens: np.ndarray, kind: int, read: int, given: int, timeout_ms: int) -> np.ndarray:
        if self._model is None:
            raise ModelUnavailable("the model is not loaded")
        dvapi = self._dvapi
        scores = np.zeros(self.vocabulary_size, dtype=np.int32)
        options = dvapi.DVInferOptions.dv_infer_options(
            enable_stats=False, infer_type=kind, active_tokens=read, valid_tokens=given, tokens_to_skip=0)
        status, request = self._model.infer_sync_with_options(
            input_tensors=[dvapi.DVTensor(tokens, params=None)], output_tensors=[dvapi.DVTensor(scores, params=None)],
            endpoint=self._endpoint._endpoint, timeout_ms=timeout_ms, infer_options=options)
        if request is not None:
            dvapi.dv_infer_free(request)
        self._checked((status, True), "the model's answer")
        return scores

    def _checked(self, result: tuple, what: str) -> Optional[object]:
        """What a dvapi call gave after its status, or `ModelUnavailable` with the status said."""
        status, value = result
        if status != self._dvapi.dv_status_code.DV_SUCCESS or value is None:
            raise ModelUnavailable(f"{what}: {self._dvapi.dv_stringify_status_code(status)}")
        return value
