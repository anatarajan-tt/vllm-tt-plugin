# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Monkeypatches for structured-output request handling (vLLM 0.30).

Each ``_install_*`` function fixes one upstream vLLM defect and follows the
conventions of ``tool_call_fixes``: it is idempotent through a
``_tt_original_*`` marker on the patched owner, and it raises when its target is
missing, so a vLLM upgrade that moves the target fails at startup.

- S1: a request with no structured-output constraint fails with HTTP 400 "You
  must use one kind of structured outputs constraint but none are specified".
  Two shapes reach it: ``response_format={"type": "json_schema",
  "json_schema": {...}}`` without ``"schema"`` (vLLM builds
  ``StructuredOutputsParams(json=None)``), and a client ``structured_outputs``
  that is ``{}`` or holds only options such as ``disable_any_whitespace``. The
  first now means JSON mode (``json_object=True``), the second means no
  constraint. Requests that carry a constraint are unchanged.
- S2: with ``--structured-outputs-config '{"backend": "xgrammar", ...}'`` a
  schema xgrammar cannot handle fails the request: HTTP 400 when the frontend
  check rejects it ("features not supported by xgrammar", converter errors), or
  HTTP 500 when the engine-side compile raises (``grammar_functor.cc``
  ``per_rule_fsm_hashes`` check). The frontend check cannot see the second
  kind: it needs the tokenizer-level compile, which can take minutes.
  Backend "auto" is no way out: vLLM refuses ``disable_any_whitespace`` with
  it, and it does not fall back after a compile error. Such a request is now
  compiled by the guidance backend (llguidance) instead, for that request only:
  the frontend routes the first kind to guidance, and the engine recompiles the
  second kind with guidance. When guidance rejects a schema the frontend
  routed, the frontend returns a clean 400 that names both errors. A schema
  whose xgrammar compile fails and that guidance also rejects still fails in
  the engine (HTTP 500); none of JSONSchemaBench's 738 schemas does.
- S3 (opt in, ``TT_SO_SCHEMA_IN_PROMPT=1``): the grammar alone does not tell the
  model what to write. With thinking on, a strict ``array of enum`` without
  ``maxItems`` loops on one item until ``max_tokens``, and ``json_object``
  prompts that never ask for JSON sometimes return ``{}``. The schema (or a
  "respond with a JSON object" instruction) is appended to the system message
  before the chat template runs. It is off by default because it changes the
  prompt the model sees: it costs prompt tokens (prefill time, prefix-cache
  keys change for every request that uses it), and the reply can differ from
  the reply to the same request without it.

TODO: remove each patch once the matching upstream fix is in the pinned vLLM.
"""

import functools
import json
import os
import threading
from typing import Any

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

SCHEMA_IN_PROMPT_ENV = "TT_SO_SCHEMA_IN_PROMPT"
SCHEMA_IN_PROMPT_MAX_CHARS_ENV = "TT_SO_SCHEMA_IN_PROMPT_MAX_CHARS"
_DEFAULT_SCHEMA_IN_PROMPT_MAX_CHARS = 16384


def install_structured_output_fixes() -> None:
    """Install the structured-output patches. Safe to call more than once."""
    _install_empty_constraint_patch()
    _install_xgrammar_fallback_patch()
    _install_schema_in_prompt_patch()
    logger.info(
        "structured-output fixes active: S1(empty constraint) "
        "S2(xgrammar->guidance per request) S3(schema in prompt, %s=%s)",
        SCHEMA_IN_PROMPT_ENV,
        "on" if _schema_in_prompt_enabled() else "off",
    )


# ---------------------------------------------------------------------------
# S1: a request without any constraint is unconstrained, not a 400.
# ---------------------------------------------------------------------------


def _schema_missing(response_format: Any) -> bool:
    """A ``json_schema`` response format whose ``json_schema`` has no schema."""
    if getattr(response_format, "type", None) != "json_schema":
        return False
    json_schema = getattr(response_format, "json_schema", None)
    return json_schema is not None and json_schema.json_schema is None


def _install_empty_constraint_patch() -> None:
    from vllm.entrypoints.generate.base import protocol as base_protocol
    from vllm.entrypoints.openai.chat_completion import protocol as chat_protocol
    from vllm.entrypoints.openai.completion import protocol as completion_protocol
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    if hasattr(base_protocol, "_tt_original_structured_outputs_from_response_format"):
        return

    # Every target is checked before anything is replaced, so a vLLM that moved
    # one of them fails here with nothing half installed.
    for owner, attr in (
        (base_protocol, "structured_outputs_from_response_format"),
        (base_protocol, "ResponseFormat"),
        (chat_protocol, "structured_outputs_from_response_format"),
        (completion_protocol, "structured_outputs_from_response_format"),
        (StructuredOutputsParams, "__post_init__"),
        (StructuredOutputsParams, "all_constraints_none"),
        (SamplingParams, "_validate_structured_outputs"),
    ):
        if not hasattr(owner, attr):
            raise RuntimeError(
                f"S1 target {getattr(owner, '__name__', owner)}.{attr} is missing"
            )
    original_convert = base_protocol.structured_outputs_from_response_format
    original_post_init = StructuredOutputsParams.__post_init__
    original_validate = SamplingParams._validate_structured_outputs
    for module in (chat_protocol, completion_protocol):
        if module.structured_outputs_from_response_format is not original_convert:
            raise RuntimeError(
                f"{module.__name__}.structured_outputs_from_response_format is not "
                "the base protocol function; S1 would miss that entry point"
            )

    @functools.wraps(original_post_init)
    def __post_init__(self) -> None:
        # Parsing a client ``structured_outputs`` builds the dataclass, and the
        # upstream check rejects one with no constraint before any request code
        # runs. Let it through here; every frontend path drops it (below) before
        # it can reach the engine. Two constraints still raise.
        if self.all_constraints_none():
            return
        original_post_init(self)

    @functools.wraps(original_convert)
    def structured_outputs_from_response_format(structured_outputs, response_format):
        if _schema_missing(response_format):
            # A json_schema format with no schema asks for JSON and nothing
            # more: JSON mode. A constraint the request already carries wins.
            if structured_outputs is not None and not (
                structured_outputs.all_constraints_none()
            ):
                result = structured_outputs
            else:
                result = original_convert(
                    structured_outputs,
                    base_protocol.ResponseFormat(type="json_object"),
                )
        else:
            result = original_convert(structured_outputs, response_format)
        if result is not None and result.all_constraints_none():
            return None
        return result

    @functools.wraps(original_validate)
    def _validate_structured_outputs(self, *args, **kwargs):
        # The catch-all for entry points that build SamplingParams from a
        # client ``structured_outputs`` without the function above (the
        # Responses API, offline ``LLM``): no constraint means no structured
        # output, so the TT runner never sees a structured row with no grammar.
        so = self.structured_outputs
        if so is not None and so.all_constraints_none():
            self.structured_outputs = None
            return None
        return original_validate(self, *args, **kwargs)

    base_protocol._tt_original_structured_outputs_from_response_format = (
        original_convert
    )
    StructuredOutputsParams._tt_original_post_init = original_post_init
    SamplingParams._tt_original_validate_structured_outputs_empty = original_validate
    base_protocol.structured_outputs_from_response_format = (
        structured_outputs_from_response_format
    )
    chat_protocol.structured_outputs_from_response_format = (
        structured_outputs_from_response_format
    )
    completion_protocol.structured_outputs_from_response_format = (
        structured_outputs_from_response_format
    )
    StructuredOutputsParams.__post_init__ = __post_init__
    SamplingParams._validate_structured_outputs = _validate_structured_outputs


# ---------------------------------------------------------------------------
# S2: fall back from xgrammar to guidance for one request.
# ---------------------------------------------------------------------------


def _xgrammar_rejected(err: BaseException, validate_fn: Any) -> bool:
    """Whether ``err`` was raised inside ``validate_xgrammar_grammar``.

    ``_validate_structured_outputs`` raises the same exception type for
    requests that no backend can serve (an empty ``choice`` list, an empty
    grammar, a diffusion model); only an xgrammar rejection may fall back.
    """
    code = validate_fn.__code__
    tb = err.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is code:
            return True
        tb = tb.tb_next
    return False


def _install_xgrammar_fallback_patch() -> None:
    import vllm.sampling_params as sampling_params_module
    import vllm.v1.structured_output as so_manager_module
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams
    from vllm.v1.structured_output import backend_guidance, backend_xgrammar

    manager_cls = so_manager_module.StructuredOutputManager
    if hasattr(SamplingParams, "_tt_original_validate_structured_outputs_fallback"):
        return

    for owner, attr in (
        (SamplingParams, "_validate_structured_outputs"),
        (sampling_params_module, "_get_llg_tokenizer"),
        (sampling_params_module, "_is_non_tekken_mistral"),
        (backend_xgrammar, "validate_xgrammar_grammar"),
        (backend_xgrammar, "XgrammarBackend"),
        (backend_guidance, "validate_guidance_grammar"),
        (backend_guidance, "GuidanceBackend"),
        (manager_cls, "grammar_init"),
        (manager_cls, "_create_grammar"),
        (so_manager_module, "XgrammarBackend"),
    ):
        if not hasattr(owner, attr):
            raise RuntimeError(
                f"S2 target {getattr(owner, '__name__', owner)}.{attr} is missing"
            )
    original_validate = SamplingParams._validate_structured_outputs
    original_grammar_init = manager_cls.grammar_init
    original_create_grammar = manager_cls._create_grammar
    xgrammar_backend_cls = backend_xgrammar.XgrammarBackend
    guidance_backend_cls = backend_guidance.GuidanceBackend

    # -- frontend: choose the backend for this request ----------------------

    def fall_back(self, so, tokenizer, reason: BaseException) -> None:
        if sampling_params_module._is_non_tekken_mistral(tokenizer):
            raise reason
        so._backend = "guidance"
        try:
            backend_guidance.validate_guidance_grammar(
                self, tokenizer=sampling_params_module._get_llg_tokenizer(tokenizer)
            )
        except VLLMValidationError as guidance_err:
            so._backend = "xgrammar"
            raise VLLMValidationError(
                f"{reason} The guidance backend cannot compile it either: "
                f"{guidance_err}"
            ) from reason
        so.__post_init__()
        logger.info(
            "structured output: xgrammar cannot compile this request (%s); "
            "using guidance for it",
            str(reason).splitlines()[0][:200] if str(reason) else type(reason),
        )

    @functools.wraps(original_validate)
    def _validate_structured_outputs(self, model_config, so_config, tokenizer):
        so = self.structured_outputs
        if (
            so is None
            or so_config is None
            or so_config.backend != "xgrammar"
            or so.all_constraints_none()
        ):
            return original_validate(self, model_config, so_config, tokenizer)
        if so._backend == "guidance":
            # Validated before (an ``n > 1`` fan-out or a resubmission reuses
            # the params). Upstream would reject the backend mismatch.
            backend_guidance.validate_guidance_grammar(
                self, tokenizer=sampling_params_module._get_llg_tokenizer(tokenizer)
            )
            return None
        try:
            original_validate(self, model_config, so_config, tokenizer)
        except VLLMValidationError as err:
            if not _xgrammar_rejected(err, backend_xgrammar.validate_xgrammar_grammar):
                raise
            fall_back(self, so, tokenizer, err)
        return None

    # -- engine: compile with the request's backend ------------------------

    # Grammars compile on the manager's thread pool. ``GuidanceBackend`` keeps
    # the grammar it is compiling on ``self.serialized_grammar`` between two
    # statements, so concurrent compiles on one instance can swap grammars;
    # guidance compiles are short (<0.2 s on JSONSchemaBench), so serialize them.
    guidance_lock = threading.Lock()

    def guidance_backend(self):
        backend = self.__dict__.get("_tt_guidance_backend")
        if backend is None:
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            backend = guidance_backend_cls(
                self.vllm_config, tokenizer=self.tokenizer, vocab_size=vocab_size
            )
            # Guidance grammars fill rows of the xgrammar backend's bitmask
            # (``ceil(vocab_size / 32)`` int32 words, the layout the TT device
            # mask consumes). A guidance vocabulary wider than that would write
            # past the row.
            words = (vocab_size + 31) // 32
            guidance_words = (backend.ll_tokenizer.vocab_size + 31) // 32
            if guidance_words != words:
                raise RuntimeError(
                    f"guidance needs {guidance_words} bitmask words per row but "
                    f"the xgrammar bitmask has {words}; cannot fall back"
                )
            self._tt_guidance_backend = backend
        return backend

    def compile_with_guidance(self, request):
        """Compile ``request`` with guidance; raise if llguidance rejects it.

        ``GuidanceBackend.compile_grammar`` only logs a schema error and returns
        a matcher in the error state, which then allows nothing but EOS. Here
        that is a failure, so the caller can surface the original error.
        """
        request_type, grammar_spec = (
            request.structured_output_request.structured_output_key
        )
        stop_token_ids = (
            request.sampling_params.all_stop_token_ids
            if request.sampling_params is not None
            else None
        )
        with guidance_lock:
            grammar = guidance_backend(self).compile_grammar(
                request_type, grammar_spec, stop_token_ids=stop_token_ids
            )
        error = grammar.ll_matcher.get_error()
        if error:
            raise ValueError(f"guidance cannot compile the grammar: {error}")
        return grammar

    @functools.wraps(original_grammar_init)
    def grammar_init(self, request) -> None:
        # The manager builds its one backend from the first structured request.
        # A guidance-routed request must not make guidance that backend: build
        # the configured xgrammar backend first, and the bitmask stays its shape.
        so_req = request.structured_output_request
        if (
            self.backend is None
            and so_req is not None
            and so_req.params._backend == "guidance"
            and self.vllm_config.structured_outputs_config.backend == "xgrammar"
        ):
            self.backend = so_manager_module.XgrammarBackend(
                self.vllm_config,
                tokenizer=self.tokenizer,
                vocab_size=self.vllm_config.model_config.get_vocab_size(),
            )
        return original_grammar_init(self, request)

    @functools.wraps(original_create_grammar)
    def _create_grammar(self, request):
        if not isinstance(self.backend, xgrammar_backend_cls):
            return original_create_grammar(self, request)
        so = request.structured_output_request.params
        if so._backend == "guidance":
            return compile_with_guidance(self, request)
        try:
            return original_create_grammar(self, request)
        except Exception as err:
            # The frontend cannot see compile-time failures (the tokenizer-level
            # FSM build), and without this they are an HTTP 500.
            try:
                grammar = compile_with_guidance(self, request)
            except Exception:
                logger.warning(
                    "structured output: guidance cannot compile request %s "
                    "either after xgrammar failed",
                    request.request_id,
                )
                raise err from None
            so._backend = "guidance"
            logger.warning(
                "structured output: xgrammar failed to compile request %s (%s); "
                "compiled it with guidance",
                request.request_id,
                str(err).splitlines()[0][:200] if str(err) else type(err).__name__,
            )
            return grammar

    SamplingParams._tt_original_validate_structured_outputs_fallback = original_validate
    manager_cls._tt_original_grammar_init = original_grammar_init
    manager_cls._tt_original_create_grammar = original_create_grammar
    SamplingParams._validate_structured_outputs = _validate_structured_outputs
    manager_cls.grammar_init = grammar_init
    manager_cls._create_grammar = _create_grammar


# ---------------------------------------------------------------------------
# S3: tell the model about the schema (opt in).
# ---------------------------------------------------------------------------


def _schema_in_prompt_enabled() -> bool:
    return os.environ.get(SCHEMA_IN_PROMPT_ENV, "0") not in ("", "0")


def _schema_in_prompt_max_chars() -> int:
    raw = os.environ.get(SCHEMA_IN_PROMPT_MAX_CHARS_ENV, "")
    try:
        return int(raw) if raw else _DEFAULT_SCHEMA_IN_PROMPT_MAX_CHARS
    except ValueError:
        return _DEFAULT_SCHEMA_IN_PROMPT_MAX_CHARS


def _message_texts(message: Any) -> list[str]:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return [content]
    texts = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return texts


def schema_instruction(response_format: Any, messages: list[Any]) -> str | None:
    """The instruction S3 adds for this request, or None to leave it alone."""
    rf_type = getattr(response_format, "type", None)
    if rf_type == "json_schema":
        json_schema = response_format.json_schema
        schema = json_schema.json_schema if json_schema is not None else None
        if schema is None:
            rf_type = "json_object"  # S1 serves it as JSON mode
        else:
            text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            if len(text) > _schema_in_prompt_max_chars():
                return (
                    "Respond with a single JSON object that conforms to the JSON "
                    "schema the request specifies."
                )
            header = (
                "Respond with a single JSON object that conforms to this JSON schema"
            )
            if json_schema.name:
                header += f' ("{json_schema.name}"'
                if json_schema.description:
                    header += f": {json_schema.description}"
                header += ")"
            return f"{header}:\n{text}"
    if rf_type == "json_object":
        # OpenAI requires the word JSON in the conversation for JSON mode; a
        # request that has it already carries its own instruction.
        if any(
            "json" in text.lower()
            for message in messages
            for text in _message_texts(message)
        ):
            return None
        return "Respond with a single JSON object."
    return None


def with_schema_instruction(messages: list[Any], instruction: str) -> list[Any]:
    """``messages`` with ``instruction`` appended to the leading system message.

    The input list and its messages are not modified. A conversation without a
    leading system message gets a new one.
    """
    out = list(messages)
    first = out[0] if out else None
    if isinstance(first, dict) and first.get("role") in ("system", "developer"):
        content = first.get("content")
        if isinstance(content, str):
            new_content: Any = f"{content}\n\n{instruction}" if content else instruction
        elif isinstance(content, list):
            new_content = [*content, {"type": "text", "text": f"\n\n{instruction}"}]
        else:
            new_content = instruction
        out[0] = {**first, "content": new_content}
        return out
    return [{"role": "system", "content": instruction}, *out]


def _install_schema_in_prompt_patch() -> None:
    from vllm.renderers.online_renderer import OnlineRenderer

    if hasattr(OnlineRenderer, "_tt_original_render_chat"):
        return
    if not hasattr(OnlineRenderer, "render_chat"):
        raise RuntimeError("S3 target OnlineRenderer.render_chat is missing")
    original = OnlineRenderer.render_chat

    @functools.wraps(original)
    async def render_chat(self, request, *args, **kwargs):
        # Checked per request, so the flag is the env of the API server process.
        if _schema_in_prompt_enabled() and not request.tools:
            instruction = schema_instruction(request.response_format, request.messages)
            if instruction is not None:
                request.messages = with_schema_instruction(
                    request.messages, instruction
                )
        return await original(self, request, *args, **kwargs)

    OnlineRenderer._tt_original_render_chat = original
    OnlineRenderer.render_chat = render_chat


__all__ = [
    "SCHEMA_IN_PROMPT_ENV",
    "install_structured_output_fixes",
    "schema_instruction",
    "with_schema_instruction",
]
