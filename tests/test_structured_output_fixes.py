# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Host tests for the structured-output monkeypatches (S1 to S3).

The request tests drive vLLM's own request models (``ChatCompletionRequest``,
``CompletionRequest``) and ``SamplingParams`` validation. The engine-side S2
tests compile real grammars with xgrammar and llguidance; they need a Qwen
tokenizer (``TT_SO_TEST_TOKENIZER``: a model directory or a ``tokenizer.json``,
default ``QWEN38_TOKENIZER_PATH``) and skip without one. Each "upstream" test
puts the vLLM functions back first, to show the defect the patch fixes.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
import torch

import vllm_tt_plugin.structured_output_fixes as fixes

TOKENIZER_PATH = os.environ.get(
    "TT_SO_TEST_TOKENIZER",
    os.environ.get("QWEN38_TOKENIZER_PATH", "/mnt/tt-data/sdawle/models/Qwen3.8-27B"),
)
# Large enough for any Qwen tokenizer; the bitmask width follows it.
VOCAB_SIZE = 151936

MESSAGES = [{"role": "user", "content": "Describe the guest list."}]


@pytest.fixture(autouse=True)
def installed_fixes():
    # tests/conftest.py restores the patched attributes after every test.
    fixes.install_structured_output_fixes()


def restore_upstream() -> None:
    """Put the upstream functions back, to show what the old code does."""
    import vllm.v1.structured_output as so_manager_module
    from vllm.entrypoints.generate.base import protocol as base_protocol
    from vllm.entrypoints.openai.chat_completion import protocol as chat_protocol
    from vllm.entrypoints.openai.completion import protocol as completion_protocol
    from vllm.renderers.online_renderer import OnlineRenderer
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    convert = base_protocol._tt_original_structured_outputs_from_response_format
    for module in (base_protocol, chat_protocol, completion_protocol):
        module.structured_outputs_from_response_format = convert
    StructuredOutputsParams.__post_init__ = (
        StructuredOutputsParams._tt_original_post_init
    )
    SamplingParams._validate_structured_outputs = (
        SamplingParams._tt_original_validate_structured_outputs_empty
    )
    manager = so_manager_module.StructuredOutputManager
    manager.grammar_init = manager._tt_original_grammar_init
    manager._create_grammar = manager._tt_original_create_grammar
    OnlineRenderer.render_chat = OnlineRenderer._tt_original_render_chat


def chat_request(**fields):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    return ChatCompletionRequest.model_validate(
        {"model": "m", "messages": MESSAGES, **fields}
    )


def chat_so(**fields):
    return chat_request(**fields).to_sampling_params(16, {}).structured_outputs


def completion_so(**fields):
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    request = CompletionRequest.model_validate({"model": "m", "prompt": "x", **fields})
    return request.to_sampling_params(16, {}).structured_outputs


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}
RF_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "person", "strict": True, "schema": SCHEMA},
}
RF_NO_SCHEMA = {"type": "json_schema", "json_schema": {"name": "person"}}
RF_NO_SCHEMA_STRICT = {
    "type": "json_schema",
    "json_schema": {"name": "person", "strict": True, "description": "a person"},
}


# ---------------------------------------------------------------------------
# S1: empty constraints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rf", [RF_NO_SCHEMA, RF_NO_SCHEMA_STRICT])
def test_json_schema_without_schema_was_a_400_upstream(rf):
    from vllm.exceptions import VLLMValidationError

    restore_upstream()
    with pytest.raises(VLLMValidationError, match="none are specified"):
        chat_so(response_format=rf)


@pytest.mark.parametrize("rf", [RF_NO_SCHEMA, RF_NO_SCHEMA_STRICT])
def test_json_schema_without_schema_is_json_mode(rf):
    so = chat_so(response_format=rf)
    assert so.json_object is True
    assert so.json is None
    so = completion_so(response_format=rf)
    assert so.json_object is True


def test_json_schema_without_schema_keeps_an_explicit_constraint():
    so = chat_so(response_format=RF_NO_SCHEMA, structured_outputs={"regex": "[a-z]+"})
    assert so.regex == "[a-z]+"
    assert so.json_object is None


EMPTY_STRUCTURED_OUTPUTS = [
    {},
    {"disable_any_whitespace": True},
    {"disable_additional_properties": True, "whitespace_pattern": " "},
]


@pytest.mark.parametrize("structured_outputs", EMPTY_STRUCTURED_OUTPUTS)
def test_empty_structured_outputs_was_a_400_upstream(structured_outputs):
    from vllm.exceptions import VLLMValidationError

    restore_upstream()
    with pytest.raises(VLLMValidationError, match="none are specified"):
        chat_so(structured_outputs=structured_outputs)


@pytest.mark.parametrize("structured_outputs", EMPTY_STRUCTURED_OUTPUTS)
def test_empty_structured_outputs_is_unconstrained(structured_outputs):
    assert chat_so(structured_outputs=structured_outputs) is None
    assert completion_so(structured_outputs=structured_outputs) is None


def test_empty_structured_outputs_with_a_schema_format_uses_the_schema():
    so = chat_so(structured_outputs={}, response_format=RF_SCHEMA)
    assert so.json == SCHEMA


def test_sampling_params_drop_an_empty_constraint():
    # The catch-all for entry points that bypass the request conversion.
    from vllm.config import StructuredOutputsConfig
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(disable_any_whitespace=True)
    )
    params._validate_structured_outputs(
        SimpleNamespace(is_diffusion=False),
        StructuredOutputsConfig(backend="xgrammar", disable_any_whitespace=True),
        object(),
    )
    assert params.structured_outputs is None


def test_two_constraints_still_fail():
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import StructuredOutputsParams

    with pytest.raises(VLLMValidationError, match="multiple are specified"):
        StructuredOutputsParams(json=SCHEMA, regex="x")
    with pytest.raises(Exception, match="one kind of constraints"):
        chat_so(structured_outputs={"json": SCHEMA, "regex": "x"})


@pytest.mark.parametrize("patched", [False, True])
def test_requests_with_a_constraint_are_unchanged(patched):
    """The same results with and without the patches."""
    if not patched:
        restore_upstream()
    so = chat_so(response_format=RF_SCHEMA)
    assert (so.json, so.json_object) == (SCHEMA, None)
    so = chat_so(response_format={"type": "json_object"})
    assert (so.json, so.json_object) == (None, True)
    assert chat_so(response_format={"type": "text"}) is None
    assert chat_so() is None
    so = chat_so(structured_outputs={"json": SCHEMA, "disable_any_whitespace": True})
    assert so.json == SCHEMA and so.disable_any_whitespace is True
    so = chat_so(structured_outputs={"choice": ["a", "b"]})
    assert so.choice == ["a", "b"]
    so = completion_so(response_format=RF_SCHEMA)
    assert so.json == SCHEMA
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": SCHEMA},
        }
    ]
    request = chat_request(tools=tools, tool_choice="auto")
    assert request.to_sampling_params(16, {}).structured_outputs is None
    assert request.messages == MESSAGES


# ---------------------------------------------------------------------------
# S2 frontend: choose guidance when xgrammar rejects the schema
# ---------------------------------------------------------------------------


def verify(constraint: dict, backend: str = "xgrammar"):
    """Run vLLM's request-level structured-output validation; return the params."""
    from vllm.config import StructuredOutputsConfig
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    params = SamplingParams(structured_outputs=StructuredOutputsParams(**constraint))
    params._validate_structured_outputs(
        SimpleNamespace(is_diffusion=False),
        StructuredOutputsConfig(
            backend=backend, disable_any_whitespace=backend != "auto"
        ),
        object(),  # an HF tokenizer: guidance validates without a tokenizer
    )
    return params.structured_outputs


def obj(prop: dict) -> dict:
    return {"type": "object", "properties": {"v": prop}, "required": ["v"]}


# Schemas xgrammar rejects (HTTP 400 upstream) that guidance compiles.
GUIDANCE_ONLY = {
    "multipleOf": obj({"type": "integer", "multipleOf": 5}),
    "patternProperties with properties": {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "patternProperties": {"^x-[a-z]+$": {"type": "string"}},
    },
    "two patternProperties, complex regex": {
        "type": "object",
        "patternProperties": {
            "^(?:[a-z]{2,3}-)?[A-Z][a-zA-Z0-9_]*$": {"type": "integer"},
            "^\\d+$": {"type": "string"},
        },
    },
    "pattern with maxLength": obj(
        {"type": "string", "pattern": "^[a-z]+$", "maxLength": 5}
    ),
    "integer bound above int64": obj({"type": "integer", "maximum": 1e20}),
    "uint64 maximum": obj(
        {"type": "integer", "minimum": 0, "maximum": 18446744073709551615}
    ),
}

# Schemas neither backend serves: a clean 400.
NEITHER = {
    "uniqueItems": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    "contains": {"type": "array", "contains": {"type": "integer"}},
    "unknown format": obj({"type": "string", "format": "textarea"}),
    "minItems above prefixItems": {
        "type": "array",
        "prefixItems": [{"type": "string"}],
        "minItems": 2,
        "items": False,
    },
}


@pytest.mark.parametrize("schema", GUIDANCE_ONLY.values(), ids=GUIDANCE_ONLY.keys())
def test_xgrammar_rejection_was_a_400_upstream(schema):
    from vllm.exceptions import VLLMValidationError

    restore_upstream()
    with pytest.raises(VLLMValidationError):
        verify({"json": schema})


@pytest.mark.parametrize("schema", GUIDANCE_ONLY.values(), ids=GUIDANCE_ONLY.keys())
def test_xgrammar_rejection_falls_back_to_guidance(schema):
    so = verify({"json": schema})
    assert so._backend == "guidance"
    assert so.json == schema


@pytest.mark.parametrize("schema", NEITHER.values(), ids=NEITHER.keys())
def test_schema_neither_backend_compiles_is_a_400(schema):
    from vllm.exceptions import VLLMValidationError

    with pytest.raises(VLLMValidationError, match="guidance backend cannot"):
        verify({"json": schema})


@pytest.mark.parametrize(
    "constraint",
    [
        {"json": SCHEMA},
        {"json": json.dumps(SCHEMA)},
        {"json_object": True},
        {"regex": "[a-z]{1,8}"},
        {"choice": ["yes", "no"]},
        {"grammar": 'root ::= "a" | "b"'},
        {"json": obj({"type": "string", "enum": ["guest", "host"]})},
    ],
)
def test_supported_requests_stay_on_xgrammar(constraint):
    assert verify(constraint)._backend == "xgrammar"


@pytest.mark.parametrize(
    ("constraint", "message"),
    [({"choice": []}, "cannot be an empty list"), ({"grammar": "  "}, "empty string")],
)
def test_errors_that_are_not_xgrammar_rejections_do_not_fall_back(constraint, message):
    from vllm.exceptions import VLLMValidationError

    with pytest.raises(VLLMValidationError, match=message):
        verify(constraint)


@pytest.mark.parametrize("backend", ["auto", "guidance"])
def test_other_backends_are_untouched(backend):
    schema = GUIDANCE_ONLY["multipleOf"]
    so = verify({"json": schema}, backend=backend)
    assert so._backend == "guidance"
    assert so._backend_was_auto is (backend == "auto")


def test_guidance_routed_params_verify_again():
    from vllm.config import StructuredOutputsConfig
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json=GUIDANCE_ONLY["multipleOf"])
    )
    config = StructuredOutputsConfig(backend="xgrammar", disable_any_whitespace=True)
    for _ in range(2):
        params._validate_structured_outputs(
            SimpleNamespace(is_diffusion=False), config, object()
        )
    assert params.structured_outputs._backend == "guidance"


# ---------------------------------------------------------------------------
# S2 engine: compile with the request's backend; recover compile crashes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    if os.path.isfile(TOKENIZER_PATH):
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast(
            tokenizer_file=TOKENIZER_PATH, eos_token="<|im_end|>"
        )
    if os.path.isdir(TOKENIZER_PATH):
        from vllm.tokenizers import get_tokenizer

        return get_tokenizer(TOKENIZER_PATH)
    pytest.skip(f"{TOKENIZER_PATH} is not available")


def make_manager(tokenizer):
    """A StructuredOutputManager with just what grammar compilation reads."""
    from vllm.config import StructuredOutputsConfig
    from vllm.v1.structured_output import StructuredOutputManager

    manager = object.__new__(StructuredOutputManager)
    manager.backend = None
    manager.tokenizer = tokenizer
    manager._use_async_grammar_compilation = False
    manager.vllm_config = SimpleNamespace(
        structured_outputs_config=StructuredOutputsConfig(
            backend="xgrammar", disable_any_whitespace=True
        ),
        model_config=SimpleNamespace(get_vocab_size=lambda: VOCAB_SIZE),
        speculative_config=None,
    )
    return manager


def make_request(schema: dict, backend: str = "xgrammar"):
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
    from vllm.v1.structured_output.request import StructuredOutputRequest

    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    params.structured_outputs._backend = backend
    return SimpleNamespace(
        request_id="req-0",
        sampling_params=params,
        structured_output_request=StructuredOutputRequest.from_sampling_params(params),
    )


def compiled(manager, request):
    manager.grammar_init(request)
    # A compile error comes back as the exception (the scheduler then fails the
    # request: FINISHED_ERROR, an HTTP 500).
    grammar = request.structured_output_request.grammar
    if isinstance(grammar, Exception):
        raise grammar
    return grammar


# xgrammar's compile (not its converter) fails on a long fixed repetition with a
# real vocabulary: "grammar_functor.cc ... per_rule_fsm_hashes ... has_value()".
# Minimised from JSONSchemaBench Github_easy o40228.
FSM_HASH_CRASH = {"type": "string", "pattern": "[a-z]{162}"}


def test_compile_crash_was_a_500_upstream(tokenizer):
    restore_upstream()
    manager = make_manager(tokenizer)
    with pytest.raises(RuntimeError, match="per_rule_fsm_hashes"):
        compiled(manager, make_request(FSM_HASH_CRASH))


def test_compile_crash_falls_back_to_guidance(tokenizer):
    from vllm.v1.structured_output.backend_guidance import GuidanceGrammar
    from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

    manager = make_manager(tokenizer)
    request = make_request(FSM_HASH_CRASH)
    grammar = compiled(manager, request)
    assert isinstance(grammar, GuidanceGrammar)
    assert isinstance(manager.backend, XgrammarBackend)
    assert request.sampling_params.structured_outputs._backend == "guidance"
    # The grammar is live: the first token must open the string.
    bitmask = manager.backend.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)
    assert allowed(bitmask[0], tokenizer.convert_tokens_to_ids('"'))
    assert not allowed(bitmask[0], tokenizer.convert_tokens_to_ids("{"))


def test_compile_failure_of_both_backends_keeps_the_xgrammar_error(tokenizer):
    # The frontend rejects uniqueItems before it gets here; this is the engine
    # guard for a schema that reaches it anyway.
    schema = {
        "type": "object",
        "properties": {
            "a": FSM_HASH_CRASH,
            "b": {"type": "array", "uniqueItems": True},
        },
    }
    manager = make_manager(tokenizer)
    with pytest.raises(RuntimeError, match="per_rule_fsm_hashes"):
        compiled(manager, make_request(schema))


def test_guidance_routed_request_compiles_with_guidance(tokenizer):
    from vllm.v1.structured_output.backend_guidance import GuidanceGrammar
    from vllm.v1.structured_output.backend_xgrammar import (
        XgrammarBackend,
        XgrammarGrammar,
    )

    manager = make_manager(tokenizer)
    # The first structured request is guidance-routed: the manager's backend
    # must still be xgrammar, for the requests after it.
    grammar = compiled(
        manager, make_request(GUIDANCE_ONLY["multipleOf"], backend="guidance")
    )
    assert isinstance(grammar, GuidanceGrammar)
    assert isinstance(manager.backend, XgrammarBackend)
    assert isinstance(compiled(manager, make_request(SCHEMA)), XgrammarGrammar)


def allowed(row: torch.Tensor, token_id: int) -> bool:
    return bool((int(row[token_id // 32]) >> (token_id % 32)) & 1)


def test_guidance_bitmask_matches_the_tt_mask_layout(tokenizer):
    """Guidance fills the xgrammar-shaped bitmask the TT mask path consumes:
    int32, ``ceil(vocab / 32)`` words, bit ``v % 32`` of word ``v // 32``."""
    import llguidance.torch as llguidance_torch

    from vllm_tt_plugin import spec_masks

    manager = make_manager(tokenizer)
    xgr_grammar = compiled(manager, make_request(SCHEMA))
    ll_grammar = compiled(manager, make_request(SCHEMA, backend="guidance"))
    xgr_mask = manager.backend.allocate_token_bitmask(2)
    assert xgr_mask.dtype == torch.int32
    assert xgr_mask.shape == (2, spec_masks.mask_words(VOCAB_SIZE))
    xgr_grammar.fill_bitmask(xgr_mask, 0)
    ll_grammar.fill_bitmask(xgr_mask, 1)
    # Same layout as llguidance's own allocation.
    own = llguidance_torch.allocate_token_bitmask(1, VOCAB_SIZE)
    assert own.shape[1] == xgr_mask.shape[1] and own.dtype == xgr_mask.dtype
    ll_grammar.fill_bitmask(own, 0)
    assert torch.equal(own[0], xgr_mask[1])
    # Both grammars open the object. (xgrammar also allows "{" alone, while
    # llguidance forces the '{"' it knows must follow.)
    brace_quote = tokenizer.convert_tokens_to_ids('{"')
    word = tokenizer.convert_tokens_to_ids("hello")
    for row in (xgr_mask[0], xgr_mask[1]):
        assert allowed(row, brace_quote)
        assert not allowed(row, word)
    # disable_any_whitespace holds on the guidance path too: no leading space.
    space_brace = tokenizer.convert_tokens_to_ids("\u0120{")
    assert space_brace != tokenizer.unk_token_id
    for row in (xgr_mask[0], xgr_mask[1]):
        assert not allowed(row, space_brace)
    # Ids at or above the tokenizer's length are never allowed.
    for row in (xgr_mask[0], xgr_mask[1]):
        assert not allowed(row, VOCAB_SIZE - 1)
    normalised = spec_masks.normalise_width(
        xgr_mask[1:2], spec_masks.mask_words(VOCAB_SIZE)
    )
    assert torch.equal(normalised, xgr_mask[1:2])


# ---------------------------------------------------------------------------
# S3: schema in the prompt (opt in)
# ---------------------------------------------------------------------------


def rendered_messages(monkeypatch, enabled: bool, **fields):
    """The messages the renderer is handed, through the patched render_chat."""
    from vllm.renderers.online_renderer import OnlineRenderer

    seen = {}

    async def fake_render_chat(self, request, **kwargs):
        seen["messages"] = request.messages
        return "rendered"

    monkeypatch.setenv(fixes.SCHEMA_IN_PROMPT_ENV, "1" if enabled else "0")
    monkeypatch.delattr(OnlineRenderer, "_tt_original_render_chat")
    monkeypatch.setattr(OnlineRenderer, "render_chat", fake_render_chat)
    fixes._install_schema_in_prompt_patch()
    request = chat_request(**fields)
    assert asyncio.run(OnlineRenderer.render_chat(object(), request)) == "rendered"
    return seen["messages"]


TAGS_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string", "enum": ["guest", "host"]},
        }
    },
    "required": ["tags"],
    "additionalProperties": False,
}
RF_TAGS = {
    "type": "json_schema",
    "json_schema": {"name": "tags", "schema": TAGS_SCHEMA},
}


def test_schema_in_prompt_is_off_by_default(monkeypatch):
    monkeypatch.delenv(fixes.SCHEMA_IN_PROMPT_ENV, raising=False)
    assert not fixes._schema_in_prompt_enabled()
    assert rendered_messages(monkeypatch, False, response_format=RF_TAGS) == MESSAGES


def test_schema_goes_into_a_new_system_message(monkeypatch):
    messages = rendered_messages(monkeypatch, True, response_format=RF_TAGS)
    assert messages[1:] == MESSAGES
    assert messages[0]["role"] == "system"
    assert json.dumps(TAGS_SCHEMA, separators=(",", ":")) in messages[0]["content"]
    assert '("tags")' in messages[0]["content"]


def test_schema_is_appended_to_the_system_message(monkeypatch):
    system = {"role": "system", "content": "You are terse."}
    messages = rendered_messages(
        monkeypatch, True, response_format=RF_TAGS, messages=[system, *MESSAGES]
    )
    assert len(messages) == 2 and messages[1:] == MESSAGES
    assert messages[0]["content"].startswith("You are terse.\n\n")
    assert '"enum":["guest","host"]' in messages[0]["content"]
    assert system == {"role": "system", "content": "You are terse."}


def test_schema_is_appended_to_a_system_message_with_parts():
    system = {"role": "system", "content": [{"type": "text", "text": "Be terse."}]}
    out = fixes.with_schema_instruction([system], "Respond with JSON.")
    assert out[0]["content"][0] == {"type": "text", "text": "Be terse."}
    assert out[0]["content"][1]["text"].endswith("Respond with JSON.")
    assert len(system["content"]) == 1


def test_json_object_without_instruction_gets_one(monkeypatch):
    messages = rendered_messages(
        monkeypatch, True, response_format={"type": "json_object"}
    )
    assert messages[0] == {
        "role": "system",
        "content": "Respond with a single JSON object.",
    }


def test_json_object_with_instruction_is_untouched(monkeypatch):
    user = [{"role": "user", "content": "Reply in JSON with a name field."}]
    messages = rendered_messages(
        monkeypatch, True, response_format={"type": "json_object"}, messages=user
    )
    assert messages == user


def test_schema_less_json_schema_is_json_mode_in_the_prompt_too(monkeypatch):
    messages = rendered_messages(monkeypatch, True, response_format=RF_NO_SCHEMA)
    assert messages[0]["content"] == "Respond with a single JSON object."


def test_large_schema_gets_a_short_instruction(monkeypatch):
    monkeypatch.setenv(fixes.SCHEMA_IN_PROMPT_MAX_CHARS_ENV, "10")
    messages = rendered_messages(monkeypatch, True, response_format=RF_TAGS)
    content = messages[0]["content"]
    assert "conforms to the JSON schema the request specifies" in content
    assert "enum" not in content


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"response_format": {"type": "text"}},
        {
            "response_format": RF_TAGS,
            "tools": [
                {"type": "function", "function": {"name": "f", "parameters": SCHEMA}}
            ],
        },
    ],
    ids=["no format", "text", "tools"],
)
def test_other_requests_are_untouched_with_the_flag_on(monkeypatch, fields):
    assert rendered_messages(monkeypatch, True, **fields) == MESSAGES


# ---------------------------------------------------------------------------
# Install conventions
# ---------------------------------------------------------------------------


def test_install_is_idempotent():
    from vllm.entrypoints.openai.chat_completion import protocol as chat_protocol
    from vllm.sampling_params import SamplingParams

    before = (
        SamplingParams._validate_structured_outputs,
        chat_protocol.structured_outputs_from_response_format,
    )
    fixes.install_structured_output_fixes()
    assert before == (
        SamplingParams._validate_structured_outputs,
        chat_protocol.structured_outputs_from_response_format,
    )


def test_missing_target_raises(monkeypatch):
    from vllm.renderers.online_renderer import OnlineRenderer

    monkeypatch.delattr(OnlineRenderer, "_tt_original_render_chat")
    monkeypatch.delattr(OnlineRenderer, "render_chat")
    with pytest.raises(RuntimeError, match="render_chat is missing"):
        fixes._install_schema_in_prompt_patch()


def test_plugin_entry_point_installs_the_fixes():
    import inspect

    from vllm_tt_plugin import model_registry

    source = inspect.getsource(model_registry.register_tt_models_from_plugin)
    assert "install_structured_output_fixes()" in source
