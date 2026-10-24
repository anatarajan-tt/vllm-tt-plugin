# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Frontend monkeypatches for Qwen tool-call argument errors.

Each ``_install_*`` function fixes one upstream vLLM defect and follows the
conventions of the ``_install_*`` patches in ``platform.py``: it is idempotent
through a ``_tt_original_*`` marker on the patched owner, and it raises when its
target is missing, so a vLLM upgrade that moves the target fails at startup.

- P1: ``$ref``/``allOf`` in tool parameter schemas hide the property types from
  argument coercion (vLLM #46924).
- P2: ``finish_reason`` is "tool_calls" even when generation stopped on
  "length" (vLLM #46303, #59324).
- P3: the final conversion drops a last ``<parameter>`` that has no
  ``</parameter>`` (vLLM #57699).
- P4: a ``["string", ...]`` union turns "007" into 7 (vLLM #62252).

TODO: remove each patch once the matching upstream fix is in the pinned vLLM.
"""

import copy
import functools
import inspect
import json
import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

_MAX_DEREF_DEPTH = 32
_SCHEMA_CACHE_SIZE = 64


def install_tool_call_fixes() -> None:
    """Install the four tool-call patches. Safe to call more than once."""
    _install_tool_schema_deref_patch()
    _install_tool_finish_reason_patch()
    _install_unclosed_parameter_patch()
    _install_string_union_patch()


# ---------------------------------------------------------------------------
# P1: dereference $ref / allOf before coercion looks up property types.
# ---------------------------------------------------------------------------


def _resolve_pointer(root: Any, ref: str) -> Any:
    node = root
    for part in ref[1:].split("/"):
        if part == "":
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _merge_schemas(base: dict, other: dict) -> dict:
    """Merge `other` into a copy of `base`: properties per key, required union."""
    merged = dict(base)
    for key, value in other.items():
        if key not in merged:
            merged[key] = value
        elif key == "properties" and isinstance(value, dict):
            props = dict(merged[key]) if isinstance(merged[key], dict) else {}
            for prop_name, prop_schema in value.items():
                existing = props.get(prop_name)
                if isinstance(existing, dict) and isinstance(prop_schema, dict):
                    props[prop_name] = _merge_schemas(existing, prop_schema)
                else:
                    props.setdefault(prop_name, prop_schema)
            merged[key] = props
        elif key == "required" and isinstance(value, list):
            current = merged[key] if isinstance(merged[key], list) else []
            merged[key] = current + [r for r in value if r not in current]
        # Any other key (including "type") keeps the first value.
    return merged


def _deref_schema(node: Any, root: Any, stack: tuple[str, ...], depth: int = 0) -> Any:
    """Return a copy of `node` with local $ref and allOf resolved."""
    if not isinstance(node, dict) or depth > _MAX_DEREF_DEPTH:
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        rest = {k: v for k, v in node.items() if k != "$ref"}
        target = _resolve_pointer(root, ref) if ref.startswith("#") else None
        if ref in stack or not isinstance(target, dict):
            return _deref_schema(rest, root, stack, depth + 1)
        return _deref_schema({**target, **rest}, root, stack + (ref,), depth + 1)

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "allOf":
            continue
        if key in ("properties", "patternProperties") and isinstance(value, dict):
            out[key] = {
                name: _deref_schema(sub, root, stack, depth + 1)
                for name, sub in value.items()
            }
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            out[key] = _deref_schema(value, root, stack, depth + 1)
        elif key in ("items", "prefixItems", "anyOf", "oneOf") and isinstance(
            value, list
        ):
            out[key] = [_deref_schema(sub, root, stack, depth + 1) for sub in value]
        else:
            out[key] = copy.deepcopy(value)
    members = node.get("allOf")
    if isinstance(members, list):
        for member in members:
            if isinstance(member, dict):
                out = _merge_schemas(out, _deref_schema(member, root, stack, depth + 1))
    return out


def _has_ref_or_all_of(node: Any) -> bool:
    if isinstance(node, dict):
        return (
            "$ref" in node
            or "allOf" in node
            or any(map(_has_ref_or_all_of, node.values()))
        )
    if isinstance(node, list):
        return any(map(_has_ref_or_all_of, node))
    return False


def _install_tool_schema_deref_patch() -> None:
    """Resolve ``$ref``/``allOf`` in the properties the parser engine coerces by.

    ``parser_engine`` binds ``find_tool_properties`` by name at import, so the
    module attribute is the target.
    """
    from vllm.parser.engine import parser_engine
    from vllm.tool_parsers import utils as tool_utils

    if hasattr(parser_engine, "_tt_original_find_tool_properties"):
        return
    original = parser_engine.find_tool_properties
    parser_engine._tt_original_find_tool_properties = original
    _is_function_tool = tool_utils._is_function_tool
    _extract_tool_info = tool_utils._extract_tool_info
    iter_info = tool_utils.iter_response_function_tool_info
    response_tool_types = (tool_utils.FunctionTool, tool_utils.NamespaceTool)
    # id(parameters) -> (parameters, dereferenced properties or None).
    cache: dict[int, tuple[Any, dict | None]] = {}

    def find_parameters(tools: Any, tool_name: str) -> Any:
        for tool in tools or []:
            if isinstance(tool, response_tool_types):
                for name, params in iter_info(tool):
                    if name == tool_name:
                        return params
                continue
            if not _is_function_tool(tool):
                continue
            name, params = _extract_tool_info(tool)
            if name == tool_name:
                return params
        return None

    def find_tool_properties(tools: Any, tool_name: str) -> dict[str, Any]:
        # The schema comes from the client. A malformed one must not fail the
        # request, so dereference errors fall back to the upstream lookup.
        try:
            params = find_parameters(tools, tool_name)
            if isinstance(params, dict):
                hit = cache.get(id(params))
                if hit is None or hit[0] is not params:
                    props = None
                    if _has_ref_or_all_of(params):
                        resolved = _deref_schema(params, params, ())
                        props = resolved.get("properties")
                        props = props if isinstance(props, dict) else {}
                    if len(cache) >= _SCHEMA_CACHE_SIZE:
                        cache.clear()
                    hit = cache[id(params)] = (params, props)
                if hit[1] is not None:
                    return hit[1]
        except Exception as exc:
            logger.debug("Schema dereference failed for %s: %s", tool_name, exc)
        return original(tools, tool_name)

    parser_engine.find_tool_properties = find_tool_properties


# ---------------------------------------------------------------------------
# P2: "tool_calls" only when the engine stopped on "stop".
# ---------------------------------------------------------------------------


async def _aclose(generator: Any) -> None:
    aclose = getattr(generator, "aclose", None)
    if aclose is not None:
        await aclose()


def _tee_finish_reasons(
    result_generator: AsyncIterator[Any], rec: dict[int, str]
) -> AsyncGenerator[Any, None]:
    """Yield the engine outputs unchanged and record each finish_reason.

    Closing the tee closes the engine generator at once. Leaving an async
    ``for`` early does not close the iterated generator by itself.
    """

    async def tee() -> AsyncGenerator[Any, None]:
        try:
            async for res in result_generator:
                for output in res.outputs:
                    if output.finish_reason is not None:
                        rec[output.index] = output.finish_reason
                yield res
        finally:
            await _aclose(result_generator)

    return tee()


def _truthful_reason(rec: dict[int, str], index: Any) -> str | None:
    reason = rec.get(index)
    return reason if reason is not None and reason != "stop" else None


def _fix_sse_chunk(chunk: str, rec: dict[int, str]) -> str:
    if not chunk.startswith("data: ") or '"finish_reason":"tool_calls"' not in chunk:
        return chunk
    obj = json.loads(chunk[len("data: ") :])
    changed = False
    for choice in obj.get("choices", ()):
        if choice.get("finish_reason") != "tool_calls":
            continue
        reason = _truthful_reason(rec, choice.get("index"))
        if reason is not None:
            choice["finish_reason"] = reason
            changed = True
    if not changed:
        return chunk
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return f"data: {payload}\n\n"


def _install_tool_finish_reason_patch() -> None:
    """Report "length" instead of "tool_calls" for a truncated tool call.

    Ports the rule of vLLM #46303/#59324 as a class-attribute patch of the two
    chat generators, which decide ``finish_reason`` inline.
    """
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    if hasattr(OpenAIServingChat, "_tt_original_chat_completion_stream_generator"):
        return
    orig_stream = OpenAIServingChat.chat_completion_stream_generator
    orig_full = OpenAIServingChat.chat_completion_full_generator
    for orig in (orig_stream, orig_full):
        if "result_generator" not in inspect.signature(orig).parameters:
            raise RuntimeError(f"{orig.__name__} has no result_generator parameter")
    OpenAIServingChat._tt_original_chat_completion_stream_generator = orig_stream
    OpenAIServingChat._tt_original_chat_completion_full_generator = orig_full

    @functools.wraps(orig_stream)
    async def stream_generator(self, request, result_generator, *args, **kwargs):
        rec: dict[int, str] = {}
        tee = _tee_finish_reasons(result_generator, rec)
        inner = orig_stream(self, request, tee, *args, **kwargs)
        try:
            async for chunk in inner:
                yield _fix_sse_chunk(chunk, rec)
        finally:
            # A client disconnect closes this generator at the yield above.
            await _aclose(inner)
            await _aclose(tee)

    @functools.wraps(orig_full)
    async def full_generator(self, request, result_generator, *args, **kwargs):
        rec: dict[int, str] = {}
        tee = _tee_finish_reasons(result_generator, rec)
        response = await orig_full(self, request, tee, *args, **kwargs)
        # An ErrorResponse has no choices.
        for choice in getattr(response, "choices", ()):
            if choice.finish_reason == "tool_calls":
                reason = _truthful_reason(rec, choice.index)
                if reason is not None:
                    choice.finish_reason = reason
        return response

    OpenAIServingChat.chat_completion_stream_generator = stream_generator
    OpenAIServingChat.chat_completion_full_generator = full_generator


# ---------------------------------------------------------------------------
# P3: keep a last <parameter> that has no </parameter>.
# ---------------------------------------------------------------------------

_TRAILING_CLOSERS_RE = re.compile(r"\s*</\s*(?:function|tool_call)\s*>\s*$")


def _install_unclosed_parameter_patch() -> None:
    """Keep an unclosed last parameter in the final argument JSON.

    ``qwen3_config`` is cached and stores the converter it was built with, so
    its cache is cleared after the module attribute is replaced.
    """
    from vllm.parser import qwen3

    if hasattr(qwen3, "_tt_original_qwen3_arg_converter"):
        return
    original = qwen3._qwen3_arg_converter
    qwen3._tt_original_qwen3_arg_converter = original
    param_re = qwen3._PARAM_RE
    partial_param_re = qwen3._PARTIAL_PARAM_RE
    trim = qwen3._trim_wrapping_newlines

    def arg_converter(raw_args: str, partial: bool) -> str:
        result = original(raw_args, partial)
        match = partial_param_re.search(param_re.sub("", raw_args))
        if match is None or not match.group(1):
            return result
        value = match.group(2)
        while _TRAILING_CLOSERS_RE.search(value):
            value = _TRAILING_CLOSERS_RE.sub("", value)
        params = json.loads(result)
        params[match.group(1)] = trim(value)
        return json.dumps(params, ensure_ascii=False)

    qwen3._qwen3_arg_converter = arg_converter
    qwen3.qwen3_config.cache_clear()


# ---------------------------------------------------------------------------
# P4: keep a string when the schema allows "string" and the conversion is lossy.
# ---------------------------------------------------------------------------


def _install_string_union_patch() -> None:
    """Keep "007" a string for a union that allows "string".

    A scalar conversion stays only when it round-trips to the raw text. Object
    and array results stay as converted.
    """
    from vllm.parser.engine import parser_engine
    from vllm.tool_parsers import utils as tool_utils

    if hasattr(parser_engine, "_tt_original_coerce_to_schema_type"):
        return
    original = parser_engine.coerce_to_schema_type
    parser_engine._tt_original_coerce_to_schema_type = original
    aliases = tool_utils._TYPE_ALIASES

    def coerce_to_schema_type(value: str, schema_type: str | list[str]) -> Any:
        result = original(value, schema_type)
        if not isinstance(value, str) or not (
            result is None or isinstance(result, (bool, int, float))
        ):
            return result
        types = [schema_type] if isinstance(schema_type, str) else schema_type
        names = {aliases.get(t.strip().lower(), t.strip().lower()) for t in types}
        if "string" in names and json.dumps(result) != value.strip():
            return value
        return result

    parser_engine.coerce_to_schema_type = coerce_to_schema_type
