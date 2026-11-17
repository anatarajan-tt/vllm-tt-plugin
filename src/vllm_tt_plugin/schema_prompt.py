# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""S1: show the model the JSON schema of a structured-output request.

vLLM enforces ``response_format={"type": "json_schema"}`` and the vLLM
``structured_outputs={"json": ...}`` field only with a token mask: the schema
never reaches the prompt. A model that cannot see the schema writes what it
would have written anyway. When the mask then forces its choices, the output
can loop until ``max_tokens``. Example: tags it invents are forced into a
string enum, and the model never chooses the allowed array close, so it writes
``"tags": ["guest", "guest", ...`` until max_tokens.
Qwen3.8 on TT: 0/12 OK without the schema in the prompt, 16/16 with it
(GRAM, 2026-11-11).

The patch wraps ``OpenAIServingChat.render_chat_request``: when the request's
constraint is a JSON schema (``response_format`` json_schema, or
``structured_outputs.json``), it renders a copy of the request whose first
system message ends with the schema. If the request has no system message, it
renders a new one. The request object itself and the mask are unchanged.
json_object, regex, choice, grammar, structural tags and tool calls are left
alone: tool schemas are already in the template.

VLLM_TT_SO_SCHEMA_PROMPT=0 turns the patch off at startup.

The patch follows the ``_install_*`` conventions of tool_call_fixes: it is
idempotent through a ``_tt_original_*`` marker, and it raises when its target is
missing.
"""

import json
import os

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

SCHEMA_PROMPT_HEADER = "Respond with JSON that conforms to this JSON schema:"


def schema_instruction(schema) -> str:
    """The text appended to the system prompt for ``schema`` (a dict or a JSON string)."""
    if not isinstance(schema, str):
        schema = json.dumps(schema, ensure_ascii=False)
    return f"{SCHEMA_PROMPT_HEADER}\n{schema}"


def request_json_schema(request):
    """The JSON schema that constrains ``request``, or None (no constraint, or not a JSON-schema one)."""
    so = request.extract_structured_outputs()
    if so is None:
        return None
    schema = getattr(so, "json", None)
    if schema is None or schema == {} or schema == "":
        return None
    return schema


def with_schema_message(messages, instruction):
    """A new message list: ``instruction`` appended to the first system message, or a new
    leading system message. The input list and its dicts are not modified."""
    msgs = list(messages)
    for i, m in enumerate(msgs):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "system":
            continue
        m = dict(m) if isinstance(m, dict) else m.model_dump()
        content = m.get("content")
        if isinstance(content, str) or content is None:
            m["content"] = f"{content}\n\n{instruction}" if content else instruction
        else:  # list of content parts
            m["content"] = list(content) + [{"type": "text", "text": "\n\n" + instruction}]
        msgs[i] = m
        return msgs
    return [{"role": "system", "content": instruction}] + msgs


def install_schema_prompt() -> None:
    if os.environ.get("VLLM_TT_SO_SCHEMA_PROMPT", "1") == "0":
        logger.info("structured-output schema prompt (S1) off: VLLM_TT_SO_SCHEMA_PROMPT=0")
        return
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    owner = OpenAIServingChat
    if getattr(owner, "_tt_original_render_chat_request", None) is not None:
        return
    original = getattr(owner, "render_chat_request", None)
    if original is None:
        raise RuntimeError("S1: OpenAIServingChat.render_chat_request not found (vLLM moved it)")

    async def render_chat_request(self, request, *args, **kwargs):
        try:
            schema = request_json_schema(request)
        except Exception as e:  # noqa: BLE001  (a malformed constraint is the original path's error to report)
            logger.debug(f"S1: no schema prompt ({e!r})")
            schema = None
        if schema is not None:
            request = request.model_copy(
                update={"messages": with_schema_message(request.messages, schema_instruction(schema))}
            )
        return await original(self, request, *args, **kwargs)

    owner._tt_original_render_chat_request = original
    owner.render_chat_request = render_chat_request
    logger.info("structured-output schema prompt (S1) active: JSON-schema requests show the schema in the system prompt")
