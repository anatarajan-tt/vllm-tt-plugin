# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Opt-in trace of where a request spends time before the scheduler queues it.

vLLM's request metrics start the clock at ``arrival_time`` (set by the
renderer before the chat template and tokenization) and count queue time from
the moment the scheduler queues the request. The gap between the two is not
reported anywhere. ``TT_PREENGINE_TRACE=1`` logs it, per request, in the
engine-core process:

- ``api_ms``: ``arrival_time`` -> engine core input thread receives the request
  (render, tokenization, input processing, IPC from the API server);
- ``pre_ms``: engine core input thread builds the ``Request`` (block hashes);
- ``loop_ms``: input thread done -> scheduler ``add_request``. The engine-core
  busy loop only drains its input queue between steps, so a request that
  arrives during a long prefill step waits here, uncounted by vLLM's queue
  time.

Off by default; when off nothing is patched. The patches follow the
``tool_call_fixes`` conventions: idempotent via a ``_tt_original_*`` marker,
and they raise at install if a target is missing.
"""

import functools
import os
import time

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

ENV = "TT_PREENGINE_TRACE"
_RECV_ATTR = "_tt_preengine_recv"
_PRE_ATTR = "_tt_preengine_pre_ms"


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def install_preengine_trace() -> bool:
    """Install the hooks when ``TT_PREENGINE_TRACE=1``; return whether installed."""
    if not enabled():
        return False
    _install_preprocess_patch()
    _install_add_request_patch()
    logger.info("pre-engine trace active (%s=1)", ENV)
    return True


def _install_preprocess_patch() -> None:
    from vllm.v1.engine.core import EngineCore

    marker = "_tt_original_preprocess_add_request"
    if getattr(EngineCore, marker, None) is not None:
        return
    original = EngineCore.preprocess_add_request  # raises if vLLM renamed it

    @functools.wraps(original)
    def preprocess_add_request(self, request):
        t0 = time.time()
        out = original(self, request)
        req = out[0] if isinstance(out, tuple) else out
        try:
            setattr(req, _RECV_ATTR, t0)
            setattr(req, _PRE_ATTR, (time.time() - t0) * 1e3)
        except Exception:  # noqa: BLE001 - tracing must never break admission
            pass
        return out

    setattr(EngineCore, marker, original)
    EngineCore.preprocess_add_request = preprocess_add_request


def format_line(request, now: float) -> str | None:
    recv = getattr(request, _RECV_ATTR, None)
    if recv is None:
        return None
    arrival = getattr(request, "arrival_time", None)
    n = getattr(request, "num_prompt_tokens", None)
    api_ms = (recv - arrival) * 1e3 if arrival else float("nan")
    pre_ms = getattr(request, _PRE_ATTR, float("nan"))
    loop_ms = (now - recv) * 1e3 - pre_ms
    return (
        f"[preengine] rid={request.request_id} prompt={n} "
        f"api_ms={api_ms:.1f} pre_ms={pre_ms:.1f} loop_ms={loop_ms:.1f}"
    )


def _install_add_request_patch() -> None:
    from vllm_tt_plugin.scheduler import TTScheduler

    marker = "_tt_original_add_request_preengine"
    if getattr(TTScheduler, marker, None) is not None:
        return
    original = TTScheduler.add_request

    @functools.wraps(original)
    def add_request(self, request):
        try:
            line = format_line(request, time.time())
            if line is not None:
                setattr(request, _RECV_ATTR, None)  # log once (continuations re-enter)
                logger.info(line)
        except Exception:  # noqa: BLE001
            pass
        return original(self, request)

    setattr(TTScheduler, marker, original)
    TTScheduler.add_request = add_request
