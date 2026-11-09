# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""TT_PREENGINE_TRACE: off patches nothing; on, admission is timed and logged."""

import types

import pytest

from vllm_tt_plugin import preengine_trace as pt


@pytest.fixture
def restore():
    from vllm.v1.engine.core import EngineCore

    from vllm_tt_plugin.scheduler import TTScheduler

    saved = (EngineCore.preprocess_add_request, TTScheduler.add_request)
    yield
    EngineCore.preprocess_add_request, TTScheduler.add_request = saved
    for owner, marker in (
        (EngineCore, "_tt_original_preprocess_add_request"),
        (TTScheduler, "_tt_original_add_request_preengine"),
    ):
        if marker in owner.__dict__:
            delattr(owner, marker)


def test_off_by_default_patches_nothing(monkeypatch, restore):
    from vllm.v1.engine.core import EngineCore

    from vllm_tt_plugin.scheduler import TTScheduler

    monkeypatch.delenv(pt.ENV, raising=False)
    before = (EngineCore.preprocess_add_request, TTScheduler.add_request)
    assert pt.install_preengine_trace() is False
    assert (EngineCore.preprocess_add_request, TTScheduler.add_request) == before


def test_on_installs_idempotently(monkeypatch, restore):
    from vllm.v1.engine.core import EngineCore

    from vllm_tt_plugin.scheduler import TTScheduler

    monkeypatch.setenv(pt.ENV, "1")
    assert pt.install_preengine_trace() is True
    first = (EngineCore.preprocess_add_request, TTScheduler.add_request)
    assert pt.install_preengine_trace() is True
    assert (EngineCore.preprocess_add_request, TTScheduler.add_request) == first
    assert EngineCore._tt_original_preprocess_add_request is not None
    assert TTScheduler._tt_original_add_request_preengine is not None


def test_preprocess_hook_stamps_request(monkeypatch, restore):
    from vllm.v1.engine.core import EngineCore

    req = types.SimpleNamespace(request_id="r1")
    monkeypatch.setattr(EngineCore, "preprocess_add_request", lambda self, r: (req, 0))
    monkeypatch.setenv(pt.ENV, "1")
    pt._install_preprocess_patch()
    out = EngineCore.preprocess_add_request(object(), object())
    assert out == (req, 0)
    assert isinstance(getattr(req, pt._RECV_ATTR), float)
    assert getattr(req, pt._PRE_ATTR) >= 0.0


def test_format_line_splits_the_gap():
    req = types.SimpleNamespace(
        request_id="r2", arrival_time=100.0, num_prompt_tokens=1234
    )
    setattr(req, pt._RECV_ATTR, 100.5)
    setattr(req, pt._PRE_ATTR, 20.0)
    line = pt.format_line(req, now=101.0)
    want = "[preengine] rid=r2 prompt=1234 api_ms=500.0 pre_ms=20.0 loop_ms=480.0"
    assert line == want


def test_format_line_skips_unstamped_requests():
    assert pt.format_line(types.SimpleNamespace(request_id="x"), now=1.0) is None


def test_add_request_logs_once_and_delegates(monkeypatch, restore):
    from vllm_tt_plugin.scheduler import TTScheduler

    seen = []
    monkeypatch.setattr(
        TTScheduler, "add_request", lambda self, r: seen.append(r.request_id)
    )
    lines = []
    monkeypatch.setattr(
        pt.logger, "info", lambda msg, *a: lines.append(msg % a if a else msg)
    )
    pt._install_add_request_patch()
    req = types.SimpleNamespace(request_id="r3", arrival_time=1.0, num_prompt_tokens=5)
    setattr(req, pt._RECV_ATTR, 1.1)
    setattr(req, pt._PRE_ATTR, 1.0)
    TTScheduler.add_request(object(), req)
    TTScheduler.add_request(object(), req)  # continuation: delegated, not logged again
    assert seen == ["r3", "r3"]
    assert len([x for x in lines if x.startswith("[preengine]")]) == 1
