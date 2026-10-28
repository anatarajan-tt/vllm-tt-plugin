# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The runner side of penalties on the ``topk`` support, and the reason counters.

``_spec_mode_for_step`` asks for ``topk`` for a penalized sampled row only
with VLLM_TT_SPEC_TOPK_PENALTIES=1 and only when every penalty can only lower
a logit; a seeded penalized request commits the same stream in both modes.
VLLM_TT_SPEC_LOGITS_REASONS=1 counts why a verify stayed in ``logits``.
"""

from __future__ import annotations

import pytest
from vllm.sampling_params import SamplingParams

from vllm_tt_plugin.spec_decode import ACCEPT_MODE_LOGITS, ACCEPT_MODE_TOPK

from .test_spec_device_topk import (
    SAMPLED,
    _mode_for,
    _speculate,
    _topk_runner,
    topk_on,  # noqa: F401  (fixture)
)
from .test_spec_sampled_runner import _add_request

PENALIZED = SamplingParams(
    temperature=1.0, top_k=5, presence_penalty=0.5, seed=3, ignore_eos=True
)


@pytest.fixture
def penalties_on(monkeypatch, topk_on):  # noqa: F811
    monkeypatch.setenv("VLLM_TT_SPEC_TOPK_PENALTIES", "1")


def test_a_lowering_penalty_asks_for_topk_with_the_switch_on(penalties_on):
    model, runner = _topk_runner()
    _add_request(runner, "s", SAMPLED, first_token=1)
    _add_request(runner, "p", PENALIZED, first_token=9)
    assert _mode_for(runner, "s", "p") == ACCEPT_MODE_TOPK


def test_the_switch_off_keeps_a_penalized_row_in_logits(topk_on, monkeypatch):  # noqa: F811
    monkeypatch.delenv("VLLM_TT_SPEC_TOPK_PENALTIES", raising=False)
    model, runner = _topk_runner()
    _add_request(runner, "p", PENALIZED)
    assert _mode_for(runner, "p") == ACCEPT_MODE_LOGITS


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(
            temperature=1.0, top_k=5, presence_penalty=-0.5, ignore_eos=True
        ),
        SamplingParams(
            temperature=1.0, top_k=5, frequency_penalty=-0.5, ignore_eos=True
        ),
        SamplingParams(
            temperature=1.0, top_k=5, repetition_penalty=0.9, ignore_eos=True
        ),
        SamplingParams(temperature=0.0, repetition_penalty=1.2, ignore_eos=True),
        SamplingParams(temperature=1.0, top_k=0, presence_penalty=0.5, ignore_eos=True),
    ],
    ids=[
        "raising presence",
        "raising frequency",
        "repetition below 1",
        "greedy",
        "top_k off",
    ],
)
def test_a_penalty_the_support_cannot_certify_stays_in_logits(penalties_on, params):
    model, runner = _topk_runner()
    _add_request(runner, "p", params)
    assert _mode_for(runner, "p") == ACCEPT_MODE_LOGITS


def test_a_penalized_seeded_stream_is_the_same_in_both_modes(monkeypatch):
    params = dict(
        temperature=1.0,
        top_k=5,
        presence_penalty=0.7,
        frequency_penalty=0.2,
        repetition_penalty=1.1,
        seed=77,
        ignore_eos=True,
    )
    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "1")
    monkeypatch.delenv("VLLM_TT_SPEC_FAST_TOPK", raising=False)
    monkeypatch.delenv("VLLM_TT_SPEC_TIE_EXACT", raising=False)
    monkeypatch.delenv("VLLM_TT_SPEC_TOPK_PENALTIES", raising=False)
    model, runner = _topk_runner()
    _add_request(runner, "s", SamplingParams(**params))
    reference, widths = _speculate(runner, "s", 40)
    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}
    assert max(widths) > 1

    monkeypatch.setenv("VLLM_TT_SPEC_TOPK_PENALTIES", "1")
    model, runner = _topk_runner()
    _add_request(runner, "s", SamplingParams(**params))
    stream, _ = _speculate(runner, "s", 40)
    assert set(model.verify_modes) == {ACCEPT_MODE_TOPK}
    assert stream == reference


def test_the_reason_counters_name_what_forced_logits(topk_on, monkeypatch):  # noqa: F811
    monkeypatch.setenv("VLLM_TT_SPEC_LOGITS_REASONS", "1")
    monkeypatch.delenv("VLLM_TT_SPEC_TOPK_PENALTIES", raising=False)
    model, runner = _topk_runner()
    _add_request(runner, "s", SAMPLED, first_token=1)
    _add_request(
        runner,
        "x",
        SamplingParams(temperature=1.0, top_k=5, logprobs=1, ignore_eos=True),
        first_token=9,
    )
    assert _mode_for(runner, "s", "x") == ACCEPT_MODE_LOGITS
    report = runner.logits_mode_report()
    assert report.startswith("; logits reasons: ")
    assert "logprobs=" in report and "penalties" not in report
    steps = runner._logits_mode_steps
    assert steps >= 1
    # One row of the two forced each logits step.
    assert runner._logits_mode_offending_rows == steps
    assert runner._logits_mode_step_rows == 2 * steps


def test_the_reason_report_is_empty_with_the_switch_off(topk_on, monkeypatch):  # noqa: F811
    monkeypatch.delenv("VLLM_TT_SPEC_LOGITS_REASONS", raising=False)
    model, runner = _topk_runner()
    _add_request(runner, "x", SamplingParams(temperature=1.0, top_k=0, ignore_eos=True))
    assert _mode_for(runner, "x") == ACCEPT_MODE_LOGITS
    assert runner.logits_mode_report() == ""
    assert "_logits_mode_steps" not in runner.__dict__
