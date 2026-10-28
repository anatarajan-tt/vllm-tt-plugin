# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Penalties on the ``topk`` support, and the logits-mode reason classifier.

With VLLM_TT_SPEC_TOPK_PENALTIES=1 a ``topk`` verify may carry rows with
presence, frequency and repetition penalties that can only lower a logit. The
walk applies vLLM's penalty op to the M candidates and decides a position from
them only when the k-th penalized value is above the smallest raw candidate,
which bounds every penalized logit outside the support; any other position is
sampled from the full logits after the same op. These tests hold that walk to
the ``logits`` walk on the full logits: the penalized candidate values bit for
bit, and the same committed tokens and generator consumption.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_tt_plugin.spec_accept import (
    SpecPenalties,
    SpecSamplingInputs,
    _FetchOnce,
    _penalize_support,
    accept_sampled_drafts,
    accept_sampled_support,
    apply_speculative_penalties,
    logits_mode_reasons,
    penalties_lower_only,
    topk_penalties_enabled,
)

V = 4096
M = 64
K = 3


@pytest.fixture
def penalties_on(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_TOPK_PENALTIES", "1")
    monkeypatch.delenv("VLLM_TT_SPEC_FAST_TOPK", raising=False)
    monkeypatch.delenv("VLLM_TT_SPEC_TIE_EXACT", raising=False)
    monkeypatch.delenv("VLLM_TT_SPEC_TOPK_AUDIT", raising=False)


def exact_support(logits: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The device's support: a full stable sort, (value desc, token id asc)."""
    rows, width, vocab = logits.shape
    values, ids = torch.sort(
        logits.reshape(-1, vocab), dim=1, descending=True, stable=True
    )
    return (
        values[:, :m].reshape(rows, width, m).contiguous(),
        ids[:, :m].reshape(rows, width, m).to(torch.int32),
    )


def bf16_logits(seed: int, rows: int, width: int, scale: float = 3.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(rows, width, V, generator=g) * scale).to(torch.bfloat16)
    return logits.to(torch.float32)


def drafts_for(logits: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width, _ = logits.shape
    g = torch.Generator().manual_seed(seed + 2000)
    top3 = torch.topk(logits[:, : width - 1], 3, dim=-1).indices
    pick = torch.randint(0, 3, (rows, width - 1), generator=g)
    drafts = top3.gather(2, pick.unsqueeze(2)).squeeze(2).to(torch.int32)
    num_valid = torch.randint(0, width, (rows,), generator=g).to(torch.int32)
    return drafts, num_valid


def penalties_for(
    logits: torch.Tensor,
    seed: int,
    presence: list[float],
    frequency: list[float],
    repetition: list[float],
) -> SpecPenalties:
    """Prompt and history drawn mostly from each row's top tokens, so they bite."""
    rows = logits.shape[0]
    g = torch.Generator().manual_seed(seed + 5000)
    top = torch.topk(logits[:, 0], 80, dim=-1).indices
    prompt_len = 40
    prompt = torch.full((rows, prompt_len), V, dtype=torch.int64)
    history: list[list[int]] = []
    for row in range(rows):
        n = int(torch.randint(5, prompt_len, (1,), generator=g))
        picks = top[row, torch.randint(0, 80, (n,), generator=g)]
        prompt[row, :n] = picks
        h = int(torch.randint(0, 30, (1,), generator=g))
        hist = top[row, torch.randint(0, 80, (h,), generator=g)].tolist()
        hist += torch.randint(0, V, (3,), generator=g).tolist()
        history.append([int(t) for t in hist])
    return SpecPenalties(
        prompt_token_ids=prompt,
        output_token_ids=history,
        presence=torch.tensor(presence, dtype=torch.float32),
        frequency=torch.tensor(frequency, dtype=torch.float32),
        repetition=torch.tensor(repetition, dtype=torch.float32),
    )


ROWS = 6
TEMPERATURE = torch.tensor([0.6, 1.0, 0.8, 1.3, 0.7, 0.0])
TOP_K = torch.tensor([20, 1, 50, 40, 5, 20], dtype=torch.int32)
TOP_P = torch.tensor([0.95, 0.95, 0.9, 1.0, 0.5, 1.0])
PRESENCE = [0.0, 1.5, 0.5, 0.0, 2.0, 0.0]
FREQUENCY = [0.0, 0.0, 0.3, 0.2, 1.0, 0.0]
REPETITION = [1.0, 1.0, 1.1, 1.3, 1.05, 1.0]


def sampling(penalties: SpecPenalties, seeds: dict[int, int]) -> SpecSamplingInputs:
    return SpecSamplingInputs(
        vocab_size=V,
        temperature=TEMPERATURE.clone(),
        top_k=TOP_K.clone(),
        top_p=TOP_P.clone(),
        generators={r: torch.Generator().manual_seed(s) for r, s in seeds.items()},
        penalties=penalties,
    )


@pytest.mark.parametrize("seed", range(8))
def test_penalized_candidates_are_the_full_op_bit_for_bit(seed):
    rows, width = ROWS, K + 1
    logits = bf16_logits(seed, rows, width)
    drafts, num_valid = drafts_for(logits, seed)
    penalties = penalties_for(logits, seed, PRESENCE, FREQUENCY, REPETITION)
    values, ids = exact_support(logits, M)

    full = apply_speculative_penalties(logits, drafts, num_valid, penalties)
    want = full.gather(2, ids.to(torch.int64))
    got = _penalize_support(values, ids, drafts, num_valid, penalties, V)
    assert got.dtype == torch.float32
    assert torch.equal(got, want)


@pytest.mark.parametrize("seed", range(24))
def test_the_penalized_support_walk_commits_what_the_logits_walk_commits(
    penalties_on, seed
):
    rows, width = ROWS, K + 1
    logits = bf16_logits(seed, rows, width)
    drafts, num_valid = drafts_for(logits, seed)
    penalties = penalties_for(logits, seed, PRESENCE, FREQUENCY, REPETITION)
    values, ids = exact_support(logits, M)
    seeds = {0: 11 + seed, 2: 22 + seed, 4: 33 + seed}

    full = sampling(penalties, seeds)
    torch.manual_seed(seed)
    reference = accept_sampled_drafts(logits, drafts, num_valid, full)
    reference_rng = torch.get_rng_state()

    support = sampling(penalties, seeds)
    torch.manual_seed(seed)
    fetch = _FetchOnce(lambda: logits, rows, width, V)
    result = accept_sampled_support(
        logits.argmax(-1).to(torch.int32),
        values,
        ids,
        drafts,
        num_valid,
        support,
        fetch,
    )
    assert torch.equal(result.committed_token_ids, reference.committed_token_ids)
    assert torch.equal(result.accepted_counts, reference.accepted_counts)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    for row, g in full.generators.items():
        assert torch.equal(support.generators[row].get_state(), g.get_state()), row


def test_a_penalty_that_pushes_the_kth_below_the_bound_falls_back_and_matches(
    penalties_on,
):
    """Flat logits and a huge presence penalty on most candidates: undecidable."""
    rows, width = 2, K + 1
    logits = bf16_logits(3, rows, width, scale=0.05)
    drafts, num_valid = drafts_for(logits, 3)
    top = torch.topk(logits[:, 0], M, dim=-1).indices
    prompt = top.to(torch.int64)  # every column-0 candidate seen in the prompt
    penalties = SpecPenalties(
        prompt_token_ids=prompt,
        output_token_ids=[top[0].tolist(), top[1].tolist()],
        presence=torch.tensor([2.0, 2.0]),
        frequency=torch.tensor([0.0, 0.0]),
        repetition=torch.tensor([1.0, 1.0]),
    )
    values, ids = exact_support(logits, M)
    make = lambda: SpecSamplingInputs(  # noqa: E731
        vocab_size=V,
        temperature=torch.tensor([1.0, 0.7]),
        top_k=torch.tensor([20, 40], dtype=torch.int32),
        top_p=torch.tensor([1.0, 0.9]),
        generators={0: torch.Generator().manual_seed(9)},
        penalties=penalties,
    )
    torch.manual_seed(4)
    reference = accept_sampled_drafts(logits, drafts, num_valid, make())
    torch.manual_seed(4)
    fetch = _FetchOnce(lambda: logits, rows, width, V)
    result = accept_sampled_support(
        logits.argmax(-1).to(torch.int32), values, ids, drafts, num_valid, make(), fetch
    )
    assert fetch.calls == 1, "column 0 cannot be decided from the support"
    assert torch.equal(result.committed_token_ids, reference.committed_token_ids)
    assert torch.equal(result.accepted_counts, reference.accepted_counts)


def test_the_audit_passes_with_penalties(penalties_on, monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_TOPK_AUDIT", "1")
    rows, width = ROWS, K + 1
    logits = bf16_logits(5, rows, width)
    drafts, num_valid = drafts_for(logits, 5)
    penalties = penalties_for(logits, 5, PRESENCE, FREQUENCY, REPETITION)
    values, ids = exact_support(logits, M)
    fetch = _FetchOnce(lambda: logits, rows, width, V)
    accept_sampled_support(
        logits.argmax(-1).to(torch.int32),
        values,
        ids,
        drafts,
        num_valid,
        sampling(penalties, {}),
        fetch,
    )
    assert fetch.calls == 1


def _one_row(presence, frequency, repetition, temperature):
    logits = bf16_logits(1, 1, K + 1)
    drafts, num_valid = drafts_for(logits, 1)
    values, ids = exact_support(logits, M)
    penalties = SpecPenalties(
        prompt_token_ids=torch.full((1, 1), V, dtype=torch.int64),
        output_token_ids=[[int(ids[0, 0, 0])]],
        presence=torch.tensor([presence]),
        frequency=torch.tensor([frequency]),
        repetition=torch.tensor([repetition]),
    )
    inputs = SpecSamplingInputs(
        vocab_size=V,
        temperature=torch.tensor([temperature]),
        top_k=torch.tensor([20], dtype=torch.int32),
        penalties=penalties,
    )
    return (logits.argmax(-1).to(torch.int32), values, ids, drafts, num_valid, inputs)


def test_penalties_stay_refused_with_the_switch_off(monkeypatch):
    monkeypatch.delenv("VLLM_TT_SPEC_TOPK_PENALTIES", raising=False)
    assert not topk_penalties_enabled()
    with pytest.raises(ValueError, match="handed \\['penalties'\\]"):
        accept_sampled_support(*_one_row(0.5, 0.0, 1.0, 1.0))


@pytest.mark.parametrize(
    "presence, frequency, repetition, temperature",
    [
        (-0.5, 0.0, 1.0, 1.0),
        (0.0, -0.1, 1.0, 1.0),
        (0.0, 0.0, 0.9, 1.0),
        (0.5, 0.0, 1.0, 0.0),
    ],
    ids=["negative presence", "negative frequency", "repetition below 1", "greedy"],
)
def test_a_penalty_the_support_cannot_certify_is_refused(
    penalties_on, presence, frequency, repetition, temperature
):
    with pytest.raises(ValueError, match="cannot\\s+certify"):
        accept_sampled_support(*_one_row(presence, frequency, repetition, temperature))


@pytest.mark.parametrize(
    "presence, frequency, repetition, lowers",
    [
        (0.0, 0.0, 1.0, True),
        (1.5, 0.5, 1.2, True),
        (-0.1, 0.0, 1.0, False),
        (0.0, -0.1, 1.0, False),
        (0.0, 0.0, 0.95, False),
    ],
)
def test_penalties_lower_only(presence, frequency, repetition, lowers):
    assert penalties_lower_only(presence, frequency, repetition) is lowers


def _params(**overrides):
    base = dict(
        temperature=1.0,
        top_k=20,
        top_p=1.0,
        logprobs=None,
        structured_outputs=None,
        allowed_token_ids=None,
        bad_words_token_ids=None,
        min_tokens=0,
        all_stop_token_ids=set(),
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides, penalized, on_support, candidates, want",
    [
        ({}, False, False, 64, ()),
        ({"temperature": 0.0, "top_k": 0}, False, False, 64, ()),
        ({"top_k": 0, "top_p": 0.95}, False, False, 64, ("top_k_unset_top_p",)),
        ({"top_k": 0}, False, False, 64, ("top_k_unset",)),
        ({"top_k": 64}, False, False, 64, ("top_k_large",)),
        ({"top_k": 63}, False, False, 64, ()),
        ({"logprobs": 0}, False, False, 64, ("logprobs",)),
        ({"structured_outputs": object()}, False, False, 64, ("grammar",)),
        ({"allowed_token_ids": [1]}, False, False, 64, ("allowed_token_ids",)),
        ({"bad_words_token_ids": [[1]]}, False, False, 64, ("bad_words",)),
        (
            {"min_tokens": 5, "all_stop_token_ids": {2}},
            False,
            False,
            64,
            ("min_tokens",),
        ),
        ({"presence_penalty": 1.5}, True, False, 64, ("penalties",)),
        ({"presence_penalty": 1.5}, True, True, 64, ()),
        ({"presence_penalty": -1.0}, True, True, 64, ("penalties",)),
        (
            {"temperature": 0.0, "repetition_penalty": 1.1},
            True,
            True,
            64,
            ("penalties",),
        ),
        (
            {"presence_penalty": 1.5, "top_k": 0},
            True,
            False,
            64,
            ("penalties", "top_k_unset"),
        ),
        ({}, False, False, None, ("no_topk",)),
    ],
)
def test_logits_mode_reasons(overrides, penalized, on_support, candidates, want):
    got = logits_mode_reasons(
        _params(**overrides),
        3,
        candidates,
        penalized=penalized,
        penalties_on_support=on_support,
    )
    assert got == want
