# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Multimodal prefix caching (VLLM_TT_MM_PREFIX_CACHE=1).

Safety: vLLM's block hashes carry each image's content hash, so a prompt with
the same text and a different image shares only the blocks before the image.
Plumbing: the scheduler lets image requests read the prefix cache only when
the flag is on and the model accepts it, keeps their pixel payloads, and the
runner passes each item's span and identifier to the model.
"""

from types import SimpleNamespace

import pytest
import torch
import vllm  # noqa: F401  (resolve the platform plugin before importing ours)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from vllm_tt_plugin.config import (
    MM_PREFIX_CACHE_ENV,
    get_tt_mm_prefix_cache,
    resolve_tt_mm_prefix_cache,
    store_tt_mm_prefix_cache,
)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler

BLOCK = 16
IMAGE_TOKEN = 7
# 20 text tokens, a 32-token image at [20, 52), then 40 text tokens: 92 tokens.
PREFIX = list(range(100, 120))
SUFFIX = list(range(200, 240))
IMAGE_SPAN = (20, 32)


def _feature(identifier, modality="image", offset=20, length=32, data=None):
    return MultiModalFeatureSpec(
        data=data,
        modality=modality,
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
        mm_hash=identifier,
    )


def _request(req_id, image_ids, suffix=SUFFIX):
    tokens = list(PREFIX)
    features = []
    for ident in image_ids:
        features.append(_feature(ident, offset=len(tokens)))
        tokens += [IMAGE_TOKEN] * IMAGE_SPAN[1]
    tokens += suffix
    return Request(
        req_id,
        tokens,
        SamplingParams(max_tokens=1),
        None,
        mm_features=features,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


@pytest.fixture
def kv_manager():
    init_none_hash(sha256)
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        enable_caching=True,
    )


def _cache(manager, request):
    blocks, hit, *_ = manager.get_computed_blocks(request)
    manager.allocate_slots(request, request.num_tokens - hit, hit, blocks)
    return hit


def _hit(manager, request):
    _, hit, *_ = manager.get_computed_blocks(request)
    return hit


# --- correctness: the block hashes carry the image content hash -------------


def test_block_hashes_diverge_at_the_first_block_holding_the_image():
    init_none_hash(sha256)
    a = _request("a", ["image-A"])
    b = _request("b", ["image-B"])
    same = _request("c", ["image-A"])
    assert list(a.all_token_ids) == list(b.all_token_ids)  # same text + placeholders
    # Block 0 ([0, 16)) is text only; block 1 ([16, 32)) holds the image start.
    assert a.block_hashes[0] == b.block_hashes[0]
    for i in range(1, len(a.block_hashes)):
        # Every block from the image on differs, the text after it included
        # (block hashes are chained).
        assert a.block_hashes[i] != b.block_hashes[i]
    assert a.block_hashes == same.block_hashes


def test_same_text_different_image_hits_only_the_text_before_the_image(
    kv_manager,
):
    assert _cache(kv_manager, _request("a", ["image-A"])) == 0
    # The image starts at 20, inside block 1: only block 0 is shared.
    assert _hit(kv_manager, _request("b", ["image-B"])) == BLOCK
    # The same image reuses every full block (92 tokens: 5 blocks; vLLM always
    # recomputes the last prompt token).
    assert _hit(kv_manager, _request("c", ["image-A"])) == 5 * BLOCK


def test_second_turn_reuses_the_first_turn_when_the_image_matches(kv_manager):
    turn1 = _request("t1", ["image-A"])
    _cache(kv_manager, turn1)
    follow_up = _request("t2", ["image-A"], suffix=SUFFIX + list(range(300, 360)))
    other_image = _request("t3", ["image-B"], suffix=SUFFIX + list(range(300, 360)))
    assert _hit(kv_manager, follow_up) == (turn1.num_tokens // BLOCK) * BLOCK
    assert _hit(kv_manager, other_image) == BLOCK


def test_image_order_matters(kv_manager):
    _cache(kv_manager, _request("ab", ["image-A", "image-B"]))
    assert _hit(kv_manager, _request("ba", ["image-B", "image-A"])) == BLOCK


# --- config: env AND model capability --------------------------------------


@pytest.mark.parametrize(
    ("env", "caps", "expected"),
    [
        (None, {"accepts_mm_prefix_cache": True}, False),
        ("0", {"accepts_mm_prefix_cache": True}, False),
        ("1", {"accepts_mm_prefix_cache": True}, True),
        ("1", {}, False),
        ("1", None, False),
    ],
)
def test_flag_needs_env_and_capability(monkeypatch, env, caps, expected):
    if env is None:
        monkeypatch.delenv(MM_PREFIX_CACHE_ENV, raising=False)
    else:
        monkeypatch.setenv(MM_PREFIX_CACHE_ENV, env)
    assert resolve_tt_mm_prefix_cache(caps) is expected
    config = SimpleNamespace(additional_config={})
    assert get_tt_mm_prefix_cache(config) is False
    store_tt_mm_prefix_cache(config, expected)
    assert get_tt_mm_prefix_cache(config) is expected


# --- scheduler ----------------------------------------------------------------


def _scheduler(enabled):
    s = TTScheduler.__new__(TTScheduler)
    s._mm_prefix_cache = enabled
    s._is_block_output_model = False
    return s


def _admit(scheduler, request, monkeypatch):
    seen = []
    monkeypatch.setattr(
        TTScheduler.__mro__[1], "add_request", lambda self, r: seen.append(r)
    )
    scheduler.add_request(request)
    assert seen == [request]
    return request.skip_reading_prefix_cache


def test_flag_off_keeps_the_skip_for_images(monkeypatch):
    assert _admit(_scheduler(False), _request("a", ["image-A"]), monkeypatch)


def test_flag_on_lets_image_requests_read_the_prefix_cache(monkeypatch):
    assert not _admit(_scheduler(True), _request("a", ["image-A"]), monkeypatch)


def test_flag_on_keeps_the_skip_for_video(monkeypatch):
    request = _request("v", [])
    request.mm_features = [_feature("video-A", modality="video")]
    assert _admit(_scheduler(True), request, monkeypatch)


def test_text_requests_are_untouched(monkeypatch):
    for enabled in (False, True):
        assert not _admit(_scheduler(enabled), _request("t", []), monkeypatch)


def _new_req(req_id, num_computed, features):
    return SimpleNamespace(
        req_id=req_id, num_computed_tokens=num_computed, mm_features=features
    )


def test_prefix_hit_payloads_are_restored_from_the_request():
    full = [_feature("image-A", data={"pixel_values": "payload"})]
    stripped = [_feature("image-A", data={"image_grid_thw": "grid only"})]
    s = _scheduler(True)
    s.requests = {"hit": SimpleNamespace(mm_features=full)}
    out = SimpleNamespace(
        scheduled_new_reqs=[
            _new_req("hit", 64, stripped),
            _new_req("cold", 0, list(stripped)),
        ]
    )
    s._restore_mm_payloads(out)
    assert out.scheduled_new_reqs[0].mm_features is full
    # No hit: vLLM stripped nothing, the list is left alone.
    assert out.scheduled_new_reqs[1].mm_features == stripped


def test_payloads_stay_stripped_with_the_flag_off():
    stripped = [_feature("image-A", data=None)]
    s = _scheduler(False)
    s.requests = {"hit": SimpleNamespace(mm_features=[_feature("image-A")])}
    out = SimpleNamespace(scheduled_new_reqs=[_new_req("hit", 64, stripped)])
    s._restore_mm_payloads(out)
    assert out.scheduled_new_reqs[0].mm_features is stripped


# --- runner: per-row item spans ------------------------------------------------


def _field(t):
    return SimpleNamespace(data=t)


def _runner(features_per_req, enabled):
    req_ids = [f"r{i}" for i in range(len(features_per_req))]
    return SimpleNamespace(
        mm_prefix_cache=enabled,
        input_batch=SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids)),
        requests={
            rid: SimpleNamespace(mm_features=feats)
            for rid, feats in zip(req_ids, features_per_req)
        },
        _validate_mm_feature=lambda f: TTModelRunner._validate_mm_feature(None, f),
        _mm_item_spans=TTModelRunner._mm_item_spans,
    )


def _image_feature(identifier, offset, length=4):
    return _feature(
        identifier,
        offset=offset,
        length=length,
        data={
            "pixel_values": _field(torch.zeros(16, 3)),
            "image_grid_thw": _field(torch.tensor([1, 4, 4])),
        },
    )


def test_item_spans_follow_the_rows_when_enabled():
    rows = [[], [_image_feature("h1", 5), _image_feature("h2", 30, 8)]]
    out = TTModelRunner._gather_multi_modal_inputs(_runner(rows, True), [1, 0])
    assert out["mm_item_spans"] == [[(5, 4, "h1"), (30, 8, "h2")], None]


def test_no_item_spans_when_disabled():
    rows = [[_image_feature("h1", 5)]]
    out = TTModelRunner._gather_multi_modal_inputs(_runner(rows, False), None)
    assert "mm_item_spans" not in out


def test_a_stripped_item_still_reports_its_span_and_grid():
    # A prefix-covered item that vLLM stripped (data keeps only the CPU grid):
    # the span and identifier come from the feature, not the payload.
    stripped = _feature(
        "h1",
        offset=5,
        length=4,
        data={"image_grid_thw": _field(torch.tensor([1, 4, 4]))},
    )
    out = TTModelRunner._gather_multi_modal_inputs(_runner([[stripped]], True), None)
    assert out["mm_item_spans"] == [[(5, 4, "h1")]]
    assert out["pixel_values"] == [[None]]
    assert torch.equal(out["image_grid_thw"][0][0], torch.tensor([1, 4, 4]))


# --- client uuids never become identifiers ------------------------------------


def test_client_uuid_is_ignored_once_the_patch_is_installed(monkeypatch):
    import vllm.multimodal.processing.inputs as mm_inputs
    from vllm.multimodal.parse import MultiModalDataParser

    from vllm_tt_plugin import platform as tt_platform

    monkeypatch.delattr(mm_inputs, "_tt_original_get_mm_hashes", raising=False)
    monkeypatch.setattr(
        mm_inputs.ProcessorInputs,
        "get_mm_hashes",
        mm_inputs.ProcessorInputs.get_mm_hashes,
    )
    parser = MultiModalDataParser()

    def hashes(image, uuid):
        items = parser.parse_mm_data({"image": [image]})
        uuids = None if uuid is None else {"image": [uuid]}
        inputs = mm_inputs.ProcessorInputs([1, 2, 3], items, uuids)
        return inputs.get_mm_hashes("model", "sha256")["image"][0]

    red = torch.zeros(3, 8, 8)
    blue = torch.ones(3, 8, 8)
    # Upstream: a uuid replaces the content hash, so two different images
    # sent with one uuid share an identifier.
    assert hashes(red, "shared") == hashes(blue, "shared") == "shared"
    tt_platform._install_mm_content_hash_patch()
    assert hashes(red, "shared") == hashes(red, None)
    assert hashes(blue, "shared") == hashes(blue, None)
    assert hashes(red, "shared") != hashes(blue, "shared")
