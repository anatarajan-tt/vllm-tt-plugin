# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for image/video multimodal input gathering."""

from types import SimpleNamespace

import pytest
import torch

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.model_runner import TTModelRunner


def _field(t):
    return SimpleNamespace(data=t)


def _image(tag):
    return SimpleNamespace(
        modality="image",
        data={
            "pixel_values": _field(torch.full((4, 3), float(tag))),
            "image_grid_thw": _field(torch.tensor([[1, 2, 2]])),
        },
    )


def _video(tag):
    return SimpleNamespace(
        modality="video",
        data={
            "pixel_values_videos": _field(torch.full((8, 3), float(tag))),
            "video_grid_thw": _field(torch.tensor([[2, 2, 2]])),
        },
    )


def _runner(features_per_req):
    req_ids = [f"r{i}" for i in range(len(features_per_req))]
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids)),
        requests={
            rid: SimpleNamespace(mm_features=feats)
            for rid, feats in zip(req_ids, features_per_req)
        },
        _validate_mm_feature=lambda f: TTModelRunner._validate_mm_feature(None, f),
    )
    return runner


def _gather(features_per_req, req_indices=None):
    return TTModelRunner._gather_multi_modal_inputs(
        _runner(features_per_req), req_indices
    )


def test_text_only_and_image_rows_have_no_video_keys():
    out = _gather([[], [_image(1)], [_image(2), _image(3)]])
    assert set(out) == {"pixel_values", "image_grid_thw"}
    assert out["pixel_values"][0] is None
    assert out["image_grid_thw"][0] is None
    assert len(out["pixel_values"][1]) == 1
    assert [t[0, 0].item() for t in out["pixel_values"][2]] == [2.0, 3.0]


def test_video_row():
    out = _gather([[], [_video(5)]])
    assert out["pixel_values_videos"][0] is None
    assert out["video_grid_thw"][0] is None
    assert out["pixel_values_videos"][1][0][0, 0].item() == 5.0
    assert out["video_grid_thw"][1][0].tolist() == [[2, 2, 2]]
    # video-only request has no image entries
    assert out["pixel_values"] == [None, None]
    assert out["image_grid_thw"] == [None, None]


def test_mixed_batch_and_req_indices():
    feats = [[], [_image(1)], [_video(2)], [_image(3), _video(4), _video(5)]]
    out = _gather(feats)
    assert out["pixel_values"][2] is None
    assert out["pixel_values_videos"][1] is None
    assert [t[0, 0].item() for t in out["pixel_values"][3]] == [3.0]
    assert [t[0, 0].item() for t in out["pixel_values_videos"][3]] == [4.0, 5.0]
    assert len(out["video_grid_thw"][3]) == 2
    for v in out.values():
        assert len(v) == 4
    # subset / reordered rows
    sub = _gather(feats, req_indices=[3, 0])
    assert len(sub["pixel_values_videos"]) == 2
    assert sub["pixel_values_videos"][1] is None
    assert len(sub["pixel_values_videos"][0]) == 2


def test_item_none_data_video():
    f = SimpleNamespace(modality="video", data=None)
    out = _gather([[f]])
    assert out["pixel_values_videos"] == [[None]]
    assert out["video_grid_thw"] == [[None]]


def test_unsupported_modality_rejected():
    f = SimpleNamespace(modality="audio", data=None)
    with pytest.raises(NotImplementedError, match="audio"):
        _gather([[f]])


def test_stripped_item_yields_none_payload():
    stripped = SimpleNamespace(
        modality="image",
        identifier="abc123",
        data={"image_grid_thw": _field(torch.tensor([[1, 2, 2]]))},
    )
    out = _gather([[stripped]])
    assert out["pixel_values"] == [[None]]
    assert out["image_grid_thw"][0][0].tolist() == [[1, 2, 2]]
