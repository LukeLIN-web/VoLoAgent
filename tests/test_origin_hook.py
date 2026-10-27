# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""origin_hook: the object-frame π0.5 tool's origin, skill and call id (--tool-origin, agent-harness #118)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from vlm_orchestrator.strategies import origin_hook as oh
from vlm_orchestrator.strategies.base import SessionState, StrategyContext

H = W = 100   # the front depth; the front raw image is twice that, so SAM2's masks are resized
CAMERA_POS = np.array([0.3, -0.2, 1.0])
# 1 m above the table looking straight down (identity OpenGL orientation looks along world -z): pixel (u, v)
# at depth d is the world point CAMERA_POS + (d (u - 50) / 100, -d (v - 50) / 100, -d).
K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
BOX = (slice(41, 60), slice(31, 70))    # rows, cols of a flat-topped box 0.1 m tall, centred on pixel (50, 50)
MASK = (slice(36, 65), slice(26, 75))   # SAM2's mask: the box and the table round it
SMALL = (slice(48, 53), slice(48, 53))  # 25 px on the box top
BOX_CENTRE = [0.3, -0.2, (0.1 + oh.TABLE_Z) / 2]   # under the camera, between the box top and the calibrated table
BOX_TOP = [0.3, -0.2, 0.1]               # the depth back-projection of TARGET's pixel
TARGET = {"phrase": "red cube", "point": (0.505, 0.505)}          # pixel (50, 50), on the box top
DESTINATION = {"phrase": "grey bowl", "point": (0.105, 0.105)}    # pixel (10, 10), on the table
FIELDS = {"episode", "step_count", "subgoal_idx", "prompt", "skill", "role", "call", "grounding", "mask_px",
          "sam2_iou", "boxtable", "pixel", "origin", "source", "wall_s"}


def _region(rows_cols) -> np.ndarray:
    region = np.zeros((H, W), bool)
    region[rows_cols] = True
    return region


def _depth() -> np.ndarray:
    depth = np.full((H, W), 1.0, np.float32)
    depth[BOX] = 0.9
    return depth


class FakeSam2:
    """GraspClient's segment: ``mask`` at the image's resolution, IoU 0.9; records the points."""

    def __init__(self):
        self.points, self.mask = [], _region(MASK)

    def segment(self, image, x, y):
        self.points.append((x, y))
        scale = image.shape[0] // H
        return self.mask.repeat(scale, 0).repeat(scale, 1), 0.9


def _hook(monkeypatch, grounding=None):
    """A hook whose master answers ``grounding`` (raises it when an exception) and whose SAM2 is FakeSam2."""
    asked = []

    def ask(image, prompt):
        asked.append(prompt)
        if isinstance(grounding, Exception):
            raise grounding
        return grounding or {"target": TARGET, "destination": DESTINATION}

    monkeypatch.setattr(oh, "ask_reference_targets", ask)
    monkeypatch.setattr(oh, "GraspClient", FakeSam2)
    return oh.OriginHook(StrategyContext(vlm=None)), asked


def _obs(prompt="pick up the red cube", gripper=0.0, depth=None):
    depth_key, k_key, pos_key, quat_key = oh.FRONT
    return {
        "prompt": prompt,
        oh.GRIPPER: np.array([gripper]),
        oh.FRONT_IMAGE: np.zeros((2 * H, 2 * W, 3), np.uint8),
        depth_key: (_depth() if depth is None else depth)[..., None],
        k_key: K,
        pos_key: CAMERA_POS,
        quat_key: np.array([1.0, 0.0, 0.0, 0.0]),
    }


def _state(tmp_path):
    return SessionState(episode_id=3, infer_count=0, episode_log_dir=str(tmp_path))


def _lines(tmp_path):
    return [json.loads(x) for x in open(tmp_path / "origin.jsonl")]


def test_boxtable_is_the_box_centre_with_its_bottom_at_the_table(monkeypatch, tmp_path):
    hook, _ = _hook(monkeypatch)
    obs = _obs()
    out = hook(obs, _state(tmp_path))
    origin = out[oh.ORIGIN_KEY]
    assert isinstance(origin, np.ndarray) and origin.dtype == np.float32 and origin.shape == (3,)
    np.testing.assert_allclose(origin, BOX_CENTRE, atol=1e-6)
    assert out[oh.SKILL_KEY] == "pick" and out[oh.CALL_KEY] == f"{os.getpid()}-3-0"
    assert out["prompt"] == "pick up the red cube" and oh.ORIGIN_KEY not in obs   # the strategy's obs is untouched
    (line,) = _lines(tmp_path)
    assert set(line) == FIELDS
    assert line["source"] == "boxtable" and line["role"] == "target"
    assert line["mask_px"] == 29 * 49 and line["sam2_iou"] == 0.9
    assert line["boxtable"] == pytest.approx(BOX_CENTRE, abs=1e-4) and line["origin"] == line["boxtable"]
    assert line["pixel"] == pytest.approx(BOX_TOP, abs=1e-4)
    assert line["grounding"]["target"] == {"phrase": "red cube", "point": [0.505, 0.505]}


def test_too_few_mask_points_fall_back_to_the_pixel(monkeypatch, tmp_path):
    hook, _ = _hook(monkeypatch)
    hook._client.mask = _region(SMALL)
    out = hook(_obs(), _state(tmp_path))
    np.testing.assert_allclose(out[oh.ORIGIN_KEY], BOX_TOP, atol=1e-6)
    (line,) = _lines(tmp_path)
    assert line["source"] == "pixel" and line["boxtable"] is None and line["mask_px"] == 25


def test_no_depth_at_the_pixel_either_is_nan(monkeypatch, tmp_path):
    hook, _ = _hook(monkeypatch)
    hook._client.mask = _region(SMALL)
    depth = _depth()
    depth[45:56, 45:56] = 0.0   # no depth under the mask nor at the pixel
    out = hook(_obs(depth=depth), _state(tmp_path))
    assert out[oh.ORIGIN_KEY].dtype == np.float32 and np.isnan(out[oh.ORIGIN_KEY]).all()
    (line,) = _lines(tmp_path)
    assert line["source"] == "none" and line["origin"] is None and line["pixel"] is None and "error" not in line


def test_a_grounding_error_loses_the_origin_not_the_step(monkeypatch, tmp_path):
    hook, _ = _hook(monkeypatch, ValueError("no grounding JSON object in 'sorry'"))
    obs = _obs()
    out = hook(obs, _state(tmp_path))
    assert out["prompt"] == obs["prompt"] and np.isnan(out[oh.ORIGIN_KEY]).all()
    assert out[oh.SKILL_KEY] == "pick" and out[oh.CALL_KEY] == f"{os.getpid()}-3-0"
    assert hook._client.points == []
    (line,) = _lines(tmp_path)
    assert line["source"] == "none" and line["grounding"] is None
    assert line["error"] == "ValueError: no grounding JSON object in 'sorry'"


def test_no_entry_for_the_role_is_nan(monkeypatch, tmp_path):
    hook, _ = _hook(monkeypatch, {"target": None, "destination": DESTINATION})
    out = hook(_obs(), _state(tmp_path))
    assert np.isnan(out[oh.ORIGIN_KEY]).all() and hook._client.points == []
    (line,) = _lines(tmp_path)
    assert line["role"] == "target" and line["source"] == "none" and "no target" in line["error"]


@pytest.mark.parametrize("gripper, destination, skill, role, point", [
    (0.0, DESTINATION, "pick", "target", TARGET["point"]),
    (0.12, DESTINATION, "place", "destination", DESTINATION["point"]),
    (0.12, None, "place", "target", TARGET["point"]),
])
def test_the_gripper_picks_the_skill_and_the_skill_the_role(monkeypatch, tmp_path, gripper, destination, skill,
                                                           role, point):
    hook, _ = _hook(monkeypatch, {"target": TARGET, "destination": destination})
    out = hook(_obs(gripper=gripper), _state(tmp_path))
    assert out[oh.SKILL_KEY] == skill and hook._client.points == [point]
    (line,) = _lines(tmp_path)
    assert line["skill"] == skill and line["role"] == role


def test_a_call_lasts_until_the_prompt_the_subgoal_or_a_tool_changes(monkeypatch, tmp_path):
    hook, asked = _hook(monkeypatch)
    state = _state(tmp_path)
    pid = os.getpid()
    first = hook(_obs("pick up the red cube"), state)
    state.infer_count = 1
    same = hook(_obs("pick up the red cube", gripper=0.5), state)   # same call: neither recomputed nor re-typed
    assert asked == ["pick up the red cube"] and same[oh.SKILL_KEY] == "pick"
    assert same[oh.CALL_KEY] == first[oh.CALL_KEY] and np.array_equal(same[oh.ORIGIN_KEY], first[oh.ORIGIN_KEY])
    state.infer_count = 2
    assert hook(_obs("put the red cube in the grey bowl"), state)[oh.CALL_KEY] == f"{pid}-3-1"   # new prompt
    state.current_subgoal_idx = 1
    assert hook(_obs("put the red cube in the grey bowl"), state)[oh.CALL_KEY] == f"{pid}-3-2"   # new subgoal
    state.tool_chain_active = True
    obs = _obs("put the red cube in the grey bowl")
    assert hook(obs, state) is obs   # a tool serves this request: left alone
    state.tool_chain_active = False
    after = hook(_obs("put the red cube in the grey bowl"), state)
    assert after[oh.CALL_KEY] == f"{pid}-3-3"   # the first π0.5 request after a tool's opens a call
    assert hook(_obs("put the red cube in the grey bowl"), state)[oh.CALL_KEY] == f"{pid}-3-3"
    state.episode_id = 4
    assert hook(_obs("put the red cube in the grey bowl"), state)[oh.CALL_KEY] == f"{pid}-4-0"
    assert len(asked) == 5
    assert [line["call"] for line in _lines(tmp_path)] == [f"{pid}-3-{i}" for i in range(4)] + [f"{pid}-4-0"]
