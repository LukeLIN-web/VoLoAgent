# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLMFailureHandler's geometric re-check (``--geom-verify``) on the ray-cast scenes of
``test_geom_verify.py``, sent as the LADDER plugin sends observations.

SAM3 is a stub that reads each pixel's box index back from the image, and the VLM a function
returning fixed JSON: no server is contacted.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vlm_orchestrator.failure_handlers.base import ACTION_NEXT
from vlm_orchestrator.failure_handlers.vlm import VLMFailureHandler

from test_geom_verify import (
    _ABOVE_PLATE,
    _BASE_FROM_CAMERA,
    _BESIDE_PLATE,
    _CUBE_HEIGHT,
    _DEEP_BIN,
    _IN_BIN,
    _K,
    _ON_PLATE,
    _PLATE,
    _cube,
    _render,
)
from test_vlm_failure_handler import _FakeState

_NEXT = '{"status": "complete", "action": "next", "reason": "the cube is on the plate"}'
_ON_THE_PLATE = "Place the cube on the plate"
# Box indices of each phrase in the scenes [_PLATE, cube, ...] and [*_DEEP_BIN, cube, cube].
_PLATE_PHRASES = {"cube": [1, 2], "plate": [0]}
_BIN_PHRASES = {"cube": [5, 6], "grey bin": [0, 1, 2, 3, 4]}


class _Sam3:
    """SAM3 on a rendered scene: the image is 1 + the index of the box each pixel sees (0 for
    the table), and each phrase segments the boxes ``phrases`` lists for it."""

    def __init__(self, phrases: dict[str, list[int]], fail: bool = False):
        self._phrases = phrases
        self._fail = fail
        self.images: list[np.ndarray] = []  # each detect_and_segment_all call's image

    def _masks(self, image: np.ndarray, phrase: str) -> list[np.ndarray]:
        ids = image[..., 0].astype(int) - 1
        return [ids == index for index in self._phrases[phrase] if (ids == index).any()]

    def detect_and_segment_all(self, image, text_prompt):
        self.images.append(image)
        if self._fail:
            raise RuntimeError("SAM3 is down")
        return [(mask, 0.9) for mask in self._masks(image, text_prompt)]

    def detect_and_segment(self, image, text_prompt):
        mask = np.logical_or.reduce(self._masks(image, text_prompt))
        rows, columns = np.nonzero(mask)
        return mask, 0.9, (columns.min(), rows.min(), columns.max(), rows.max())


def _obs(boxes: list, gripper: float = 0.0, scale: int = 1) -> dict:
    """The LADDER plugin's observation of *boxes*: the image at *scale* × the depth's
    resolution, depth (H, W) float32, K and camera → base flat, the gripper position."""
    depth, ids = _render(boxes)
    image = np.repeat((ids + 1).astype(np.uint8)[..., None], 3, axis=-1)
    return {
        "observation/exterior_image_1_left_raw": image.repeat(scale, axis=0).repeat(scale, axis=1),
        "observation/depth_external": depth,
        "observation/camera_K": _K.reshape(-1),
        "observation/camera_extrinsic": _BASE_FROM_CAMERA.reshape(-1),
        "observation/gripper_position": np.array([gripper], dtype=np.float32),
    }


def _handler(geom_verify: str, sam3: _Sam3) -> VLMFailureHandler:
    """A handler checking every step, whose VLM always says complete / next."""
    handler = VLMFailureHandler(
        vlm_call_fn=lambda system_prompt, user_content: _NEXT,
        image_builder_fn=lambda text, init_img, cur_img, extra, **kw: [
            {"type": "text", "text": text}
        ],
        get_vlm_image_fn=lambda obs: obs["observation/exterior_image_1_left_raw"],
        get_extra_images_fn=lambda obs: None,
        check_interval=1,
        recovery_mode="replan",
        geom_verify=geom_verify,
    )
    handler._geom_client = sam3
    return handler


def _state(subgoal: str, log_dir: str | None = None) -> _FakeState:
    state = _FakeState()
    state.subgoals = [subgoal]
    state.episode_log_dir = log_dir
    return state


def _check(handler: VLMFailureHandler, state: _FakeState, obs: dict):
    """One check on *obs*: the handler's result and the ``vlm_detect`` event it logged."""
    state.tick()
    result = handler.step(obs, state)
    event = state.log_entries[-1]
    assert event["type"] == "vlm_detect"
    return result, event


def test_veto_turns_next_into_continue_when_the_cube_is_beside_the_plate():
    obs = _obs([_PLATE, _BESIDE_PLATE])
    handler, state = _handler("veto", _Sam3(_PLATE_PHRASES)), _state(_ON_THE_PLATE)
    handler.on_episode_start(obs, state)
    result, event = _check(handler, state, obs)
    assert result is None  # in_progress / continue: nothing for the strategy to act on
    assert event["action"] == ACTION_NEXT  # the VLM's own call is what the event records
    assert event["geom"]["status"] == "not_in", event["geom"]["reason"]
    assert event["geom"]["vetoed"] is True


def test_shadow_keeps_next_when_the_cube_is_beside_the_plate():
    obs = _obs([_PLATE, _BESIDE_PLATE])
    handler, state = _handler("shadow", _Sam3(_PLATE_PHRASES)), _state(_ON_THE_PLATE)
    handler.on_episode_start(obs, state)
    result, event = _check(handler, state, obs)
    assert result.action == ACTION_NEXT
    assert event["geom"]["status"] == "not_in", event["geom"]["reason"]
    assert event["geom"]["vetoed"] is False


def test_cube_on_the_plate_is_not_vetoed():
    obs = _obs([_PLATE, _ON_PLATE], scale=2)  # masks at twice the depth's resolution
    handler, state = _handler("veto", _Sam3(_PLATE_PHRASES)), _state(_ON_THE_PLATE)
    handler.on_episode_start(obs, state)
    result, event = _check(handler, state, obs)
    assert result.action == ACTION_NEXT
    assert event["geom"]["status"] == "in", event["geom"]["reason"]
    assert event["geom"]["vetoed"] is False


@pytest.mark.parametrize("gripper, status", [(1.0, "not_in"), (0.0, "in"), (None, "in")])
def test_cube_over_the_plate_is_held_only_with_the_gripper_closed(gripper, status):
    """0.10 m over the plate; no gripper position in the observation reads as open."""
    obs = _obs([_PLATE, _ABOVE_PLATE], gripper=gripper or 0.0)
    if gripper is None:
        del obs["observation/gripper_position"]
    handler, state = _handler("veto", _Sam3(_PLATE_PHRASES)), _state(_ON_THE_PLATE)
    handler.on_episode_start(obs, state)
    _, event = _check(handler, state, obs)
    assert event["geom"]["status"] == status, event["geom"]["reason"]
    assert event["geom"]["vetoed"] is (status == "not_in")


@pytest.mark.parametrize("opening_frame, seen, vetoed", [(True, 2, False), (False, 1, True)])
def test_cube_hidden_in_the_bin_is_counted_against_the_opening_frame(opening_frame, seen, vetoed):
    """Two cubes on the table at the start; now one lies on the deep bin's floor, out of sight,
    and the other is still on the table.  Against the opening frame's two, the hidden one is in
    the bin (taller than a cube, the gripper open); without the opening frame the episode has
    seen one cube, and it is elsewhere."""
    start = _obs([*_DEEP_BIN, _cube(0.35, 0.42, 0.0), _BESIDE_PLATE])
    now = _obs([*_DEEP_BIN, _IN_BIN, _BESIDE_PLATE])
    handler = _handler("veto", _Sam3(_BIN_PHRASES))
    state = _state("Place the cube in the grey bin")
    handler.on_episode_start(start if opening_frame else {}, state)
    _, event = _check(handler, state, now)
    assert event["geom"]["seen"] == seen, event["geom"]["reason"]
    assert event["geom"]["n_located"] == 1
    assert event["geom"]["status"] == ("not_in" if vetoed else "in")
    assert event["geom"]["vetoed"] is vetoed


@pytest.mark.parametrize("gripper, status", [(0.0, "in"), (1.0, "unknown")])
def test_only_cube_hidden_in_the_bin_is_counted_by_the_opening_frames_height(gripper, status):
    """The one cube, on the table at the start, now lies out of sight on the deep bin's floor
    and is located nowhere.  The opening frame's cube height makes the bin a container that
    hides it: with the gripper open it is in; closed, the gripper may hold it, so unknown."""
    handler = _handler("veto", _Sam3(_BIN_PHRASES))
    state = _state("Place the cube in the grey bin")
    handler.on_episode_start(_obs([*_DEEP_BIN, _cube(0.35, 0.42, 0.0)]), state)
    result, event = _check(handler, state, _obs([*_DEEP_BIN, _IN_BIN], gripper=gripper))
    assert event["geom"]["status"] == status, event["geom"]["reason"]
    assert (event["geom"]["seen"], event["geom"]["n_located"]) == (1, 0)
    assert result.action == ACTION_NEXT
    assert event["geom"]["vetoed"] is False


def test_a_held_cube_does_not_change_the_object_height(tmp_path):
    """Held 0.10 m up, the cube's box reads 0.15 m above the table; the object's height stays
    the opening frame's, so a bin is not made to look shallower than the cube."""
    handler, state = _handler("veto", _Sam3(_PLATE_PHRASES)), _state(_ON_THE_PLATE, str(tmp_path))
    handler.on_episode_start(_obs([_PLATE, _BESIDE_PLATE]), state)
    _, held = _check(handler, state, _obs([_PLATE, _cube(0.5, 0.42, 0.10)], gripper=1.0))
    assert held["geom"]["height"] == pytest.approx(_CUBE_HEIGHT + 0.10, abs=0.01)
    _check(handler, state, _obs([_PLATE, _BESIDE_PLATE]))
    with np.load(tmp_path / "geom" / f"{state.episode_step:06d}.npz") as saved:
        assert float(saved["object_height"]) == pytest.approx(_CUBE_HEIGHT, abs=0.01)


def test_cube_located_nowhere_by_the_plate_is_unknown_and_not_vetoed():
    """The cube was on the table at the start; now SAM3 segments it nowhere (behind the arm,
    say).  The plate hides nothing, so the cube is not counted placed: unknown, and the VLM's
    next stands."""
    handler, state = _handler("veto", _Sam3(_PLATE_PHRASES)), _state(_ON_THE_PLATE)
    handler.on_episode_start(_obs([_PLATE, _BESIDE_PLATE]), state)
    result, event = _check(handler, state, _obs([_PLATE]))
    assert result.action == ACTION_NEXT
    assert event["geom"]["status"] == "unknown", event["geom"]["reason"]
    assert (event["geom"]["seen"], event["geom"]["n_located"]) == (1, 0)
    assert "container: no" in event["geom"]["reason"]
    assert event["geom"]["vetoed"] is False


def test_sam3_failure_is_unknown_and_not_raised():
    obs = _obs([_PLATE, _BESIDE_PLATE])
    handler, state = _handler("veto", _Sam3(_PLATE_PHRASES, fail=True)), _state(_ON_THE_PLATE)
    handler.on_episode_start(obs, state)
    result, event = _check(handler, state, obs)
    assert result.action == ACTION_NEXT
    assert event["geom"]["status"] == "unknown"
    assert "geom verify failed" in event["geom"]["reason"]
    assert event["geom"]["vetoed"] is False


def test_no_depth_is_unknown_without_segmenting(tmp_path):
    obs = _obs([_PLATE, _BESIDE_PLATE])
    del obs["observation/depth_external"]
    sam3 = _Sam3(_PLATE_PHRASES)
    handler, state = _handler("veto", sam3), _state(_ON_THE_PLATE, str(tmp_path))
    handler.on_episode_start(obs, state)
    result, event = _check(handler, state, obs)
    assert result.action == ACTION_NEXT
    assert event["geom"]["status"] == "unknown" and event["geom"]["vetoed"] is False
    assert sam3.images == []
    assert not (tmp_path / "geom").exists()


def test_arrays_are_saved_per_check(tmp_path):
    obs = _obs([_PLATE, _BESIDE_PLATE], gripper=1.0)
    depth = obs["observation/depth_external"]
    depth[0, :8] = np.finfo(np.float32).max
    depth[1, :8] = np.nan
    handler, state = _handler("veto", _Sam3(_PLATE_PHRASES)), _state(_ON_THE_PLATE, str(tmp_path))
    handler.on_episode_start(obs, state)
    _, event = _check(handler, state, obs)
    with np.load(tmp_path / "geom" / f"{state.episode_step:06d}.npz") as saved:
        assert saved["depth"].dtype == np.float32
        assert (saved["depth"][:2, :8] == 0.0).all()
        np.testing.assert_array_equal(saved["depth"][2:], depth[2:])
        assert bool(saved["gripper_closed"]) is True
        assert int(saved["seen"]) == 1  # the opening frame's one cube
        assert float(saved["object_height"]) == pytest.approx(_CUBE_HEIGHT, abs=0.01)
        assert tuple(saved["object_masks_shape"]) == (1, *depth.shape)
        assert json.loads(str(saved["verdict"])) == event["geom"]
        assert str(saved["subgoal"]) == _ON_THE_PLATE


def test_opening_frame_is_segmented_once_per_phrase():
    start = _obs([_PLATE, _BESIDE_PLATE])
    now = _obs([_PLATE, _ON_PLATE])
    sam3 = _Sam3(_PLATE_PHRASES)
    handler, state = _handler("veto", sam3), _state(_ON_THE_PLATE)
    handler.on_episode_start(start, state)
    for _ in range(2):
        _check(handler, state, now)
    start_image = start["observation/exterior_image_1_left_raw"]
    assert [np.array_equal(image, start_image) for image in sam3.images] == [True, False, False]
