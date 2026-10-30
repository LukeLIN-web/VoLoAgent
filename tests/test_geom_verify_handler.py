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

from vlm_orchestrator.failure_handlers import vlm as vlm_handler
from vlm_orchestrator.failure_handlers.base import ACTION_NEXT
from vlm_orchestrator.failure_handlers.vlm import GEOM_VETO_PATIENCE_STEPS, VLMFailureHandler
from vlm_orchestrator.grasp import client as grasp_client

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
# Each phrase's instances, as the box indices each is made of, in the scenes [_PLATE, cube, ...]
# and [*_DEEP_BIN, cube, cube].
_PLATE_PHRASES = {"cube": [[1], [2]], "plate": [[0]]}
_BIN_PHRASES = {"cube": [[5], [6]], "grey bin": [[0, 1, 2, 3, 4]]}


class _Sam3:
    """SAM3 on a rendered scene: the image is 1 + the index of the box each pixel sees (0 for
    the table), and each phrase segments a mask per instance ``phrases`` lists for it."""

    def __init__(self, phrases: dict[str, list[list[int]]], fail: bool = False):
        self._phrases = phrases
        self._fail = fail
        self.calls: list[tuple[np.ndarray, str]] = []  # each call's image and phrase

    def detect_and_segment_all(self, image, text_prompt):
        self.calls.append((image, text_prompt))
        if self._fail:
            raise RuntimeError("SAM3 is down")
        ids = image[..., 0].astype(int) - 1
        masks = [np.isin(ids, boxes) for boxes in self._phrases[text_prompt]]
        return [(mask, 0.9) for mask in masks if mask.any()]


class _NoisySam3(_Sam3):
    """SAM3 as it answers in the scene, sorted by score: a faint detection of the table (0.2),
    and per instance (0.9) a repeat of it (0.8) and its upper half (0.7)."""

    def detect_and_segment_all(self, image, text_prompt):
        detections = [(image[..., 0] == 0, 0.2)]
        for mask, score in super().detect_and_segment_all(image, text_prompt):
            rows = np.nonzero(mask.any(axis=1))[0]
            detections += [
                (mask, score),
                (mask & (np.arange(mask.shape[1]) != np.nonzero(mask[rows[0]])[0][0]), 0.8),
                (mask & (np.arange(mask.shape[0]) < rows.mean())[:, None], 0.7),
            ]
        return detections


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


@pytest.mark.parametrize(
    "gripper, status", [(1.0, "not_in"), (0.39, "not_in"), (0.1, "in"), (0.0, "in"), (None, "in")]
)
def test_cube_over_the_plate_is_held_only_with_the_gripper_closed(gripper, status):
    """0.10 m over the plate.  Holding an object leaves the gripper part open (0.39: 0.61 of
    its travel open), which reads closed; no gripper position in the observation reads as open."""
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


def test_repeats_parts_and_faint_detections_are_not_instances():
    obs = _obs([_PLATE, _BESIDE_PLATE])
    handler, state = _handler("veto", _NoisySam3(_PLATE_PHRASES)), _state(_ON_THE_PLATE)
    handler.on_episode_start(obs, state)
    _, event = _check(handler, state, obs)
    assert (event["geom"]["seen"], event["geom"]["n_located"]) == (1, 1), event["geom"]["reason"]
    assert event["geom"]["status"] == "not_in" and event["geom"]["vetoed"] is True


def test_the_support_is_the_median_of_the_episodes_estimates(monkeypatch, tmp_path):
    """With the arm hiding most of the table, a frame's own estimate lands on a raised surface
    (0.04 m, which would put the cube's points in the support band).  The support is the median
    of the episode's estimates, the opening frame's included, as LADDER's Scene.support_z takes
    it, so the cube is still located beside the plate."""
    estimates = iter([0.0, 0.0, 0.04])
    monkeypatch.setattr(vlm_handler, "support_plane", lambda *frame: next(estimates))
    obs = _obs([_PLATE, _BESIDE_PLATE])
    handler, state = _handler("veto", _Sam3(_PLATE_PHRASES)), _state(_ON_THE_PLATE, str(tmp_path))
    handler.on_episode_start(obs, state)
    _check(handler, state, obs)
    _, event = _check(handler, state, obs)
    assert event["geom"]["status"] == "not_in", event["geom"]["reason"]
    assert "support at z 0.000 m" in event["geom"]["reason"]
    with np.load(tmp_path / "geom" / f"{state.episode_step:06d}.npz") as saved:
        assert saved["planes"].tolist() == [0.0, 0.0, 0.04]
        assert float(saved["support"]) == 0.0


def test_sam3_failure_is_unknown_and_not_raised():
    obs = _obs([_PLATE, _BESIDE_PLATE])
    handler, state = _handler("veto", _Sam3(_PLATE_PHRASES, fail=True)), _state(_ON_THE_PLATE)
    handler.on_episode_start(obs, state)
    result, event = _check(handler, state, obs)
    assert result.action == ACTION_NEXT
    assert event["geom"]["status"] == "unknown"
    assert "geom verify failed" in event["geom"]["reason"]
    assert event["geom"]["vetoed"] is False


@pytest.mark.parametrize("depth", [None, 0.0])
def test_no_depth_is_unknown_without_segmenting(tmp_path, depth):
    """No depth in the observation, or none valid (no support plane to judge from)."""
    obs = _obs([_PLATE, _BESIDE_PLATE])
    if depth is None:
        del obs["observation/depth_external"]
    else:
        obs["observation/depth_external"][:] = depth
    sam3 = _Sam3(_PLATE_PHRASES)
    handler, state = _handler("veto", sam3), _state(_ON_THE_PLATE, str(tmp_path))
    handler.on_episode_start(obs, state)
    result, event = _check(handler, state, obs)
    assert result.action == ACTION_NEXT
    assert event["geom"]["status"] == "unknown" and event["geom"]["vetoed"] is False
    assert sam3.calls == []
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
    assert [phrase for image, phrase in sam3.calls if np.array_equal(image, start_image)] == ["cube"]
    assert [phrase for _, phrase in sam3.calls] == ["cube", "cube", "plate", "cube", "plate"]


def test_vetoes_stop_after_the_patience_from_the_first():
    """The cube stays beside the plate and the VLM keeps saying next: vetoed until
    GEOM_VETO_PATIENCE_STEPS after the first veto, then the VLM's next stands."""
    obs = _obs([_PLATE, _BESIDE_PLATE])
    handler, state = _handler("veto", _Sam3(_PLATE_PHRASES)), _state(_ON_THE_PLATE)
    handler.on_episode_start(obs, state)
    result, event = _check(handler, state, obs)
    assert result is None and event["geom"]["vetoed"] is True
    state.episode_step += GEOM_VETO_PATIENCE_STEPS - 2  # the next check is 1 step short of it
    result, event = _check(handler, state, obs)
    assert result is None and event["geom"]["vetoed"] is True
    result, event = _check(handler, state, obs)
    assert result.action == ACTION_NEXT
    assert event["geom"]["status"] == "not_in" and event["geom"]["vetoed"] is False
    handler.on_episode_start(obs, state)  # a new episode starts the patience afresh
    result, event = _check(handler, state, obs)
    assert result is None and event["geom"]["vetoed"] is True


class _Response:
    def __init__(self, arrays: dict):
        self.status_code = 200
        self.content = grasp_client.GraspClient._pack_npz(**arrays)

    def raise_for_status(self):
        pass


@pytest.mark.parametrize("used_variant, found", [(False, 1), (True, 0)])
def test_a_simpler_variants_detections_are_not_the_phrase(monkeypatch, used_variant, found):
    """The server falls back to the phrase's last word when the phrase detects nothing
    ("pumpkin" for "orange pumpkin"): those masks may be another object, so none are returned."""
    mask = np.ones((1, 4, 4), dtype=np.uint8)
    reply = _Response({"masks": mask, "scores": np.array([0.9]), "used_variant": np.array(used_variant)})
    monkeypatch.setattr(grasp_client.requests, "post", lambda *args, **kwargs: reply)
    client = grasp_client.GraspClient(url="http://sam3.invalid")
    detections = client.detect_and_segment_all(np.zeros((4, 4, 3), dtype=np.uint8), "orange pumpkin")
    assert len(detections) == found
