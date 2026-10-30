# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""geom_verify on ray-cast tabletop scenes, and its subgoal parser.

The scenes are a z = 0 table, a plate or a bin and 5 cm cubes, all axis-aligned boxes, seen by an
over-shoulder camera 0.6 m above the table looking down at it obliquely.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vlm_orchestrator.failure_handlers.geom_verify import (
    PlaceGoal,
    parse_place_subgoal,
    required_count,
    verify_in_destination,
)

# The over-shoulder camera's intrinsics (fx = fy = 500 px at 1280×720), at 320×180.
_H, _W = 180, 320
_K = np.array([[125.0, 0.0, 160.0], [0.0, 125.0, 90.0], [0.0, 0.0, 1.0]])


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera → base transform of a camera at *eye* looking at *target* (OpenCV axes)."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    transform[:3, 3] = eye
    return transform


_BASE_FROM_CAMERA = _look_at(np.array([0.1, 0.55, 0.6]), np.array([0.5, 0.15, 0.0]))


def _box(base_centre: tuple[float, float, float], size: tuple[float, float, float]):
    x, y, z = base_centre
    sx, sy, sz = size
    return np.array([x - sx / 2, y - sy / 2, z]), np.array([x + sx / 2, y + sy / 2, z + sz])


def _cube(x: float, y: float, z: float):
    return _box((x, y, z), (0.05, 0.05, 0.05))


_PLATE = _box((0.5, 0.2, 0.0), (0.2, 0.2, 0.02))
_BIN = [  # 0.2 × 0.2 × 0.1 m, floor and walls 1 cm thick, where the plate would be
    _box((0.5, 0.2, 0.0), (0.2, 0.2, 0.01)),
    _box((0.405, 0.2, 0.0), (0.01, 0.2, 0.1)),
    _box((0.595, 0.2, 0.0), (0.01, 0.2, 0.1)),
    _box((0.5, 0.105, 0.0), (0.2, 0.01, 0.1)),
    _box((0.5, 0.295, 0.0), (0.2, 0.01, 0.1)),
]
_ON_PLATE = _cube(0.5, 0.2, 0.02)
_BESIDE_PLATE = _cube(0.5, 0.42, 0.0)
_ABOVE_PLATE = _cube(0.5, 0.2, 0.12)
_IN_BIN = _cube(0.5, 0.2, 0.01)


def _render(boxes: list) -> tuple[np.ndarray, np.ndarray]:
    """Depth along the optical axis (0 where a ray hits nothing) and, per pixel, the index of
    the box it sees (-1 for the table)."""
    rows, columns = np.mgrid[0:_H, 0:_W]
    # Unit camera z, so the ray parameter of a hit is its depth.
    rays = np.stack(
        [(columns - _K[0, 2]) / _K[0, 0], (rows - _K[1, 2]) / _K[1, 1], np.ones((_H, _W))],
        axis=-1,
    ).reshape(-1, 3) @ _BASE_FROM_CAMERA[:3, :3].T
    eye = _BASE_FROM_CAMERA[:3, 3]
    seen = np.full(len(rays), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(rays[:, 2] < 0.0, -eye[2] / rays[:, 2], np.inf)
        for index, (low, high) in enumerate(boxes):
            near, far = (low - eye) / rays, (high - eye) / rays
            enter = np.minimum(near, far).max(axis=1)
            leave = np.maximum(near, far).min(axis=1)
            hit = (enter <= leave) & (enter > 0.0) & (enter < depth)
            depth[hit] = enter[hit]
            seen[hit] = index
    depth[np.isinf(depth)] = 0.0
    return depth.reshape(_H, _W).astype(np.float32), seen.reshape(_H, _W)


def _verify(depth, seen, destination, objects, required=1, every=False):
    """The verdict for boxes *destination* as the destination and a mask per box in *objects*."""
    return verify_in_destination(
        depth, _K, _BASE_FROM_CAMERA, [seen == index for index in objects],
        np.isin(seen, destination), required, every,
    )


def test_cube_on_plate_is_in():
    verdict = _verify(*_render([_PLATE, _ON_PLATE]), [0], [1])
    assert verdict.status == "in", verdict.reason
    assert (verdict.n_located, verdict.n_in, verdict.required) == (1, 1, 1)


def test_cube_on_table_beside_plate_is_not_in():
    verdict = _verify(*_render([_PLATE, _BESIDE_PLATE]), [0], [1])
    assert verdict.status == "not_in", verdict.reason
    assert (verdict.n_located, verdict.n_in) == (1, 0)
    assert "off its footprint" in verdict.reason


def test_cube_held_above_plate_is_not_in():
    verdict = _verify(*_render([_PLATE, _ABOVE_PLATE]), [0], [1])
    assert verdict.status == "not_in", verdict.reason
    assert (verdict.n_located, verdict.n_in) == (1, 0)
    assert "above its top" in verdict.reason


def test_cube_in_bin_below_its_rim_is_in():
    verdict = _verify(*_render([*_BIN, _IN_BIN]), [0, 1, 2, 3, 4], [5])
    assert verdict.status == "in", verdict.reason
    assert (verdict.n_located, verdict.n_in) == (1, 1)


def test_empty_object_mask_is_unknown():
    depth, seen = _render([_PLATE])
    verdict = verify_in_destination(
        depth, _K, _BASE_FROM_CAMERA, [np.zeros((_H, _W), dtype=bool)], seen == 0, 1, False
    )
    assert verdict.status == "unknown", verdict.reason
    assert verdict.n_located == 0


def test_no_destination_mask_is_unknown():
    depth, seen = _render([_PLATE, _ON_PLATE])
    verdict = verify_in_destination(depth, _K, _BASE_FROM_CAMERA, [seen == 1], None, 1, False)
    assert verdict.status == "unknown", verdict.reason


@pytest.mark.parametrize(
    "required, every, status", [(1, False, "in"), (2, False, "not_in"), (1, True, "not_in")]
)
def test_two_cubes_one_on_plate(required, every, status):
    depth, seen = _render([_PLATE, _ON_PLATE, _BESIDE_PLATE])
    verdict = _verify(depth, seen, [0], [1, 2], required, every)
    assert verdict.status == status, verdict.reason
    assert (verdict.n_located, verdict.n_in, verdict.required) == (2, 1, required)


def test_fewer_located_than_required_is_unknown():
    verdict = _verify(*_render([_PLATE, _ON_PLATE]), [0], [1], 2)
    assert verdict.status == "unknown", verdict.reason
    assert (verdict.n_located, verdict.n_in, verdict.required) == (1, 1, 2)


def test_nan_and_zero_depth_holes():
    depth, seen = _render([_PLATE, _ON_PLATE])
    holes = np.random.default_rng(0).random(depth.shape)
    depth[holes < 0.05] = np.nan
    depth[(holes >= 0.05) & (holes < 0.1)] = 0.0
    verdict = _verify(depth, seen, [0], [1])
    assert verdict.status == "in", verdict.reason
    log = verdict.as_log()
    assert set(log) == {"status", "reason", "n_located", "n_in", "required"}
    assert json.loads(json.dumps(log)) == log


def test_no_valid_depth_is_unknown():
    depth, seen = _render([_PLATE, _ON_PLATE])
    verdict = _verify(np.full_like(depth, np.nan), seen, [0], [1])
    assert verdict.status == "unknown", verdict.reason


def test_depth_channel_axis_and_masks_at_image_resolution():
    depth, seen = _render([_PLATE, _ON_PLATE])

    def upscaled(mask):
        return np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)

    verdict = verify_in_destination(
        depth[..., None], _K, _BASE_FROM_CAMERA,
        [upscaled(seen == 1)], upscaled(seen == 0), 1, False,
    )
    assert verdict == _verify(depth, seen, [0], [1])


@pytest.mark.parametrize(
    "text, object_phrase, destination_phrase, every",
    [
        ("Place the yellow lemon on the beige plate", "yellow lemon", "beige plate", False),
        ("Place the orange pumpkin in the teal bin", "orange pumpkin", "teal bin", False),
        ("Pick the orange and place it in the grey bin", "orange", "grey bin", False),
        ("Pick the orange from the plate and place it in the grey bin", "orange", "grey bin",
         False),
        ("Pick each lime and place it in the grey bin", "lime", "grey bin", True),
        ("Pick each green lime and place it in the grey bin", "green lime", "grey bin", True),
        ("Pick the yellow banana and place it on the beige plate", "yellow banana", "beige plate",
         False),
        ("Put the red apple onto the white plate.", "red apple", "white plate", False),
        ("Move the lime into the grey bin", "lime", "grey bin", False),
        ("place the Lemon inside the Teal Bin.", "lemon", "teal bin", False),
        ("Put the mug on top of the wooden box", "mug", "wooden box", False),
        ("Pick up all the limes and put them in the grey bin", "limes", "grey bin", True),
    ],
)
def test_parse_place_subgoal(text, object_phrase, destination_phrase, every):
    assert parse_place_subgoal(text) == PlaceGoal(object_phrase, destination_phrase, every)


@pytest.mark.parametrize(
    "text", ["Open the drawer", "Pick up the lemon", "Place the lemon next to the plate"]
)
def test_parse_place_subgoal_rejects_other_subgoals(text):
    assert parse_place_subgoal(text) is None


def test_required_count_counts_repeats_of_the_same_goal():
    subgoals = [
        "Place the orange in the bin",
        "Place the lemon on the plate",
        "Place the orange in the bin",
        "pick the Orange and put it into the bin.",
    ]
    assert [required_count(subgoals, idx) for idx in range(len(subgoals))] == [1, 1, 2, 3]
    assert required_count(["Open the drawer"], 0) == 1
