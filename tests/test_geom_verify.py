# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""geom_verify on ray-cast tabletop scenes, and its subgoal parser.

The scenes are a z = 0 table, a plate or a bin (10 or 25 cm deep) and 5 cm cubes, all
axis-aligned boxes, seen by an over-shoulder camera 0.6 m above the table looking down at it
obliquely.  They are also the handler-level scenes of ``test_geom_verify_handler.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vlm_orchestrator.failure_handlers.geom_verify import (
    MIN_POINTS,
    PlaceGoal,
    parse_place_subgoal,
    required_count,
    sam3_instances,
    support_plane,
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


_CUBE_HEIGHT = 0.05


def _cube(x: float, y: float, z: float):
    return _box((x, y, z), (0.05, 0.05, _CUBE_HEIGHT))


def _bin(height: float) -> list:
    """A 0.2 × 0.2 m bin *height* tall, floor and walls 1 cm thick, where the plate would be."""
    return [
        _box((0.5, 0.2, 0.0), (0.2, 0.2, 0.01)),
        _box((0.405, 0.2, 0.0), (0.01, 0.2, height)),
        _box((0.595, 0.2, 0.0), (0.01, 0.2, height)),
        _box((0.5, 0.105, 0.0), (0.2, 0.01, height)),
        _box((0.5, 0.295, 0.0), (0.2, 0.01, height)),
    ]


_PLATE = _box((0.5, 0.2, 0.0), (0.2, 0.2, 0.02))
_BIN = _bin(0.1)
_DEEP_BIN = _bin(0.25)  # its near walls hide a cube on its floor from the camera
_ON_PLATE = _cube(0.5, 0.2, 0.02)
_BESIDE_PLATE = _cube(0.5, 0.42, 0.0)
_ABOVE_PLATE = _cube(0.5, 0.2, 0.12)  # 0.10 m above the plate's top
_IN_BIN = _cube(0.5, 0.2, 0.01)
_IN_BIN_CORNER = _cube(0.55, 0.25, 0.01)


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
    ids = np.full(len(rays), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(rays[:, 2] < 0.0, -eye[2] / rays[:, 2], np.inf)
        for index, (low, high) in enumerate(boxes):
            near, far = (low - eye) / rays, (high - eye) / rays
            enter = np.minimum(near, far).max(axis=1)
            leave = np.maximum(near, far).min(axis=1)
            hit = (enter <= leave) & (enter > 0.0) & (enter < depth)
            depth[hit] = enter[hit]
            ids[hit] = index
    depth[np.isinf(depth)] = 0.0
    return depth.reshape(_H, _W).astype(np.float32), ids.reshape(_H, _W)


def _verify(depth, ids, destination, objects, required=1, every=False, **counting):
    """The verdict for boxes *destination* as the destination and a mask per box in *objects*;
    *counting* is ``seen``, ``object_height`` and ``gripper_closed``."""
    return verify_in_destination(
        depth, _K, _BASE_FROM_CAMERA, [ids == index for index in objects],
        np.isin(ids, destination), required, every, **counting,
    )


def test_cube_on_plate_is_in():
    """Located on the plate: in, without counting by absence (the plate is no container)."""
    verdict = _verify(*_render([_PLATE, _ON_PLATE]), [0], [1])
    assert verdict.status == "in", verdict.reason
    assert (verdict.n_located, verdict.n_in, verdict.required) == (1, 1, 1)
    assert (verdict.seen, verdict.placed) == (1, 1)
    assert "container: no" in verdict.reason and "absence: not used" in verdict.reason


def test_cube_on_table_beside_plate_is_not_in():
    verdict = _verify(*_render([_PLATE, _BESIDE_PLATE]), [0], [1])
    assert verdict.status == "not_in", verdict.reason
    assert (verdict.n_located, verdict.n_in, verdict.placed) == (1, 0, 0)
    assert verdict.height == pytest.approx(_CUBE_HEIGHT, abs=0.01)
    assert "off its footprint" in verdict.reason


@pytest.mark.parametrize("gripper_closed, status, n_in", [(True, "not_in", 0), (False, "in", 1)])
def test_cube_over_plate_is_held_only_with_the_gripper_closed(gripper_closed, status, n_in):
    """0.10 m over the plate: held over it with the gripper closed; with the gripper open
    nothing holds it, so it rests on what is on the plate."""
    verdict = _verify(*_render([_PLATE, _ABOVE_PLATE]), [0], [1], gripper_closed=gripper_closed)
    assert verdict.status == status, verdict.reason
    assert (verdict.n_located, verdict.n_in) == (1, n_in)
    assert ("held over it" in verdict.reason) is gripper_closed


def test_cube_in_bin_below_its_rim_is_in():
    verdict = _verify(*_render([*_BIN, _IN_BIN]), [0, 1, 2, 3, 4], [5])
    assert verdict.status == "in", verdict.reason
    assert (verdict.n_located, verdict.n_in) == (1, 1)


@pytest.mark.parametrize("half", [1, 2])
def test_the_floor_seen_through_the_bins_opening_is_not_the_bin(half):
    """The bin's mask leaves out the cube in it and takes in a patch of floor at its middle
    (3 × 3 pixels, too few points to fit, or 5 × 5, enough), nearer the middle of the walls'
    points than any wall: the bin is the largest blob, its walls."""
    depth, ids = _render([*_BIN, _IN_BIN_CORNER])
    rows, columns = np.nonzero(ids == 0)
    near_middle = (np.abs(np.arange(_H)[:, None] - int(np.median(rows))) <= half) & (
        np.abs(np.arange(_W)[None] - int(np.median(columns))) <= half
    )
    destination = np.isin(ids, [1, 2, 3, 4]) | ((ids == 0) & near_middle)
    verdict = verify_in_destination(depth, _K, _BASE_FROM_CAMERA, [ids == 5], destination, 1, False)
    assert verdict.status == "in", verdict.reason
    assert (verdict.n_located, verdict.n_in) == (1, 1)


@pytest.mark.parametrize("seen, status", [(2, "in"), (0, "not_in")])
def test_cube_hidden_in_deep_bin_is_counted_by_absence(seen, status):
    """One cube on the deep bin's floor, out of the camera's sight, the other on the table.  The
    bin rises higher than a cube is tall and the gripper is open, so a cube not located now is
    in it: with two seen earlier in the episode, one is.  With none seen earlier there is no
    opening frame and so no object height: the only instance is the one on the table, and the
    bin is not taken for a container."""
    depth, ids = _render([*_DEEP_BIN, _IN_BIN, _BESIDE_PLATE])
    assert not (ids == 5).any()  # the bin's walls hide the cube on its floor
    verdict = _verify(
        depth, ids, [0, 1, 2, 3, 4], [5, 6], seen=seen, object_height=_CUBE_HEIGHT if seen else 0.0
    )
    assert verdict.status == status, verdict.reason
    assert (verdict.n_located, verdict.n_in) == (1, 0)
    assert (verdict.seen, verdict.placed) == (max(seen, 1), max(seen, 1) - 1)
    assert ("container: yes" in verdict.reason) is bool(seen)
    assert ("absence: used" in verdict.reason) is bool(seen)
    assert "mask 0: not located" in verdict.reason and "off its footprint" in verdict.reason


@pytest.mark.parametrize("gripper_closed, status", [(False, "in"), (True, "unknown")])
def test_cube_not_located_counts_in_deep_bin_only_with_the_gripper_open(gripper_closed, status):
    """The one cube seen earlier is not located now, and the deep bin can hide it.  With the
    gripper open it is in the bin; with the gripper closed it may be held out of sight, so the
    call is left to the VLM rather than counted placed."""
    depth, ids = _render([*_DEEP_BIN, _IN_BIN])
    assert not (ids == 5).any()
    verdict = _verify(
        depth, ids, [0, 1, 2, 3, 4], [5],
        seen=1, object_height=_CUBE_HEIGHT, gripper_closed=gripper_closed,
    )
    assert verdict.status == status, verdict.reason
    assert (verdict.n_located, verdict.seen, verdict.height) == (0, 1, 0.0)
    assert verdict.placed == (0 if gripper_closed else 1)
    assert "container: yes" in verdict.reason
    assert ("absence: used" in verdict.reason) is not gripper_closed


@pytest.mark.parametrize(
    "mask, seen, object_height",
    [("empty", 0, 0.0), ("empty", 1, _CUBE_HEIGHT), ("too few points", 1, _CUBE_HEIGHT)],
)
def test_cube_located_nowhere_by_a_plate_is_unknown(mask, seen, object_height):
    """No instance located, the plate in plain view: unknown when none was seen in the episode,
    and also when one was.  The plate rises less above the table than a cube is tall, so it is
    not a container and hides nothing: the cube not located may be behind the arm or in the
    gripper, and is not counted placed."""
    depth, ids = _render([_PLATE])
    object_mask = np.zeros((_H, _W), dtype=bool)
    if mask == "too few points":
        object_mask.flat[np.flatnonzero(ids == 0)[: MIN_POINTS - 1]] = True
    verdict = verify_in_destination(
        depth, _K, _BASE_FROM_CAMERA, [object_mask], ids == 0, 1, False,
        seen=seen, object_height=object_height,
    )
    assert verdict.status == "unknown", verdict.reason
    assert (verdict.n_located, verdict.seen, verdict.placed, verdict.height) == (0, seen, 0, 0.0)
    assert "container: no" in verdict.reason and "absence: not used" in verdict.reason


@pytest.mark.parametrize("destination_mask", [None, np.zeros((_H, _W), dtype=bool)])
def test_no_destination_is_unknown_but_still_locates(destination_mask):
    depth, ids = _render([_PLATE, _ON_PLATE, _BESIDE_PLATE])
    verdict = verify_in_destination(
        depth, _K, _BASE_FROM_CAMERA, [ids == 1, ids == 2], destination_mask, 1, False
    )
    assert verdict.status == "unknown", verdict.reason
    assert (verdict.n_located, verdict.n_in, verdict.seen, verdict.placed) == (2, 0, 2, 0)
    assert verdict.height == pytest.approx(0.07, abs=0.01)  # the cube on the plate's top


@pytest.mark.parametrize(
    "required, every, status", [(1, False, "in"), (2, False, "not_in"), (1, True, "not_in")]
)
def test_two_cubes_one_on_plate(required, every, status):
    depth, ids = _render([_PLATE, _ON_PLATE, _BESIDE_PLATE])
    verdict = _verify(depth, ids, [0], [1, 2], required, every)
    assert verdict.status == status, verdict.reason
    assert (verdict.n_located, verdict.n_in, verdict.required) == (2, 1, required)
    assert (verdict.seen, verdict.placed) == (2, 1)


@pytest.mark.parametrize("seen, status", [(0, "not_in"), (2, "unknown")])
def test_fewer_located_than_required(seen, status):
    """One cube on the plate, two required.  With it the only one seen in the episode, too few;
    with a second seen earlier and not located now, unknown: the plate hides nothing, so the
    second is not counted placed, but it may be behind the arm."""
    verdict = _verify(
        *_render([_PLATE, _ON_PLATE]), [0], [1], 2, seen=seen, object_height=_CUBE_HEIGHT
    )
    assert verdict.status == status, verdict.reason
    assert (verdict.n_located, verdict.n_in, verdict.required) == (1, 1, 2)
    assert (verdict.seen, verdict.placed) == (max(seen, 1), 1)
    assert "container: no" in verdict.reason


@pytest.mark.parametrize(
    "boxes, destination, objects, seen, status",
    [
        # One on the plate, the other not located: the plate hides nothing, so it may be anywhere.
        ([_PLATE, _ON_PLATE], [0], [1], 2, "unknown"),
        # One hidden in the deep bin, the other still on the table.
        ([*_DEEP_BIN, _IN_BIN, _BESIDE_PLATE], [0, 1, 2, 3, 4], [5, 6], 2, "not_in"),
        # The only one hidden in the deep bin, the gripper open.
        ([*_DEEP_BIN, _IN_BIN], [0, 1, 2, 3, 4], [5], 1, "in"),
    ],
)
def test_every_instance(boxes, destination, objects, seen, status):
    depth, ids = _render(boxes)
    verdict = _verify(
        depth, ids, destination, objects, every=True, seen=seen, object_height=_CUBE_HEIGHT
    )
    assert verdict.status == status, verdict.reason


@pytest.mark.parametrize("invalid", [np.nan, np.inf, np.finfo(np.float32).max])
def test_invalid_and_zero_depth_holes(invalid):
    depth, ids = _render([_PLATE, _ON_PLATE])
    holes = np.random.default_rng(0).random(depth.shape)
    depth[holes < 0.05] = invalid
    depth[(holes >= 0.05) & (holes < 0.1)] = 0.0
    verdict = _verify(depth, ids, [0], [1])
    assert verdict.status == "in", verdict.reason
    log = verdict.as_log()
    assert set(log) == {
        "status", "reason", "n_located", "n_in", "required", "seen", "placed", "height"
    }
    assert json.loads(json.dumps(log)) == log


def test_no_valid_depth_is_unknown():
    depth, ids = _render([_PLATE, _ON_PLATE])
    verdict = _verify(np.full_like(depth, np.nan), ids, [0], [1])
    assert verdict.status == "unknown", verdict.reason


def test_depth_channel_axis_and_masks_at_image_resolution():
    depth, ids = _render([_PLATE, _ON_PLATE])

    def upscaled(mask):
        return np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)

    verdict = verify_in_destination(
        depth[..., None], _K, _BASE_FROM_CAMERA,
        [upscaled(ids == 1)], upscaled(ids == 0), 1, False,
    )
    assert verdict == _verify(depth, ids, [0], [1])


def test_given_support_replaces_the_frames_estimate():
    depth, ids = _render([_PLATE, _BESIDE_PLATE])
    plane = support_plane(depth, _K, _BASE_FROM_CAMERA)
    assert plane == pytest.approx(0.0, abs=0.005)
    assert _verify(depth, ids, [0], [1], support=plane) == _verify(depth, ids, [0], [1])
    raised = _verify(depth, ids, [0], [1], support=0.02)
    assert raised.height == pytest.approx(_CUBE_HEIGHT - 0.02, abs=0.01), raised.reason


def test_sam3_instances_are_sam3s_own():
    """As SAM3 lists a phrase's instances: detections scored above 0.5, by score, each
    overlapping no higher-scored one by more than 0.1 mask IoU, so a repeat of an instance and
    a part of it (a held object split by a finger) are not instances; a separate one is."""
    whole = np.zeros((10, 10), dtype=bool)
    whole[2:6, 2:8] = True
    repeat, half, other, faint = (np.zeros_like(whole) for _ in range(4))
    repeat[2:6, 2:7] = True
    half[2:6, 2:5] = True
    other[7:9, 2:8] = True
    faint[0:2, 8:10] = True
    instances = sam3_instances(
        [(faint, 0.4), (half, 0.7), (other, 0.6), (whole, 0.9), (repeat, 0.85), (other, 0.5)]
    )
    assert [mask.tolist() for mask in instances] == [whole.tolist(), other.tolist()]
    assert sam3_instances([]) == []


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
        ("Place the second orange in the grey bin", "orange", "grey bin", False),
        ("Put another lime in the grey bin", "lime", "grey bin", False),
        ("Place both oranges in the grey bin", "oranges", "grey bin", True),
    ],
)
def test_parse_place_subgoal(text, object_phrase, destination_phrase, every):
    assert parse_place_subgoal(text) == PlaceGoal(object_phrase, destination_phrase, every)


@pytest.mark.parametrize(
    "text",
    [
        "Open the drawer",
        "Pick up the lemon",
        "Place the lemon next to the plate",
        "Place the lemon on the plate, then in the grey bin",
        "Place the lemon on the plate then in the grey bin",
        "Place the orange, lemon, and lime in the bin",
        "Pick the lemon and the lime and put them in the bin",
        "Place the lemon on the plate; place the lime in the bin",
    ],
)
def test_parse_place_subgoal_rejects_other_and_compound_subgoals(text):
    assert parse_place_subgoal(text) is None


@pytest.mark.parametrize(
    "subgoals, required",
    [
        (
            [
                "Place the orange in the bin",
                "Place the lemon on the plate",
                "Place the orange in the bin",
                "pick the Orange and put it into the bin.",
            ],
            [1, 1, 2, 3],
        ),
        (["Place the orange in the grey bin"] * 2, [1, 2]),
        (["Place the orange in the grey bin", "Place the second orange in the grey bin"], [1, 2]),
        # A revisit chain: the one lemon goes to the bin and back to the plate.
        (
            [
                "Place the lemon on the plate",
                "Place the lemon in the grey bin",
                "Place the lemon on the plate",
            ],
            [1, 1, 1],
        ),
        # A compound subgoal is not counted.
        (["Place the orange, lemon, and lime in the bin", "Place the orange in the bin"], [1, 1]),
        (["Open the drawer"], [1]),
    ],
)
def test_required_count(subgoals, required):
    assert [required_count(subgoals, idx) for idx in range(len(subgoals))] == required
