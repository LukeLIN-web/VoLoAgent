# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""reference_image: the letterbox lands where openpi's resize_with_pad puts a pixel, projection agrees
with grasp/camera.py's pose convention and inverts backproject, the hull, the rendered colours, and
reading the grounding reply."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from vlm_orchestrator.grasp.camera import pose_opengl_to_opencv
from vlm_orchestrator.perception import reference_image as ri

sys.path.insert(0, "/home/user1/volo/openpi/packages/openpi-client/src")
from openpi_client.image_tools import resize_with_pad  # noqa: E402

# The exterior camera of a robolab episode (ep json cams.exterior).
_QUAT = np.array([-0.3929999768733978, -0.19503459334373474, 0.3990708589553833, 0.8051427602767944])
CAM = {
    "K": [[499.9999694824219, 0.0, 640.0], [0.0, 499.9999694824219, 360.0], [0.0, 0.0, 1.0]],
    "pos": [0.05000000074505806, 0.5699999928474426, 0.6600000262260437],
    "quat": (_QUAT / np.linalg.norm(_QUAT)).tolist(),
    "hw": [720, 1280],
}


# --------------------------------------------------------------------------- #
#  letterbox_224 vs openpi_client.image_tools.resize_with_pad
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hw", [(720, 1280), (480, 864)])
def test_letterbox_edges_are_the_resized_content_box(hw):
    h, w = hw
    padded = resize_with_pad(np.full((h, w, 3), 255, np.uint8), ri.SIZE, ri.SIZE)
    rows, cols = np.nonzero(padded[..., 0])
    edges = ri.letterbox_224(np.array([[-0.5, -0.5], [w - 0.5, h - 0.5]]), hw)
    np.testing.assert_allclose(
        edges, [[cols.min() - 0.5, rows.min() - 0.5], [cols.max() + 0.5, rows.max() + 0.5]], atol=1e-9)


def test_letterbox_1280x720_is_224x126_from_row_49():
    edges = ri.letterbox_224(np.array([[-0.5, -0.5], [1279.5, 719.5]]), (720, 1280))
    np.testing.assert_allclose(edges, [[-0.5, 48.5], [223.5, 174.5]], atol=1e-9)


@pytest.mark.parametrize("hw", [(720, 1280), (480, 864)])
def test_letterbox_follows_a_white_dot_through_resize_with_pad(hw):
    h, w = hw
    rows, cols = np.mgrid[0:ri.SIZE, 0:ri.SIZE]
    for x, y in [(37, 23), (w // 2, h // 2), (w - 41, h - 29), (311, h - 77)]:
        image = np.zeros((h, w, 3), np.uint8)
        image[y - 3:y + 4, x - 3:x + 4] = 255   # a dot centred on pixel (x, y)
        resized = resize_with_pad(image, ri.SIZE, ri.SIZE)[..., 0].astype(np.float64)
        centroid = [(resized * cols).sum() / resized.sum(), (resized * rows).sum() / resized.sum()]
        np.testing.assert_allclose(ri.letterbox_224(np.array([[x, y]]), hw)[0], centroid, atol=0.05)


# --------------------------------------------------------------------------- #
#  project / backproject
# --------------------------------------------------------------------------- #


def test_project_agrees_with_camera_py_pose():
    cam_to_world = pose_opengl_to_opencv(np.array(CAM["pos"]), np.array(CAM["quat"]))
    offsets = np.random.default_rng(0).uniform([-0.3, -0.2, 0.4], [0.3, 0.2, 1.2], (50, 3))   # right, down, ahead
    points = np.array(CAM["pos"]) + offsets @ cam_to_world[:3, :3].T
    px, depth = ri.project(points, CAM["K"], CAM["pos"], CAM["quat"])
    in_camera = (np.linalg.inv(cam_to_world) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    uvw = in_camera @ np.array(CAM["K"]).T
    np.testing.assert_allclose(depth, in_camera[:, 2], atol=1e-12)
    np.testing.assert_allclose(px, uvw[:, :2] / uvw[:, 2:], rtol=1e-9)

    # A point on the optical axis lands on the principal point, in front of the camera.
    ahead = np.array(CAM["pos"]) + 0.8 * cam_to_world[:3, 2]
    px, depth = ri.project(ahead[None], CAM["K"], CAM["pos"], CAM["quat"])
    np.testing.assert_allclose(px, [[640.0, 360.0]], atol=1e-6)
    np.testing.assert_allclose(depth, [0.8], atol=1e-9)


def test_backproject_then_project_returns_the_mask_pixels():
    mask = np.zeros((720, 1280), bool)
    mask[300:340, 500:560] = True
    depth = np.tile(np.linspace(0.6, 1.2, 1280), (720, 1))
    depth[310, 510], depth[311, 511], depth[312, 512] = np.nan, 0.0, np.inf
    valid = mask & np.isfinite(depth) & (depth > 0)
    rows, cols = np.nonzero(valid)

    points = ri.backproject(mask, depth, CAM["K"], CAM["pos"], CAM["quat"], max_points=10_000)
    assert points.shape == (valid.sum(), 3) == (mask.sum() - 3, 3)
    px, d = ri.project(points, CAM["K"], CAM["pos"], CAM["quat"])
    np.testing.assert_allclose(px, np.c_[cols, rows], atol=1e-6)
    np.testing.assert_allclose(d, depth[rows, cols], atol=1e-9)


def test_backproject_subsamples_deterministically():
    mask = np.zeros((720, 1280), bool)
    mask[300:340, 500:560] = True
    depth = np.full((720, 1280), 0.9)
    full = ri.backproject(mask, depth, CAM["K"], CAM["pos"], CAM["quat"], max_points=10_000)
    sub = ri.backproject(mask, depth, CAM["K"], CAM["pos"], CAM["quat"], max_points=100)
    assert sub.shape == (100, 3)
    np.testing.assert_array_equal(sub, ri.backproject(mask, depth, CAM["K"], CAM["pos"], CAM["quat"], max_points=100))
    assert all((np.abs(full - p).max(axis=1) < 1e-12).any() for p in sub)
    assert len(ri.backproject(np.zeros((720, 1280), bool), depth, CAM["K"], CAM["pos"], CAM["quat"])) == 0


# --------------------------------------------------------------------------- #
#  convex_hull / hull_224
# --------------------------------------------------------------------------- #


def test_convex_hull_square_with_inner_edge_and_repeated_points():
    points = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0], [2, 1], [0, 0]])
    hull = ri.convex_hull(points)
    np.testing.assert_array_equal(hull, [[0, 0], [2, 0], [2, 2], [0, 2]])
    x, y = hull[:, 0], hull[:, 1]
    assert np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) > 0   # counter-clockwise in (x, y)


def test_convex_hull_degenerate_inputs():
    np.testing.assert_array_equal(ri.convex_hull(np.array([[1, 1], [1, 1], [3, 2]])), [[1, 1], [3, 2]])
    np.testing.assert_array_equal(ri.convex_hull(np.array([[0, 0], [1, 1], [2, 2]])), [[0, 0], [2, 2]])
    assert ri.convex_hull(np.zeros((0, 2))).shape == (0, 2)


def test_hull_224_drops_points_behind_the_camera():
    cam_to_world = pose_opengl_to_opencv(np.array(CAM["pos"]), np.array(CAM["quat"]))
    right, down, forward = cam_to_world[:3, 0], cam_to_world[:3, 1], cam_to_world[:3, 2]
    origin = np.array(CAM["pos"])
    ahead = [origin + 0.8 * forward + a * right + b * down for a, b in [(0, 0), (0.1, 0), (0, 0.1)]]
    behind = [origin - 0.8 * forward + a * right for a in (0.0, 0.2, 0.4)]

    hull = ri.hull_224(np.array(ahead + behind), CAM)
    assert hull is not None and len(hull) == 3
    expected = ri.letterbox_224(ri.project(np.array(ahead), CAM["K"], CAM["pos"], CAM["quat"])[0], CAM["hw"])
    np.testing.assert_allclose(hull, ri.convex_hull(expected))
    assert ri.hull_224(np.array(ahead[:2] + behind), CAM) is None


def test_hull_224_from_pixels():
    hull = ri.hull_224_from_pixels(np.array([[0, 0], [1279, 0], [1279, 719], [0, 719], [640, 360]]), (720, 1280))
    assert len(hull) == 4
    assert ri.hull_224_from_pixels(np.array([[0, 0], [1279, 719], [0, 0]]), (720, 1280)) is None


# --------------------------------------------------------------------------- #
#  render
# --------------------------------------------------------------------------- #

_TARGET = np.array([[10, 10], [60, 10], [60, 60], [10, 60]], float)
_DESTINATION = np.array([[40, 40], [120, 40], [120, 120], [40, 120]], float)


def test_render_darkens_outside_and_tints_target_over_destination():
    image = np.full((224, 224, 3), 101, np.uint8)
    before = image.copy()
    out = ri.render(image, _TARGET, _DESTINATION)
    np.testing.assert_array_equal(image, before)
    assert out.dtype == np.uint8 and out.shape == (224, 224, 3)
    np.testing.assert_array_equal(out[200, 200], [50, 50, 50])     # outside both: halved, floor
    np.testing.assert_array_equal(out[20, 20], [178, 50, 50])      # target only: (101 + red) // 2
    np.testing.assert_array_equal(out[100, 100], [50, 50, 178])    # destination only: (101 + blue) // 2
    np.testing.assert_array_equal(out[50, 50], [152, 25, 89])      # both: blue, then red on top


def test_render_with_one_polygon_or_none():
    image = np.full((224, 224, 3), 101, np.uint8)
    assert ri.render(image, None, None) is None

    out = ri.render(image, None, _DESTINATION)
    np.testing.assert_array_equal(out[20, 20], [50, 50, 50])
    np.testing.assert_array_equal(out[100, 100], [50, 50, 178])

    past_the_frame = np.array([[200, 200], [400, 200], [400, 400], [200, 400]], float)
    out = ri.render(image, past_the_frame, None)
    np.testing.assert_array_equal(out[210, 210], [178, 50, 50])
    np.testing.assert_array_equal(out[100, 100], [50, 50, 50])


# --------------------------------------------------------------------------- #
#  grounding prompt and reply
# --------------------------------------------------------------------------- #


def test_grounding_prompt_formats():
    prompt = ri.GROUNDING_PROMPT.format(subgoal="put the white milk jug in the brown bowl")
    assert "put the white milk jug in the brown bowl" in prompt
    assert '{"target": {"phrase": "...", "point_2d": [x, y]}, "destination": {' in prompt


def test_parse_grounding_plain_json():
    reply = ('{"target": {"phrase": "white milk jug", "point_2d": [500, 250]}, '
             '"destination": {"phrase": "brown bowl", "point_2d": [100, 900]}}')
    assert ri.parse_grounding(reply) == {
        "target": {"phrase": "white milk jug", "point": (0.5, 0.25)},
        "destination": {"phrase": "brown bowl", "point": (0.1, 0.9)},
    }


def test_parse_grounding_markdown_fence_null_destination_and_clipping():
    reply = ('Here it is:\n```json\n{"target": {"phrase": "red apple", "point_2d": [1200, -5]}, '
             '"destination": null}\n```')
    assert ri.parse_grounding(reply) == {"target": {"phrase": "red apple", "point": (1.0, 0.0)}, "destination": None}


def test_parse_grounding_empty_phrase_is_no_entry():
    reply = ('{"target": {"phrase": "", "point_2d": [1, 2]}, '
             '"destination": {"phrase": "blue bowl", "point_2d": [300, 400]}}')
    assert ri.parse_grounding(reply) == {"target": None, "destination": {"phrase": "blue bowl", "point": (0.3, 0.4)}}


@pytest.mark.parametrize("reply", [
    "I cannot find the object.",
    '{"target": {"phrase": "red apple", "point_2d": [1',          # cut off
    '{"point_2d": [500, 500]}',                                      # no "target"
    '{"target": {"phrase": "red apple"}, "destination": null}',      # phrase without a point
])
def test_parse_grounding_unreadable_raises(reply):
    with pytest.raises(ValueError):
        ri.parse_grounding(reply)
