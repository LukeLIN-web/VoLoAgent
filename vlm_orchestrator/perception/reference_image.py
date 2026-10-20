# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference image π0.5 DROID gets in its third image slot (agent-harness #67).

It is the exterior-camera 224 frame π0.5 sees (``resize_with_pad`` to 224x224) at
the start of the current subgoal, with everything outside the two highlighted
regions darkened to half, the destination tinted blue and the target tinted red
on top. Each region is a convex hull in that 224 frame: :func:`hull_224` builds
one from world points (simulator ground truth, or a mask back-projected with
:func:`backproject`), :func:`hull_224_from_pixels` from a mask's raw-image pixels.

Pixel coordinates are array indices: pixel (row r, column c) is the point
(x=c, y=r), as in IsaacLab's depth unprojection.

stdlib, numpy and PIL only: the openpi training venv loads this file by path.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

SIZE = 224
REFERENCE_KEY = "observation/reference_image"
GENERIC_PROMPT = "put the red-highlighted object in or on the blue-highlighted object"

GROUNDING_PROMPT = (
    "The image is the robot's front camera view. The robot's current subgoal is: {subgoal}\n"
    "Find the object this subgoal acts on (target) and the place it puts that object in or on "
    "(destination). Reply with JSON only:\n"
    '{{"target": {{"phrase": "...", "point_2d": [x, y]}}, '
    '"destination": {{"phrase": "...", "point_2d": [x, y]}}}}\n'
    '- phrase: color plus category, e.g. "red apple", naming the object the way the subgoal does.\n'
    "- point_2d: one point in the middle of the object's visible surface, x and y on a 0-1000 "
    "scale of this image.\n"
    "- destination is null when the subgoal has no place to put the object (open, close, push, ...)."
)

_TARGET_TINT = (255, 0, 0)
_DESTINATION_TINT = (0, 0, 255)


# --------------------------------------------------------------------------- #
#  Grounding reply
# --------------------------------------------------------------------------- #


def parse_grounding(reply: str) -> dict:
    """-> {"target": {"phrase": str, "point": (x, y)} | None, "destination": {...} | None}; point normalised to [0,1].
    ValueError when the reply has no readable JSON object.

    The answer is the first JSON object in the reply (bare, inside a markdown
    fence or after prose) that has a "target" key. An entry is None when it is
    null, missing, or its phrase is empty; ``point_2d`` is clipped to the
    0-1000 grid. An entry with a phrase but no ``[x, y]`` numbers is a
    ValueError.
    """
    answer = _grounding_object(reply)
    return {"target": _entry(answer.get("target")), "destination": _entry(answer.get("destination"))}


def _grounding_object(reply: str) -> dict:
    decoder = json.JSONDecoder()
    for start, ch in enumerate(reply):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(reply, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "target" in value:
            return value
    raise ValueError(f"no grounding JSON object in {reply[:200]!r}")


def _entry(value) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"grounding entry must be an object or null; got {value!r}")
    phrase = str(value.get("phrase") or "").strip()
    if not phrase:
        return None
    try:
        x, y = (min(max(float(v), 0.0), 1000.0) / 1000.0 for v in value["point_2d"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"grounding entry {phrase!r}: point_2d must be [x, y] numbers; got {value!r}") from e
    return {"phrase": phrase, "point": (x, y)}


# --------------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------------- #


def letterbox_224(px: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """(N,2) raw-image pixel (x, y) -> (N,2) float pixel in the 224x224 frame produced by
    openpi_client.image_tools.resize_with_pad(img, 224, 224) for an image of height/width hw (same integer sizes/offsets)."""
    height, width = hw
    # The integer content size and offsets, computed exactly as _resize_with_pad_pil does.
    ratio = max(width / SIZE, height / SIZE)
    resized_height = int(height / ratio)
    resized_width = int(width / ratio)
    pad_height = max(0, int((SIZE - resized_height) / 2))
    pad_width = max(0, int((SIZE - resized_width) / 2))
    # PIL maps pixel centres: raw x + 0.5 scales by resized/raw to resized x + 0.5.
    scale = np.array([resized_width / width, resized_height / height])
    offset = np.array([pad_width, pad_height])
    return (np.asarray(px, dtype=np.float64).reshape(-1, 2) + 0.5) * scale - 0.5 + offset


def _camera_rotation(quat_wxyz) -> np.ndarray:
    """Camera-to-world rotation with OpenCV axes (x right, y down, z forward)
    from an OpenGL-axes camera orientation: R_opengl @ diag(1, -1, -1)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    rotation_opengl = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return rotation_opengl * np.array([1.0, -1.0, -1.0])


def project(points_world: np.ndarray, K, pos, quat_wxyz) -> tuple[np.ndarray, np.ndarray]:
    """World (env-local) points (N,3) -> raw pixels (N,2) and camera depth (N,) (>0 in front).
    pos/quat_wxyz are the camera pose as written in ep json `cams.<cam>` / observation/camera_* : world frame, OpenGL axes
    (convert with diag(1,-1,-1) as VoLoAgent grasp/camera.py pose_opengl_to_opencv)."""
    points_camera = (np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
                     - np.asarray(pos, dtype=np.float64)) @ _camera_rotation(quat_wxyz)
    homogeneous = points_camera @ np.asarray(K, dtype=np.float64).T
    return homogeneous[:, :2] / homogeneous[:, 2:], points_camera[:, 2]


def backproject(mask: np.ndarray, depth: np.ndarray, K, pos, quat_wxyz, max_points: int = 2000) -> np.ndarray:
    """mask (H,W) bool + depth (H,W) metres (same camera) -> (M,3) world points of mask pixels with finite depth > 0,
    deterministic subsample to <= max_points (evenly spaced over the pixel list)."""
    depth = np.asarray(depth, dtype=np.float64)
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0))
    if len(rows) > max_points:
        keep = np.arange(max_points) * len(rows) // max_points
        rows, cols = rows[keep], cols[keep]
    pixels = np.stack([cols, rows, np.ones_like(cols)], axis=1).astype(np.float64)
    points_camera = pixels @ np.linalg.inv(np.asarray(K, dtype=np.float64)).T * depth[rows, cols][:, None]
    return points_camera @ _camera_rotation(quat_wxyz).T + np.asarray(pos, dtype=np.float64)


def convex_hull(px: np.ndarray) -> np.ndarray:
    """(N,2) -> (H,2) counter-clockwise hull (Andrew monotone chain); fewer than 3 distinct points -> returns them.

    Counter-clockwise in (x, y) axes (positive shoelace area); on screen, with y
    pointing down, it runs clockwise. Collinear points give the two end points.
    """
    ordered = sorted(set(map(tuple, np.asarray(px, dtype=np.float64).reshape(-1, 2).tolist())))   # by x, then y
    if len(ordered) < 3:
        return np.array(ordered, dtype=np.float64).reshape(-1, 2)

    def half(chain_points):
        chain = []
        for p in chain_points:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        return chain[:-1]

    return np.array(half(ordered) + half(ordered[::-1]))


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull_224(points_world: np.ndarray, cam: dict) -> np.ndarray | None:
    """cam = {"K","pos","quat","hw"}; project, drop points with depth <= 0, letterbox_224, convex_hull. None if < 3 points."""
    px, depth = project(points_world, cam["K"], cam["pos"], cam["quat"])
    return hull_224_from_pixels(px[depth > 0], cam["hw"])


def hull_224_from_pixels(px_raw: np.ndarray, hw) -> np.ndarray | None:
    """raw-pixel points (e.g. a ground-truth mask's pixels on the exterior raw image) -> letterbox_224 -> hull; None if < 3."""
    hull = convex_hull(letterbox_224(px_raw, hw))
    return hull if len(hull) >= 3 else None


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #


def render(image224: np.ndarray, target: np.ndarray | None, destination: np.ndarray | None) -> np.ndarray | None:
    """The reference image described above; None when both polygons are None.

    Outside both polygons each value is halved (floor); inside, the
    destination is blended half-and-half with blue, then the target with red
    on top (floor). Polygons may reach past the frame; the input is not
    modified.
    """
    tints = [(_polygon_mask(polygon), colour)
             for polygon, colour in ((destination, _DESTINATION_TINT), (target, _TARGET_TINT))
             if polygon is not None]
    if not tints:
        return None
    image = np.asarray(image224).astype(np.uint16)
    inside = np.logical_or.reduce([mask for mask, _ in tints])
    out = np.where(inside[..., None], image, image // 2)
    for mask, colour in tints:
        out[mask] = (out[mask] + np.array(colour, dtype=np.uint16)) // 2
    return out.astype(np.uint8)


def _polygon_mask(polygon: np.ndarray) -> np.ndarray:
    canvas = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in np.asarray(polygon, dtype=np.float64).tolist()], fill=1)
    return np.asarray(canvas, dtype=bool)
