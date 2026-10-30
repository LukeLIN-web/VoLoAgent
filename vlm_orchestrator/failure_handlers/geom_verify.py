# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometric re-check of a place subgoal: does the object rest in its destination, judged from
the scene camera's depth under SAM3 masks of both.

Follows the perception of LADDER's PSP-Agent harness (``perception/scene.py``,
``perception/fitting.py``, ``tools/geometry.py``), with masks in place of its VLM boxes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

# Same names and values as in LADDER harness perception/scene.py, perception/fitting.py and
# tools/geometry.py.
_DEPTH_RANGE_M = (0.05, 2.5)
_SUPPORT_BAND_M = 0.006
_OBJECT_DEPTH_BAND_M = 0.08
_CLOUD_STRIDE = 4
MIN_POINTS = 12
_EXTENT_PERCENTILE = 2.0
_BLOB_CELL_M = 0.012
PLACE_CLEARANCE_M = 0.02

_PLACE_VERB = r"(?:place|put|move)"
_SOURCE = r"(?: (?:from|off|out of) .+?)?"
_SUBGOAL = re.compile(
    rf"(?:pick(?: up)? (?P<picked>.+?){_SOURCE} and {_PLACE_VERB} (?:it|them)"
    rf"|{_PLACE_VERB} (?P<placed>.+?){_SOURCE})"
    r" (?:on top of|onto|into|inside(?: of)?|in|on) (?P<destination>.+?)\.?",
    re.IGNORECASE,
)
_DETERMINER = re.compile(
    r"(?:(?P<every>each|all|every)(?: of)? )?(?:(?:the|a|an) )?(?P<phrase>.+)", re.IGNORECASE
)


@dataclass(frozen=True)
class PlaceGoal:
    """A subgoal that puts ``object_phrase`` in or on ``destination_phrase``.

    Both phrases are lowercase, without their leading article or each / all.
    """

    object_phrase: str
    destination_phrase: str
    every: bool
    """The subgoal says each / all: every instance seen must be in the destination."""


def parse_place_subgoal(text: str) -> PlaceGoal | None:
    """The :class:`PlaceGoal` of *text*, or ``None`` when it does not put an object somewhere."""
    match = _SUBGOAL.fullmatch(" ".join(text.split()))
    if match is None:
        return None
    every, object_phrase = _strip_determiner(match["picked"] or match["placed"])
    _, destination_phrase = _strip_determiner(match["destination"])
    return PlaceGoal(object_phrase, destination_phrase, every)


def required_count(subgoals: list[str], idx: int) -> int:
    """How many instances must be in the destination for ``subgoals[idx]`` to be done.

    The k-th subgoal moving the same object to the same destination needs k there, as LADDER's
    verifier counts repeated plan lines.  1 when ``subgoals[idx]`` is not a place subgoal.
    """
    goals = [parse_place_subgoal(text) for text in subgoals[: idx + 1]]
    if goals[-1] is None:
        return 1
    key = (goals[-1].object_phrase, goals[-1].destination_phrase)
    return sum(
        1 for goal in goals
        if goal is not None and (goal.object_phrase, goal.destination_phrase) == key
    )


@dataclass(frozen=True)
class GeomVerdict:
    """Result of :func:`verify_in_destination`.

    ``status`` is ``in`` (enough instances rest in the destination), ``not_in`` (they do not:
    off its footprint, or above its top) or ``unknown`` (no destination, or no instance located,
    so the call is left to the VLM).
    """

    status: str
    reason: str
    n_located: int
    n_in: int
    required: int

    def as_log(self) -> dict:
        return asdict(self)


def verify_in_destination(
    depth_m: np.ndarray,
    K: np.ndarray,
    base_from_camera: np.ndarray,
    object_masks: list[np.ndarray],
    destination_mask: np.ndarray | None,
    required: int,
    every: bool,
) -> GeomVerdict:
    """Whether the masked object instances rest in the masked destination.

    ``depth_m`` is metric depth along the optical axis, (H, W) or (H, W, 1), 0 or non-finite
    where invalid; ``K`` its intrinsics; ``base_from_camera`` the camera → robot base transform
    (OpenCV camera axes, base z up).  Masks may be at another resolution of the same camera.
    With *every* all located instances must be in, otherwise *required* of them; fewer than
    *required* located and in is unknown (the rest may be hidden, e.g. deep in a bin).
    """
    if destination_mask is None:
        return GeomVerdict("unknown", "no destination mask", 0, 0, required)
    depth = np.nan_to_num(np.asarray(depth_m, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    depth = depth.reshape(depth.shape[:2])
    K = np.asarray(K, dtype=np.float64)
    base_from_camera = np.asarray(base_from_camera, dtype=np.float64)

    support = _support_z(depth, K, base_from_camera)
    if support is None:
        reason = "too little valid depth to estimate the support plane"
        return GeomVerdict("unknown", reason, 0, 0, required)
    destination = _blob(_lift(depth, K, base_from_camera, destination_mask))
    if destination is None:
        reason = f"the destination mask has fewer than {MIN_POINTS} usable depth points"
        return GeomVerdict("unknown", reason, 0, 0, required)
    centre, extent, yaw = fit_box(destination, support)
    top = float(np.percentile(destination[:, 2], 90))
    to_local = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])

    n_in, readings = 0, []
    for index, mask in enumerate(object_masks):
        points = _object_points(depth, K, base_from_camera, mask, support)
        if points is None:
            continue
        offset = to_local @ (fit_box(points, support)[0][:2] - centre[:2])
        foot = float(np.percentile(points[:, 2], 2))
        if np.any(np.abs(offset) > 0.5 * extent[:2]):
            state = "off its footprint"
        elif foot > top + PLACE_CLEARANCE_M:
            state = "above its top"
        else:
            state = "in"
            n_in += 1
        readings.append(
            f"; mask {index}: offset ({offset[0]:+.3f}, {offset[1]:+.3f}) m, "
            f"foot {foot:.3f} m, {state}"
        )
    n_located = len(readings)

    if n_located == 0:
        status = "unknown"
    elif every:
        status = "in" if n_in == n_located else "not_in"
    elif n_in >= required:
        status = "in"
    else:
        status = "not_in" if n_located >= required else "unknown"
    reason = (
        f"{n_located} of {len(object_masks)} object masks located (>= {MIN_POINTS} depth points "
        f"above the support at z {support:.3f} m), {n_in} in the destination "
        f"(need {'all located' if every else required}; top {top:.3f} m, "
        f"half-extent {extent[0] / 2:.3f} x {extent[1] / 2:.3f} m)"
    )
    return GeomVerdict(status, reason + "".join(readings), n_located, n_in, required)


def _strip_determiner(phrase: str) -> tuple[bool, str]:
    match = _DETERMINER.fullmatch(phrase)
    return match["every"] is not None, match["phrase"].lower()


def _lift(
    depth: np.ndarray, K: np.ndarray, base_from_camera: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Base-frame points of the valid-depth pixels under *mask*, within ``_DEPTH_RANGE_M`` of
    the camera.  *mask* is read on the depth grid by scaling pixel coordinates, as
    ``grasp/server.py`` maps depth pixels into a mask."""
    mask = np.asarray(mask, dtype=bool)
    on_grid = mask[np.ix_(
        np.arange(depth.shape[0]) * mask.shape[0] // depth.shape[0],
        np.arange(depth.shape[1]) * mask.shape[1] // depth.shape[1],
    )]
    rows, columns = np.nonzero(on_grid & (depth > 0.0))
    z = depth[rows, columns]
    camera = np.stack(
        [(columns - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z], axis=1
    )
    distance = np.linalg.norm(camera, axis=1)
    low, high = _DEPTH_RANGE_M
    camera = camera[(distance > low) & (distance < high)]
    return camera @ base_from_camera[:3, :3].T + base_from_camera[:3, 3]


# From LADDER harness perception/scene.py Scene.support_z (MIT License, (c) 2026 INSAIT).
def _support_z(depth: np.ndarray, K: np.ndarray, base_from_camera: np.ndarray) -> float | None:
    sampled = np.zeros(depth.shape, dtype=bool)
    sampled[::_CLOUD_STRIDE, ::_CLOUD_STRIDE] = True
    z = _lift(depth, K, base_from_camera, sampled)[:, 2]
    if z.size < MIN_POINTS:
        return None
    counts, edges = np.histogram(z, bins=max(10, int(np.ptp(z) / 0.005) + 1))
    peak = int(np.argmax(counts))
    return float(np.median(z[(z >= edges[peak]) & (z <= edges[peak + 1])]))


def _object_points(
    depth: np.ndarray,
    K: np.ndarray,
    base_from_camera: np.ndarray,
    mask: np.ndarray,
    support: float,
) -> np.ndarray | None:
    """An instance's blob: points above the support and within ``_OBJECT_DEPTH_BAND_M`` of its
    nearest point along the optical axis, as LADDER's ``Scene._fit`` drops mask bleed onto what
    lies behind."""
    points = _lift(depth, K, base_from_camera, mask)
    points = points[points[:, 2] > support + _SUPPORT_BAND_M]
    if len(points) < MIN_POINTS:
        return None
    along = (points - base_from_camera[:3, 3]) @ base_from_camera[:3, 2]
    return _blob(points[along < along.min() + _OBJECT_DEPTH_BAND_M])


def _blob(points: np.ndarray) -> np.ndarray | None:
    """The connected blob nearest the points' median xy; ``None`` below ``MIN_POINTS``."""
    if len(points) < MIN_POINTS:
        return None
    points = nearest_blob(points, np.median(points[:, :2], axis=0))
    return points if len(points) >= MIN_POINTS else None


# From LADDER harness perception/fitting.py (MIT License, (c) 2026 INSAIT).
def nearest_blob(points_m: np.ndarray, seed_xy: np.ndarray) -> np.ndarray:
    if points_m.shape[0] < MIN_POINTS:
        return points_m
    cells = np.floor(points_m[:, :2] / _BLOB_CELL_M).astype(np.int64)
    members: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(map(tuple, cells)):
        members.setdefault(cell, []).append(index)

    def _distance_to_seed(cell: tuple[int, int]) -> float:
        centre = (np.array(cell) + 0.5) * _BLOB_CELL_M
        return float(np.hypot(centre[0] - seed_xy[0], centre[1] - seed_xy[1]))

    seed_cell = min(members, key=_distance_to_seed)
    seen = {seed_cell}
    frontier = [seed_cell]
    while frontier:
        cx, cy = frontier.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbour = (cx + dx, cy + dy)
                if neighbour in members and neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
    return points_m[np.concatenate([members[c] for c in seen])]


# From LADDER harness perception/fitting.py (MIT License, (c) 2026 INSAIT).
def fit_box(points_m: np.ndarray, support_z: float) -> tuple[np.ndarray, np.ndarray, float]:
    xy = points_m[:, :2]
    centred = xy - xy.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    yaw = float(np.arctan2(axes[0, 1], axes[0, 0]))
    rotated = centred @ np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]).T
    low_xy = np.percentile(rotated, _EXTENT_PERCENTILE, axis=0)
    high_xy = np.percentile(rotated, 100.0 - _EXTENT_PERCENTILE, axis=0)
    top = float(np.percentile(points_m[:, 2], 90))
    mid = (low_xy + high_xy) / 2
    centre_xy = xy.mean(axis=0) + np.array(
        [
            np.cos(yaw) * mid[0] - np.sin(yaw) * mid[1],
            np.sin(yaw) * mid[0] + np.cos(yaw) * mid[1],
        ]
    )
    extent = np.array(
        [high_xy[0] - low_xy[0], high_xy[1] - low_xy[1], max(top - support_z, 1e-3)]
    )
    if extent[1] > extent[0]:
        extent = np.array([extent[1], extent[0], extent[2]])
        yaw += np.pi / 2
    yaw = float((yaw + np.pi / 2) % np.pi - np.pi / 2)
    return np.array([centre_xy[0], centre_xy[1], 0.5 * (top + support_z)]), extent, yaw
