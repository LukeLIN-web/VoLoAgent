# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometric re-check of a place subgoal: are enough of the object's instances in its
destination, judged from the scene camera's depth under SAM3 masks of them and of it.

An instance not located now counts as placed only where LADDER's own verifier counts it so
(``hlc/verifiers.py``, ``_CONTAINER_SUFFIX``): the destination is a container whose interior
the camera cannot see into, and the gripper is open, so not holding it.  The container test is
PSP-Agent's (``tools/placing.py``, ``hidden``): the destination's top rises higher above the
support than the object is tall, so an object dropped in sits out of sight below the rim.  On a
destination in plain view an instance not located is not counted placed (the arm may hide it,
or the gripper hold it).  The episode's instances are the most located in any frame so far,
the opening frame included; the object's height is its tallest instance's in the opening frame,
as PSP-Agent measures a held object where it was located before the pick.

Perception follows LADDER's PSP-Agent harness (``perception/scene.py``,
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
    r"(?:(?P<every>each|all|every|both)(?: of)? )?(?:(?:the|a|an) )?(?P<phrase>.+)",
    re.IGNORECASE,
)
# Picks one of several instances of the object, which SAM3 segments under the bare noun.
_ORDINAL = re.compile(r"^(?:second|third|other|another|remaining|next|last|first) ")
# Several objects, or several destinations in a row, in one subgoal.
_COMPOUND = re.compile(r",|;| then | and ")


@dataclass(frozen=True)
class PlaceGoal:
    """A subgoal that puts ``object_phrase`` in or on ``destination_phrase``.

    Both phrases are lowercase, without their leading article or each / all / every / both;
    the object phrase also without a leading ordinal (second, other, remaining, ...).
    """

    object_phrase: str
    destination_phrase: str
    every: bool
    """The subgoal says each / all / every / both: no instance may rest elsewhere."""


def parse_place_subgoal(text: str) -> PlaceGoal | None:
    """The :class:`PlaceGoal` of *text*, or ``None`` when it does not put an object somewhere,
    or is compound: several objects or a chain of destinations (``,``, ``;``, ``then``,
    ``and`` in either phrase, as in "Place the lemon on the plate, then in the grey bin")."""
    match = _SUBGOAL.fullmatch(" ".join(text.split()))
    if match is None:
        return None
    every, object_phrase = _strip_determiner(match["picked"] or match["placed"])
    _, destination_phrase = _strip_determiner(match["destination"])
    if _COMPOUND.search(object_phrase) or _COMPOUND.search(destination_phrase):
        return None
    return PlaceGoal(_ORDINAL.sub("", object_phrase), destination_phrase, every)


def required_count(subgoals: list[str], idx: int) -> int:
    """How many instances must be placed for ``subgoals[idx]`` to be done.

    As LADDER's verified advancer counts (``hlc/advancers/verified.py``): the k-th subgoal
    moving the same object to the same destination needs k there, except on a revisit chain,
    where the one object goes somewhere and comes back ("on the plate", "in the bin", "on the
    plate") and each leg needs 1.  A revisit chain is read off the plan: an earlier subgoal
    sends the same object phrase to another destination.  Subgoals that do not parse are not
    counted; 1 when ``subgoals[idx]`` does not parse.
    """
    goal = parse_place_subgoal(subgoals[idx])
    if goal is None:
        return 1
    destinations = [  # of this object, in subgoals[0] … subgoals[idx]
        parsed.destination_phrase
        for parsed in map(parse_place_subgoal, subgoals[: idx + 1])
        if parsed is not None and parsed.object_phrase == goal.object_phrase
    ]
    if any(destination != goal.destination_phrase for destination in destinations):
        return 1
    return len(destinations)


@dataclass(frozen=True)
class GeomVerdict:
    """Result of :func:`verify_in_destination`.

    ``status`` is ``in`` (enough instances placed), ``not_in`` (too few, whatever the instances
    not located now are: others are located off the destination's footprint, or held over it)
    or ``unknown`` (no support plane or destination, no instance in the episode so far, or
    instances not located now that cannot be counted by absence, so the call is left to the
    VLM).  ``seen`` is the episode's instance count (the most located in any frame, this one
    included); ``placed`` the instances located in the destination, plus those not located now
    when they are counted by absence; ``height`` the tallest fitted-box height above the support
    of the instances located in this frame (m; 0.0 when none is).
    """

    status: str
    reason: str
    n_located: int
    n_in: int
    required: int
    seen: int
    placed: int
    height: float

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
    seen: int = 0,
    object_height: float = 0.0,
    gripper_closed: bool = False,
) -> GeomVerdict:
    """Whether enough of the object's instances (a mask each) are in the masked destination.

    ``depth_m`` is metric depth along the optical axis, (H, W) or (H, W, 1), 0 or non-finite
    where invalid; ``K`` its intrinsics; ``base_from_camera`` the camera → robot base transform
    (OpenCV camera axes, base z up).  Masks may be at another resolution of the same camera.

    A located instance is in when it lies on the destination's footprint and either its foot is
    no higher than the destination's top + ``PLACE_CLEARANCE_M`` or the gripper is open: above
    the top with nothing holding it, it can only rest on what is already in the destination (a
    heap in a bin, a stack on a plate).  Over the footprint, above the top, with the gripper
    closed, it is held over it.  An instance's height is its fitted box's above the support.

    The episode has ``max(seen, n_located)`` instances, *seen* the most located in an earlier
    frame.  Those not located now count as placed (absence) only as LADDER's verifier counts
    them with ``_CONTAINER_SUFFIX``: the destination is a container the camera cannot see into
    and the gripper is open.  It is a container by PSP-Agent's ``hidden`` test: its top rises
    higher above the support than *object_height*, PSP-Agent's ``held_height``: the box of the
    object located before the pick, here the tallest instance in the opening frame (0.0 when
    none was located there, and then no destination is a container).  On a
    destination in plain view, or with the gripper closed, an instance not located is not
    counted placed, and the verdict is ``unknown`` unless the located ones settle it.  With
    *every* none may be located elsewhere, otherwise *required* must be placed.  Without a
    destination mask the instances are still located (``n_located``, ``height``), so an opening
    frame can be counted with ``destination_mask=None``.
    """
    depth = np.nan_to_num(np.asarray(depth_m, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    depth = depth.reshape(depth.shape[:2])
    K = np.asarray(K, dtype=np.float64)
    base_from_camera = np.asarray(base_from_camera, dtype=np.float64)

    support = _support_z(depth, K, base_from_camera)
    if support is None:
        reason = "too little valid depth to estimate the support plane"
        return GeomVerdict("unknown", reason, 0, 0, required, seen, 0, 0.0)

    located = {}
    for index, mask in enumerate(object_masks):
        points = _object_points(depth, K, base_from_camera, mask, support)
        if points is not None:
            located[index] = points
    boxes = {index: fit_box(points, support) for index, points in located.items()}
    feet = {index: float(np.percentile(points[:, 2], 2)) for index, points in located.items()}
    heights = {index: float(boxes[index][1][2]) for index in located}
    height = max(heights.values(), default=0.0)
    n_located = len(located)
    total = max(seen, n_located)
    readings = [
        f"{heights[index]:.3f} m tall, foot {feet[index]:.3f} m" if index in located
        else "not located"
        for index in range(len(object_masks))
    ]

    destination = (
        None if destination_mask is None
        else _blob(_lift(depth, K, base_from_camera, destination_mask))
    )
    if destination is None:
        status, n_in, placed = "unknown", 0, 0
        judged = (
            "no destination mask" if destination_mask is None
            else f"the destination mask has fewer than {MIN_POINTS} usable depth points"
        )
    else:
        centre, extent, yaw = fit_box(destination, support)
        top = float(np.percentile(destination[:, 2], 90))
        to_local = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        n_in = 0
        for index in located:
            offset = to_local @ (boxes[index][0][:2] - centre[:2])
            if np.any(np.abs(offset) > 0.5 * extent[:2]):
                state = "off its footprint"
            elif feet[index] > top + PLACE_CLEARANCE_M and gripper_closed:
                state = "held over it"
            else:
                state = "in"
                n_in += 1
            readings[index] += f", offset ({offset[0]:+.3f}, {offset[1]:+.3f}) m, {state}"
        # PSP-Agent's ``hidden``: an object dropped in sits below the rim, out of sight.
        container = object_height > 0.0 and top - support > object_height
        absence = container and not gripper_closed
        n_elsewhere = n_located - n_in
        hidden = total - n_located
        placed = n_in + hidden if absence else n_in
        if total == 0:
            status = "unknown"
        elif every:
            status = (
                "not_in" if n_elsewhere > 0
                else "in" if hidden == 0 or absence
                else "unknown"
            )
        elif placed >= required:
            status = "in"
        else:
            status = "unknown" if hidden > 0 and not absence else "not_in"
        judged = (
            f"{placed} placed ({n_in} in the destination, {n_elsewhere} elsewhere, {hidden} not "
            f"located; need {'none elsewhere' if every else required}); "
            f"container: {'yes' if container else 'no'} (destination {top - support:.3f} m "
            f"above the support, object {object_height:.3f} m tall); "
            f"absence: {'used' if absence else 'not used'} "
            f"(gripper {'closed' if gripper_closed else 'open'}); "
            f"half-extent {extent[0] / 2:.3f} x {extent[1] / 2:.3f} m"
        )
    reason = (
        f"{n_located} of {len(object_masks)} object masks located (>= {MIN_POINTS} depth points "
        f"above the support at z {support:.3f} m), {total} seen in the episode; {judged}"
        + "".join(f"; mask {index}: {reading}" for index, reading in enumerate(readings))
    )
    return GeomVerdict(status, reason, n_located, n_in, required, total, placed, height)


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
