# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The object-frame π0.5 tool's origin at each call (``--tool-origin``, agent-harness #118).

A call is a run of π0.5 requests of one episode under one prompt and one
``current_subgoal_idx`` that no tool interrupts: a new one starts on the
episode's first π0.5 request, on the first π0.5 request after a tool served one,
when the subgoal index or the prompt changes, and once a pick call whose
grounding names a destination, its prompt not a pick-only subgoal (``PICK_ONLY``),
has raised the end effector ``PICK_DONE_LIFT_M`` above where the gripper closed,
where a demonstration's pick call gives way to its place call. The call's line
names the first of these as its ``cause``. At its start:

- the skill is ``place`` when the gripper reads closed (``> GRIPPER_CLOSED``), else ``pick``;
- the master names the target and the destination on the front camera's raw frame
  (:func:`~vlm_orchestrator.vlm.api.ask_reference_targets`); pick uses the target,
  place the destination (the target when there is none);
- SAM2 grows a mask from that point, back-projected with the front depth; the
  origin is :func:`boxtable` of its points on the table top :func:`table_z`
  estimates from the same front depth, else the point's own depth
  back-projection, else NaN.

Every request of the call carries ``observation/tool_origin`` (float32 (3,),
env-local world frame), ``observation/tool_skill`` and ``observation/tool_call``
(one id per call); requests a tool serves are left alone. Each call appends a
line to ``<episode log dir>/origin.jsonl``. A failure loses the call's origin
(NaN), never the proxy step.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np

from vlm_orchestrator.grasp.client import GraspClient
from vlm_orchestrator.grasp.ik import forward_kinematics_robolab
from vlm_orchestrator.perception.reference_image import backproject
from vlm_orchestrator.strategies.reference_hook import FRONT, FRONT_IMAGE, _camera, _vla_bypassed
from vlm_orchestrator.vlm.api import ask_reference_targets

logger = logging.getLogger(__name__)

ORIGIN_KEY = "observation/tool_origin"
SKILL_KEY = "observation/tool_skill"
CALL_KEY = "observation/tool_call"
GRIPPER = "observation/gripper_position"
JOINTS = "observation/joint_position"
# The gripper reads 0.0 open and 0.12-1.0 holding an object, so closed on nothing reads as holding.
GRIPPER_CLOSED = 0.1
# 2 mm bins over the heights a task's table top can have in the robot base frame.
TABLE_BINS = np.arange(-0.05, 0.3, 0.002)
# agent-harness robolab_run.py POLICY_PICK_DONE_LIFT_M: a demonstration's pick call ends once the object is this
# high above where it started; the end effector above where the gripper closed stands in for the object.
PICK_DONE_LIFT_M = 0.08
# A subgoal that only picks ("Pick up the X", "Grasp the X in the Y") ends holding the object and the master's next
# subgoal places it, so a destination its grounding names (often the container the X sits in) starts no place call.
PICK_ONLY = re.compile(r"^\W*(pick|grasp|lift|take|grab)\b"
                       r"(?!.*\b(place|put|drop|move|toss|throw|insert|set|stack it|into|onto)\b)", re.I | re.S)
SAM2_MIN_PX = 50   # VoLo rejects smaller point-prompt masks as degenerate


def table_z(depth: np.ndarray, K, pos, quat) -> float:
    """The table top's height: the median z of the front depth's world points in the fullest ``TABLE_BINS`` bin,
    the table being the largest plane in the front view; NaN with no point in the bins' range."""
    z = backproject(np.ones(depth.shape, bool), depth, K, pos, quat, max_points=depth.size)[:, 2]
    z = z[(z >= TABLE_BINS[0]) & (z < TABLE_BINS[-1])]
    if not len(z):
        return float("nan")
    i = int(np.argmax(np.histogram(z, bins=TABLE_BINS)[0]))
    return float(np.median(z[(z >= TABLE_BINS[i]) & (z < TABLE_BINS[i + 1])]))


def boxtable(points: np.ndarray, table: float) -> np.ndarray:
    """An object's centre from the world points of its mask: the centre of the box (2nd to 98th percentile)
    of the points above the table top at height ``table`` with its bottom at the table top, the underside being
    out of view; NaN with fewer than ``SAM2_MIN_PX`` such points, so whenever ``table`` is NaN (agent-harness
    robolab_run.py Scene.point_estimates)."""
    pts = points[np.isfinite(points).all(1)]
    pts = pts[pts[:, 2] > table + 0.005]   # drop the table the mask spills onto
    if len(pts) < SAM2_MIN_PX:
        return np.full(3, np.nan)
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    return np.r_[(lo[:2] + hi[:2]) / 2, (hi[2] + table) / 2]


def _rounded(point: np.ndarray) -> list[float] | None:
    """A point as logged: None when it is NaN."""
    return np.round(point, 4).tolist() if np.isfinite(point).all() else None


@dataclass
class _Call:
    index: int          # the episode's calls before this one
    prompt: str
    subgoal_idx: int
    keys: dict          # the three keys its requests carry
    skill: str
    destination: bool   # its grounding names a destination
    open: bool = True   # False once a tool serves a request
    closed_z: float | None = None   # the end effector's height where the gripper closed; None while it reads open

    def lifted(self, obs: dict) -> bool:
        """Whether the end effector is more than ``PICK_DONE_LIFT_M`` above where the gripper closed in this call."""
        if float(np.ravel(obs[GRIPPER])[0]) <= GRIPPER_CLOSED:
            self.closed_z = None
            return False
        z = float(forward_kinematics_robolab(np.asarray(obs[JOINTS], dtype=np.float64))[2, 3])
        if self.closed_z is None:
            self.closed_z = z
        return z - self.closed_z > PICK_DONE_LIFT_M


class OriginHook:
    """Adds the object-frame tool's origin, skill and call id to π0.5's requests."""

    def __init__(self, ctx):
        self.ctx = ctx
        # The grasp tool's perception service (PERCEPTION_SERVER_URL / GRASP_SERVER_*).
        self._client = GraspClient()
        # episode id -> its latest call; one small entry per episode.
        self._calls: dict[int, _Call] = {}

    def __call__(self, obs: dict, state) -> dict:
        call = self._calls.get(state.episode_id)
        if _vla_bypassed(state):
            if call is not None:
                call.open = False
            return obs
        prompt = self.ctx.get_prompt(obs)
        cause = ("episode" if call is None else "tool" if not call.open
                 else "subgoal" if call.subgoal_idx != state.current_subgoal_idx
                 else "prompt" if call.prompt != prompt
                 else "lift" if call.skill == "pick" and call.destination and call.lifted(obs)
                 else None)
        if cause is not None:
            index = 0 if call is None else call.index + 1
            call = self._calls[state.episode_id] = self._start(obs, state, prompt, index, cause)
        return {**obs, **call.keys}

    def _start(self, obs: dict, state, prompt: str, index: int, cause: str) -> _Call:
        """A new call; logs it."""
        gripper = float(np.ravel(obs[GRIPPER])[0])
        skill = "place" if gripper > GRIPPER_CLOSED else "pick"
        call_id = f"{os.getpid()}-{state.episode_id}-{index}"
        record = {"episode": state.episode_id, "step_count": state.infer_count,
                  "subgoal_idx": state.current_subgoal_idx, "prompt": prompt, "skill": skill, "role": None,
                  "call": call_id, "cause": cause, "gripper": round(gripper, 3), "grounding": None, "mask_px": None,
                  "sam2_iou": None, "table_z": None, "boxtable": None, "pixel": None}
        start = time.perf_counter()
        try:
            origin, source = self._origin(obs, prompt, skill, record)
            error = None
        except Exception as e:
            logger.exception(f"tool origin: episode {state.episode_id} step {state.infer_count}")
            origin, source, error = np.full(3, np.nan), "none", f"{type(e).__name__}: {e}"
        record.update(origin=_rounded(origin), source=source)
        if error is not None:
            record["error"] = error
        record["wall_s"] = round(time.perf_counter() - start, 4)
        self._write(state, record)
        keys = {ORIGIN_KEY: origin.astype(np.float32), SKILL_KEY: skill, CALL_KEY: call_id}
        destination = bool(record["grounding"] and record["grounding"]["destination"]) and not PICK_ONLY.match(prompt)
        return _Call(index, prompt, state.current_subgoal_idx, keys, skill, destination)

    def _origin(self, obs: dict, prompt: str, skill: str, record: dict) -> tuple[np.ndarray, str]:
        """The origin and where it came from (boxtable, pixel or none); logs the steps in ``record``."""
        image = np.asarray(obs[FRONT_IMAGE], dtype=np.uint8)
        grounding = record["grounding"] = ask_reference_targets(image, prompt)
        role = record["role"] = "destination" if skill == "place" and grounding["destination"] is not None else "target"
        entry = grounding[role]
        if entry is None:
            raise ValueError(f"the grounding names no {role}")
        depth, front = _camera(obs, FRONT)
        h, w = depth.shape
        camera = front["K"], front["pos"], front["quat"]
        table = table_z(depth, *camera)
        record["table_z"] = round(table, 4) if np.isfinite(table) else None
        mask, iou = self._client.segment(image, *entry["point"])
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        record.update(mask_px=int(mask.sum()), sam2_iou=round(float(iou), 4))
        box = boxtable(backproject(mask, depth, *camera, max_points=mask.size), table)
        x, y = entry["point"]
        point_px = np.zeros((h, w), bool)
        point_px[min(int(y * h), h - 1), min(int(x * w), w - 1)] = True
        at_point = backproject(point_px, depth, *camera)
        pixel = at_point[0] if len(at_point) else np.full(3, np.nan)
        record.update(boxtable=_rounded(box), pixel=_rounded(pixel))
        for origin, source in ((box, "boxtable"), (pixel, "pixel")):
            if np.isfinite(origin).all():
                return origin, source
        return np.full(3, np.nan), "none"

    @staticmethod
    def _write(state, record: dict) -> None:
        """One line in origin.jsonl beside rewrites.jsonl."""
        if not state.episode_log_dir:
            return
        try:
            with open(os.path.join(state.episode_log_dir, "origin.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            logger.exception(f"tool origin: cannot write the record in {state.episode_log_dir}")
