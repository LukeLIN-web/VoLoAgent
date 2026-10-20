# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""π0.5's reference image at each subgoal start (``--reference-image``, agent-harness #67).

The image (:mod:`vlm_orchestrator.perception.reference_image`) is drawn on the
224 exterior frame π0.5 gets, from two convex hulls, the target's and the
destination's:

``master``
    The master names both on the front camera's raw frame
    (:func:`~vlm_orchestrator.vlm.api.ask_reference_targets`); each phrase is
    segmented there (``--reference-seg``: ``sam3`` the phrase's top SAM3
    detection; ``sam3_point`` its detection under the master's point, else SAM2
    grown from the point, as the grasp tool's ``sam3_point``), and the mask is
    back-projected with the front depth and projected into the exterior camera.
``gt``
    The simulator's mask on the exterior camera of the object and destination
    of the first unsatisfied step in ``gt_state["plan"]``, the evaluator's
    current subtask's ``[object, destination, relation, satisfied]``
    (``robolab_run.py agent --gt-plan``); with no such step, of the current
    subtask's first unsatisfied object alone.

It is computed on the first π0.5 request of each (episode, prompt) and attached
to every π0.5 request until the prompt changes; requests a tool serves instead
of π0.5 are left alone. Each computation appends a line to
``<episode log dir>/reference.jsonl`` and saves the attached image as
``reference/<step_count>.png`` beside it. A failure loses the image (or one
polygon), never the proxy step.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import requests
from PIL import Image

from vlm_orchestrator.grasp.client import GraspClient
from vlm_orchestrator.perception.gt_segmentation import POINT_SNAP_PX, gt_object_mask
from vlm_orchestrator.perception.reference_image import (
    REFERENCE_KEY, backproject, hull_224, hull_224_from_pixels, render,
)
from vlm_orchestrator.vlm.api import ask_reference_targets

logger = logging.getLogger(__name__)

EXTERIOR_224 = "observation/exterior_image_1_left"
FRONT_IMAGE = "observation/front_image_left_raw"
# (depth, K, pos, quat) per camera as RoboLab's VoLo client sends them (policies/volo/metadata.py):
# depth in metres to the image plane, K in pixels, env-local position, world-frame OpenGL (w, x, y, z).
FRONT = ("observation/depth_front", "observation/camera_K_front",
         "observation/camera_pos_front", "observation/camera_quat_front")
EXTERIOR = ("observation/depth_external", "observation/camera_K",
            "observation/camera_pos", "observation/camera_quat")


def segment(client: GraspClient, image: np.ndarray, phrase: str,
            point: tuple[float, float], seg: str) -> tuple[np.ndarray, float, str] | None:
    """``(mask, score, model)`` of ``phrase`` in ``image``; None when SAM3 has
    no detection (``sam3``). ``sam3_point``: the detection under ``point``
    (normalised), else SAM2 from the point, its score the IoU estimate."""
    near = dict(point=point, radius_px=POINT_SNAP_PX) if seg == "sam3_point" else {}
    try:
        mask, score, _ = client.detect_and_segment(image, phrase, **near)
        return mask, score, "sam3"
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 422:
            raise
    if seg == "sam3":
        return None
    mask, iou = client.segment(image, *point)
    return mask, iou, "sam2"


def _camera(obs: dict, keys: tuple[str, str, str, str]) -> tuple[np.ndarray, dict]:
    """A camera's ``(H, W)`` depth and its ``hull_224`` cam dict."""
    depth_key, k_key, pos_key, quat_key = keys
    depth = np.asarray(obs[depth_key], dtype=np.float32)
    depth = depth.reshape(depth.shape[:2])
    return depth, {
        "K": np.asarray(obs[k_key], dtype=np.float64).reshape(3, 3),
        "pos": np.asarray(obs[pos_key], dtype=np.float64).reshape(3),
        "quat": np.asarray(obs[quat_key], dtype=np.float64).reshape(4),
        "hw": depth.shape,
    }


def _vla_bypassed(state) -> bool:
    """The proxy serves a tool's chunk, not π0.5's, after this step (proxy.py's routing after process())."""
    return bool((state.grasp_tool_active and state.grasp_tool_executor is not None)
                or (state.place_tool_active and state.place_tool_executor is not None)
                or state.tool_chain_active)


def _polygon(record: dict, role: str, entry: dict | None, hull_of) -> np.ndarray | None:
    """``hull_of(entry, row)`` for one role, logged in ``record[role]``; its failure loses this polygon only."""
    if entry is None:
        record[role] = None
        return None
    row = record[role] = dict(entry)
    try:
        polygon = hull_of(entry, row)
    except Exception as e:
        logger.exception(f"reference image: {role} {entry}")
        row["error"] = f"{type(e).__name__}: {e}"
        return None
    row["polygon"] = None if polygon is None else np.round(polygon, 2).tolist()
    return polygon


class ReferenceHook:
    """Adds ``observation/reference_image`` to π0.5's requests (``mode`` master or gt)."""

    def __init__(self, ctx, mode: str, seg: str = "sam3"):
        self.ctx, self.mode, self.seg = ctx, mode, seg
        # The grasp tool's perception service (PERCEPTION_SERVER_URL / GRASP_SERVER_*).
        self._client = GraspClient() if mode == "master" else None
        # episode id -> (prompt, its image or None); one small entry per episode.
        self._cache: dict[int, tuple[str, np.ndarray | None]] = {}

    def __call__(self, obs: dict, state) -> dict:
        if _vla_bypassed(state):
            return obs
        prompt = self.ctx.get_prompt(obs)
        cached = self._cache.get(state.episode_id)
        if cached is None or cached[0] != prompt:
            cached = self._cache[state.episode_id] = (prompt, self._compute(obs, state, prompt))
        if cached[1] is None:
            return obs
        obs = dict(obs)
        obs[REFERENCE_KEY] = cached[1]
        return obs

    def _compute(self, obs: dict, state, prompt: str) -> np.ndarray | None:
        record = {"episode": state.episode_id, "step_count": state.infer_count, "prompt": prompt, "mode": self.mode}
        if self.mode == "master":
            record["seg"] = self.seg
        try:
            target, destination = (self._master if self.mode == "master" else self._gt)(obs, prompt, record)
            image = render(np.asarray(obs[EXTERIOR_224]), target, destination)
        except Exception as e:
            logger.exception(f"reference image: episode {state.episode_id} step {state.infer_count}")
            record["error"] = f"{type(e).__name__}: {e}"
            image = None
        record["attached"] = image is not None
        self._write(state, record, image)
        return image

    def _master(self, obs: dict, prompt: str, record: dict):
        image = np.asarray(obs[FRONT_IMAGE], dtype=np.uint8)
        try:
            grounding = ask_reference_targets(image, prompt)
        except ValueError as e:   # no readable grounding JSON in the reply (its text: vlm_calls.jsonl)
            record["grounding_error"] = str(e)
            return None, None
        record["grounding"] = grounding
        depth, front = _camera(obs, FRONT)
        _, exterior = _camera(obs, EXTERIOR)

        def hull_of(entry, row):
            found = segment(self._client, image, entry["phrase"], entry["point"], self.seg)
            row["found"] = found is not None
            if found is None:
                return None
            mask, score, model = found
            row.update(score=float(score), model=model, mask_px=int(mask.sum()))
            return hull_224(backproject(mask, depth, front["K"], front["pos"], front["quat"]), exterior)

        return tuple(_polygon(record, role, grounding[role], hull_of) for role in ("target", "destination"))

    def _gt(self, obs: dict, prompt: str, record: dict):
        gt_state = obs["gt_state"]
        plan = record["plan"] = gt_state["plan"]
        step = next((row for row in plan if not row[3]), None)
        if step is not None:
            target, destination = step[0], step[1]
        else:   # a relation subtask, or every step holds
            target = next((row["object"] for row in gt_state["subtask"]["conditions"] if not row["satisfied"]), None)
            destination = None

        def hull_of(entry, row):
            mask = gt_object_mask(obs, entry["object"], front=False)
            row.update(found=bool(mask.any()), mask_px=int(mask.sum()))
            rows, cols = np.nonzero(mask)
            return hull_224_from_pixels(np.stack([cols, rows], axis=1), mask.shape)

        return tuple(_polygon(record, role, None if name is None else {"object": name}, hull_of)
                     for role, name in (("target", target), ("destination", destination)))

    @staticmethod
    def _write(state, record: dict, image: np.ndarray | None) -> None:
        """One line in reference.jsonl beside rewrites.jsonl, and the attached image as reference/<step_count>.png."""
        if not state.episode_log_dir:
            return
        try:
            line = json.dumps(record)
            if image is not None:
                path = os.path.join(state.episode_log_dir, "reference", f"{state.infer_count}.png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                Image.fromarray(image).save(path)
            with open(os.path.join(state.episode_log_dir, "reference.jsonl"), "a") as f:
                f.write(line + "\n")
        except Exception:
            logger.exception(f"reference image: cannot write the record in {state.episode_log_dir}")
