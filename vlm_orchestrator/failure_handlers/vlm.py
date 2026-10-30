# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM-based failure detection and recovery handler.

Periodically sends BEFORE + NOW images to a VLM and asks it to assess
the current subgoal status and decide the next action.  Replaces both
GT-based failure detection and the old VLM subgoal completion check
with a single unified call.

The VLM outputs two independent fields:

* **status** — diagnosis (``complete`` / ``failure`` / ``in_progress``)
* **action** — decision (``next`` / ``replan`` / ``continue`` /
  ``grasp_tool``)

The available actions are configured by ``--recovery-mode``:

* ``replan``       → next, continue, replan
* ``replan_grasp`` → next, continue, replan, grasp_tool
* ``grasp``        → next, continue, grasp_tool
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

import numpy as np
import requests

from vlm_orchestrator.grasp.client import GraspClient
from vlm_orchestrator.vlm import POINT_CONVENTION, parse_json, parse_norm_point

from .base import (
    ACTION_CONTINUE,
    ACTION_GRASP,
    ACTION_NEXT,
    ACTION_PLACE,
    ACTION_REPLAN,
    FailureHandler,
    HandlerResult,
    STATUS_COMPLETE,
    STATUS_FAILURE,
    STATUS_IN_PROGRESS,
)
from .geom_verify import (
    GeomVerdict,
    parse_place_subgoal,
    required_count,
    verify_in_destination,
)

logger = logging.getLogger(__name__)

# Scene camera for the geometric re-check: raw RGB, metric depth (0 = invalid),
# flat 3×3 intrinsics, flat 4×4 camera→base (OpenCV axes).
_GEOM_OBS_KEYS = (
    "observation/exterior_image_1_left_raw",
    "observation/depth_external",
    "observation/camera_K",
    "observation/camera_extrinsic",
)


# ======================================================================
# Available action sets per recovery mode
# ======================================================================

_ACTION_SETS: dict[str, set[str]] = {
    "replan": {ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN},
    "replan_grasp": {
        ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN, ACTION_GRASP,
    },
    "grasp": {ACTION_NEXT, ACTION_CONTINUE, ACTION_GRASP},
    # Place-only analogs of the grasp tokens.
    "place": {ACTION_NEXT, ACTION_CONTINUE, ACTION_PLACE},
    "replan_place": {
        ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN, ACTION_PLACE,
    },
    # Combined: VLM may pick either tool based on failure context.
    "tools": {ACTION_NEXT, ACTION_CONTINUE, ACTION_GRASP, ACTION_PLACE},
    "replan_tools": {
        ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN,
        ACTION_GRASP, ACTION_PLACE,
    },
}


# ======================================================================
# Prompt
# ======================================================================

def _build_system_prompt(
    available_actions: set[str],
    *,
    use_front_camera: bool = False,
    stack_mode_enabled: bool = False,
    point_tools: frozenset[str] = frozenset(),
    primary_label: str = "External camera",
) -> str:
    """Build the system prompt with the available action vocabulary.

    For each tool in ``point_tools`` the VLM must also give a point in the
    primary NOW image (``grasp_point`` / ``place_point``).

    When ``use_front_camera`` is True, the prompt warns the VLM that
    the front (egocentric_mirrored) camera view is left-right flipped
    relative to the robot's actual perspective, so the VLM should use
    the robot's frame when emitting any directional references.
    """

    action_descriptions = []
    if ACTION_NEXT in available_actions:
        action_descriptions.append(
            '- "next" — this subgoal is COMPLETE. The target object has '
            "moved to its destination. Advance to the next subgoal."
        )
    if ACTION_CONTINUE in available_actions:
        action_descriptions.append(
            '- "continue" — the robot is making progress or the subgoal '
            "is not yet complete. Keep working with the current instruction."
        )
    if ACTION_REPLAN in available_actions:
        action_descriptions.append(
            '- "replan" — something went wrong (object dropped, stuck, '
            "wrong object picked, no visible progress) OR the current "
            "plan is not working. Replan the remaining subtasks from "
            "the current scene state."
        )
    if ACTION_GRASP in available_actions:
        action_descriptions.append(
            '- "grasp_tool" — fire only for WRONG-OBJECT recovery: the '
            "gripper holds (or is closing on) an object whose identity "
            "differs from the subgoal's target. Do NOT fire while the "
            "robot is still attempting the correct object — that is "
            'in-progress execution; emit "continue". When firing, set '
            '"grasp_target" to the noun phrase of the correct object.'
            + (
                ' Also set "grasp_point" to a point on that object in the '
                f"NOW {primary_label} image: {POINT_CONVENTION}."
                if "grasp" in point_tools else ""
            )
        )
    if ACTION_PLACE in available_actions:
        action_descriptions.append(
            '- "place_tool" — fire only for WRONG-DESTINATION recovery: '
            "the held object is heading to or already released at a "
            "destination different from the subgoal's destination. Do "
            "NOT fire while the robot is en route to the correct "
            'destination — that is in-progress execution; emit '
            '"continue". When firing, set "place_destination" to a '
            "FREEFORM destination phrase. PHRASING RULES:\n"
            "  1. Inside a container → \"in <container>\" "
            '(e.g. "in the white bowl").\n'
            "  2. Beside/near an anchor → prepend \"empty space\" "
            '(e.g. "empty space next to the green lemon").\n'
            "  3. Open table → \"empty space on the table\".\n"
            "  4. On a surface object → \"on <surface>\" "
            '(e.g. "on the wire rack shelf").\n'
            + (
                'Also set "place_point" to where the held object should go '
                f"in the NOW {primary_label} image (on the destination object, "
                f"or on the open table spot): {POINT_CONVENTION}.\n"
                if "place" in point_tools else ""
            )
            + "Optionally also set \"place_held_object\" with the noun "
            "phrase of the object in the gripper.\n"
            "LIMITATION: place_tool is unreliable for high containers "
            '(e.g. a shelf, a high vase). For those destinations prefer '
            '"continue" (let the VLA finish) or "replan" instead of '
            "firing place_tool."
            + (
                "\n"
                "STACK CONTROL: place_tool accepts an optional boolean "
                '"place_stack" arg.\n'
                '  - "place_stack": false (DEFAULT) — plain top-down '
                "release. Use for ordinary pick-and-place: dropping an "
                "object into a bowl/bin, onto an open surface, or beside "
                "another object.\n"
                '  - "place_stack": true — keep the held object\'s current '
                "orientation at release instead of forcing top-down. Use "
                "ONLY when setting an object ON TOP of another in a way "
                "that preserves how it is held (stacking a block on a "
                "block, nesting a lid on a container). When in doubt, "
                'prefer false.'
                if stack_mode_enabled
                else ""
            )
        )

    actions_block = "\n".join(action_descriptions)

    front_cam_block = ""
    if use_front_camera:
        front_cam_block = """\

NOTE on camera orientation: the "Front camera" view is BOTH L/R \
and front/behind FLIPPED from the robot's perspective. \
image-LEFT↔robot-RIGHT, image-TOP↔robot-BEHIND.

To avoid frame confusion, describe targets by VISUAL FEATURES \
(color, type, proximity to landmarks) in every output field — \
NOT by left/right/front/behind. Example: "the grey container next \
to the red block" instead of "the right bin". If the subgoal uses a \
direction, first apply the flip to find which object it means in the \
image, then re-describe by visual features."""

    return f"""\
You are monitoring a robot arm executing a tabletop manipulation task.

You receive two sets of images:
- **BEFORE** images: the scene at the START of the episode (for reference).
- **NOW** images: the current scene.
{front_cam_block}

You also receive the overall task instruction, the current subgoal, and \
any remaining subgoals after the current one.

HOW TO REASON:
- Use the BEFORE images and overall task to understand the goal.
- Use the NOW images, current subgoal, and remaining subgoals to \
  understand the current state and what still needs to happen.
- Decide whether the robot can keep executing the current subgoal \
  from the NOW scene. If yes → "in_progress" / "continue". \
  If the current subgoal is already done → "complete" / "next". \
  If something actively prevents progress (wrong object in gripper, \
  object dropped, robot stuck) → "failure".
- The subgoal list may have been replanned mid-episode. The current \
  subgoal and remaining subgoals already reflect what needs to be \
  done from the current state — do not treat them as stale.

Your job:
1. **Assess status** — look at the NOW images and determine the status \
   of the CURRENT SUBGOAL.
2. **Decide action** — choose the best action for the robot.

STATUS:
- "complete" — the current subgoal is done (the target object is at \
  its destination in the NOW images).
- "failure" — the robot is clearly stuck, holding the wrong object, \
  or the target object is unreachable.
- "in_progress" — the subgoal is not yet complete but is still \
  achievable. The robot may be approaching, reaching, or grasping.

ACTION (what to do?):
{actions_block}

IMPORTANT RULES:
- Only say "complete" if you see CLEAR visual evidence the subgoal is \
  done in the NOW images (target object at its destination).
- Say "in_progress" if the robot has not yet started or is still \
  working on the current subgoal — this is NORMAL, not a failure.
- Only say "failure" if the robot is clearly stuck, has picked up the \
  WRONG object, or has been making no progress for a long time.
- When in doubt, say "in_progress" with action "continue".
- Do NOT replan just because the scene looks different from BEFORE. \
  Only replan if the current subgoal is impossible or the plan no \
  longer makes sense given the NOW scene.

Output ONLY valid JSON (no markdown, no explanation):
{{"status": "...", "action": "...", "reason": "brief explanation"}}"""


# ======================================================================
# Parser
# ======================================================================

def _parse_vlm_detection(
    raw: str,
    available_actions: set[str],
    point_tools: frozenset[str] = frozenset(),
) -> HandlerResult:
    """Parse VLM JSON response into a HandlerResult.

    Falls back to ``(in_progress, continue)`` on parse failure; a tool
    action whose required point (``point_tools``) is missing or off the
    image falls back to ``continue``.
    """
    try:
        data = parse_json(raw)
    except (ValueError, KeyError):
        logger.warning(f"  VLM detection: failed to parse JSON: {raw[:200]}")
        return HandlerResult(
            status=STATUS_IN_PROGRESS,
            action=ACTION_CONTINUE,
            reason="parse_failure",
        )

    status = data.get("status", STATUS_IN_PROGRESS)
    action = data.get("action", ACTION_CONTINUE)
    reason = data.get("reason", "")
    grasp_target = data.get("grasp_target")
    place_destination = data.get("place_destination")
    place_target = data.get("place_target")
    place_relation = data.get("place_relation", "in")
    place_held_object = data.get("place_held_object")
    place_stack = bool(data.get("place_stack", False))

    # Validate status
    if status not in (STATUS_COMPLETE, STATUS_FAILURE, STATUS_IN_PROGRESS):
        logger.warning(
            f"  VLM detection: invalid status {status!r}, "
            f"defaulting to in_progress"
        )
        status = STATUS_IN_PROGRESS

    # Validate action against available set
    if action not in available_actions:
        logger.warning(
            f"  VLM detection: action {action!r} not in "
            f"{available_actions}, defaulting to continue"
        )
        action = ACTION_CONTINUE

    points = {"grasp": None, "place": None}
    for tool, act in (("grasp", ACTION_GRASP), ("place", ACTION_PLACE)):
        if action != act or tool not in point_tools:
            continue
        try:
            points[tool] = parse_norm_point(data.get(f"{tool}_point"))
        except ValueError as e:
            logger.warning(
                f"  VLM detection: {act} needs {tool}_point ({e}); "
                f"defaulting to continue"
            )
            action = ACTION_CONTINUE

    if place_relation not in ("in", "on", "on_top_of"):
        logger.warning(
            f"  VLM detection: invalid place_relation {place_relation!r}, "
            f"defaulting to 'in'"
        )
        place_relation = "in"

    return HandlerResult(
        status=status,
        action=action,
        reason=reason,
        grasp_target=grasp_target,
        place_destination=place_destination,
        place_target=place_target,
        place_relation=place_relation,
        place_held_object=place_held_object,
        place_stack=place_stack,
        grasp_point=points["grasp"],
        place_point=points["place"],
    )


# ======================================================================
# Handler
# ======================================================================

class VLMFailureHandler(FailureHandler):
    """Periodic VLM-based failure detection and action selection.

    Parameters
    ----------
    vlm_call_fn:
        Callable ``(system_prompt, user_content) -> str`` that makes
        a VLM API call.  Provided by the strategy layer.
    image_builder_fn:
        Callable ``(text, initial_image, current_image, extra_images,
        initial_extra_images) -> list[dict]`` that builds the
        multi-image user content.  Provided by the strategy layer.
    get_vlm_image_fn:
        Callable ``(obs) -> np.ndarray | None`` that extracts the VLM
        image from an observation dict.
    get_extra_images_fn:
        Callable ``(obs) -> list[np.ndarray] | None`` that extracts
        extra camera images from an observation dict.
    check_interval:
        Number of sim steps between VLM checks.  Gated by step delta
        against ``state.episode_step`` so cadence is invariant across
        VLAs whose action chunks differ in length.
    recovery_mode:
        Controls available actions: ``"replan"``, ``"replan_grasp"``,
        or ``"grasp"``.
    point_tools:
        Tools (``"grasp"`` / ``"place"``) whose actions must carry a point
        in the primary NOW image (their ``gt_sim`` resolves targets from it).
    geom_verify:
        ``"off"`` | ``"shadow"`` | ``"veto"``.  Not off: every check also
        segments the current place subgoal's object and destination with SAM3
        (``GraspClient`` perception server) and judges from scene depth whether
        the object is in the destination; the verdict goes into the
        ``vlm_detect`` event as ``geom`` and its arrays into
        ``<episode log>/geom/<step>.npz``.  ``"veto"`` also turns a VLM
        ``next`` into ``continue`` when the verdict is ``not_in``.
    """

    def __init__(
        self,
        vlm_call_fn: Callable,
        image_builder_fn: Callable,
        get_vlm_image_fn: Callable,
        get_extra_images_fn: Callable,
        check_interval: int = 80,
        recovery_mode: str = "replan",
        primary_label: str = "External camera",
        extra_labels: list[str] | None = None,
        use_front_camera: bool = False,
        stack_mode_enabled: bool = False,
        point_tools: frozenset[str] = frozenset(),
        geom_verify: str = "off",
    ):
        self._vlm_call = vlm_call_fn
        self._build_check_message = image_builder_fn
        self._get_vlm_image = get_vlm_image_fn
        self._get_extra_images = get_extra_images_fn
        self._check_interval = check_interval
        self._primary_label = primary_label
        self._extra_labels = extra_labels
        self._geom_verify = geom_verify
        self._geom_client = GraspClient() if geom_verify != "off" else None

        if recovery_mode not in _ACTION_SETS:
            raise ValueError(
                f"Invalid recovery_mode {recovery_mode!r} for VLM handler, "
                f"expected one of {list(_ACTION_SETS)}"
            )
        self._available_actions = _ACTION_SETS[recovery_mode]
        self._point_tools = point_tools
        self._system_prompt = _build_system_prompt(
            self._available_actions,
            use_front_camera=use_front_camera,
            stack_mode_enabled=stack_mode_enabled,
            point_tools=point_tools,
            primary_label=primary_label,
        )

        # Per-episode state — sim step at which the last VLM check
        # fired.  Cadence is gated by step delta against
        # ``state.episode_step`` so that VLA chunks of different sizes
        # produce the same wall-clock check frequency.
        self._step_at_last_check: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_episode_start(self, obs: dict, state: Any) -> None:
        self._step_at_last_check = state.episode_step

    def on_subgoal_advanced(self, obs: dict, state: Any, idx: int) -> None:
        self._step_at_last_check = state.episode_step

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def step(self, obs: dict, state: Any) -> HandlerResult | None:
        """Run periodic VLM detection.

        Returns ``None`` if it's not time to check yet.
        """
        # Not time to check yet
        if (state.episode_step - self._step_at_last_check
                < self._check_interval):
            return None

        # No subgoals to check against
        if not state.subgoals:
            logger.warning("  VLM handler: no subgoals, skipping")
            return None

        self._step_at_last_check = state.episode_step
        logger.info(
            f"  VLM handler: checking subgoal {state.current_subgoal_idx+1}/"
            f"{len(state.subgoals)} at step {state.episode_step}"
        )

        # Get current image
        current_image = self._get_vlm_image(obs)
        if current_image is None:
            logger.warning("  VLM handler: current_image is None, skipping")
            return None

        # Build context
        sg_idx = state.current_subgoal_idx
        current_sg = state.subgoals[sg_idx]
        total = len(state.subgoals)

        remaining = state.subgoals[sg_idx + 1:]
        if remaining:
            remaining_str = "Remaining subgoals after this:\n" + "\n".join(
                f"  {i}. {s}"
                for i, s in enumerate(remaining, start=sg_idx + 2)
            )
        else:
            remaining_str = "This is the LAST subgoal."

        text = (
            f'Overall task: "{state.original_instruction}"\n'
            f'Current subgoal ({sg_idx + 1}/{total}): "{current_sg}"\n'
            f"{remaining_str}\n\n"
            f"Assess the current subgoal status and decide the action."
        )

        # Build message with BEFORE + NOW images
        extra_images = self._get_extra_images(obs)
        user_content = self._build_check_message(
            text,
            state.initial_image,
            current_image,
            extra_images,
            initial_extra_images=state.initial_extra_images,
            primary_label=self._primary_label,
            extra_labels=self._extra_labels,
        )

        # VLM call
        t0 = time.time()
        try:
            raw = self._vlm_call(self._system_prompt, user_content)
            elapsed = time.time() - t0
        except Exception as e:
            logger.warning(f"  VLM detection call failed: {e}; continuing")
            return None

        logger.info(f"  VLM detect raw ({elapsed:.1f}s): {raw[:500]}")

        # Parse result
        result = _parse_vlm_detection(raw, self._available_actions, self._point_tools)
        result.extra["vlm_latency_s"] = round(elapsed, 2)
        result.extra["vlm_raw"] = raw

        # Log
        status_icon = {
            STATUS_COMPLETE: "✓",
            STATUS_FAILURE: "⚠",
            STATUS_IN_PROGRESS: "…",
        }.get(result.status, "?")

        action_icon = {
            ACTION_NEXT: "→",
            ACTION_REPLAN: "↻",
            ACTION_CONTINUE: "…",
            ACTION_GRASP: "🤏",
        }.get(result.action, "?")

        logger.info(
            f"  {status_icon} VLM detect [{sg_idx + 1}/{total}]: "
            f"status={result.status}, action={result.action} "
            f"{action_icon} — {result.reason} ({elapsed:.1f}s)"
        )

        geom = None
        if self._geom_verify != "off":
            verdict, arrays = self._geom_verdict(obs, state.subgoals, sg_idx)
            vetoed = (self._geom_verify == "veto"
                      and result.action == ACTION_NEXT
                      and verdict.status == "not_in")
            geom = verdict.as_log() | {"vetoed": vetoed}
            logger.info(
                f"  geom verify [{sg_idx + 1}/{total}]: "
                f"status={verdict.status}, in={verdict.n_in}/{verdict.required}, "
                f"located={verdict.n_located}{', VETO next' if vetoed else ''} "
                f"— {verdict.reason}"
            )
            if arrays is not None:
                self._save_geom(state, {
                    **arrays, "verdict": json.dumps(geom), "subgoal": current_sg,
                })

        # Always log the VLM check to state (even for in_progress/continue)
        self._store_check_result(state, result)
        event = {
            "type": "vlm_detect",
            "status": result.status,
            "action": result.action,
            "reason": result.reason,
            "subgoal_idx": sg_idx,
            "subgoal": current_sg,
            "grasp_target": result.grasp_target,
            "grasp_point": result.grasp_point,
            "place_point": result.place_point,
            "vlm_latency_s": round(elapsed, 2),
            "vlm_raw": raw[:500],
            "step_count": state.infer_count,
        }
        if geom is not None:
            event["geom"] = geom
        state.log(event)

        if geom is not None and geom["vetoed"]:
            result.status = STATUS_IN_PROGRESS
            result.action = ACTION_CONTINUE
            result.reason = f"[geom veto] {verdict.reason}"

        # Return None for in_progress + continue (no action needed)
        if (result.status == STATUS_IN_PROGRESS
                and result.action == ACTION_CONTINUE):
            return None

        return result

    # ------------------------------------------------------------------
    # Geometric re-check
    # ------------------------------------------------------------------

    def _geom_verdict(
        self, obs: dict, subgoals: list[str], sg_idx: int,
    ) -> tuple[GeomVerdict, dict | None]:
        """Whether subgoal *sg_idx*'s object is in its destination, from SAM3
        masks on the scene depth.

        Also returns, when the verdict was judged from masks, the arrays it
        was judged from (packed for ``geom/<step>.npz``).
        """
        image, depth, K, extrinsic = (obs.get(k) for k in _GEOM_OBS_KEYS)
        if any(a is None for a in (image, depth, K, extrinsic)):
            return GeomVerdict("unknown", "no scene depth in observation", 0, 0, 0), None
        goal = parse_place_subgoal(subgoals[sg_idx])
        if goal is None:
            return GeomVerdict("unknown", "subgoal is not a place sentence", 0, 0, 0), None

        image = np.asarray(image, dtype=np.uint8)
        depth = np.asarray(depth, dtype=np.float32)
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        base_from_camera = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
        try:
            object_masks = [
                m for m, _ in self._geom_client.detect_and_segment_all(
                    image, goal.object_phrase,
                )
            ]
            destination_mask = self._destination_mask(image, goal.destination_phrase)
            verdict = verify_in_destination(
                depth, K, base_from_camera, object_masks, destination_mask,
                required=required_count(subgoals, sg_idx), every=goal.every,
            )
        except Exception as e:
            logger.warning(f"  geom verify failed ({e}); verdict unknown")
            return GeomVerdict("unknown", f"geom verify failed: {e}", 0, 0, 0), None

        stacked = (np.stack(object_masks).astype(bool) if object_masks
                   else np.zeros((0, *image.shape[:2]), dtype=bool))
        arrays = {
            "depth": depth.astype(np.float16),
            "object_masks": np.packbits(stacked, axis=-1),
            "object_masks_shape": np.array(stacked.shape),
            "K": K,
            "base_from_camera": base_from_camera,
        }
        if destination_mask is not None:
            arrays["destination_mask"] = np.packbits(destination_mask.astype(bool), axis=-1)
            arrays["destination_mask_shape"] = np.array(destination_mask.shape)
        return verdict, arrays

    def _destination_mask(self, image: np.ndarray, phrase: str) -> np.ndarray | None:
        """SAM3's best mask of *phrase*, ``None`` when it has no detection (422)."""
        try:
            return self._geom_client.detect_and_segment(image, phrase)[0]
        except requests.HTTPError as e:
            if e.response.status_code == 422:
                return None
            raise

    @staticmethod
    def _save_geom(state: Any, arrays: dict) -> None:
        """Write *arrays* to ``<episode log>/geom/<episode_step:06d>.npz``."""
        log_dir = getattr(state, "episode_log_dir", None)
        if not log_dir:
            return
        path = os.path.join(log_dir, "geom", f"{state.episode_step:06d}.npz")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, **arrays)
        except Exception as e:
            logger.warning(f"  geom verify: could not save {path}: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _store_check_result(state: Any, result: HandlerResult) -> None:
        """Store check result on state for video annotation."""
        state.vlm_check_result = {
            "done": result.status == STATUS_COMPLETE,
            "status": result.status,
            "action": result.action,
            "reason": result.reason,
        }
        state.vlm_check_infer_step = state.infer_count
