# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for the grasp prediction server.

Thin wrapper around ``requests`` that talks to :mod:`grasp_server`.
No heavy ML dependencies — safe to import in the orchestrator proxy
process.

Usage::

    client = GraspClient("http://localhost:8003")
    assert client.health()["status"] == "ok"

    grasp_pose, confidence = client.compute_grasp(
        point_cloud=pts,       # (N, 3) float32
        mask=mask,             # (H, W) bool
        image_hw=(720, 1280),
        focal_length_px=500.0,
    )

    # Optional (if server was started with --enable-sam2):
    mask, iou = client.segment(image_rgb, point_x=0.5, point_y=0.3)
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 180.0  # seconds — GraspGen inference can take a while (first call includes JIT warmup)


def _default_grasp_url() -> str:
    """Build grasp server URL from env vars or fall back to localhost."""
    host = os.environ.get("GRASP_SERVER_HOST", "localhost")
    port = os.environ.get("GRASP_SERVER_PORT", "8003")
    return f"http://{host}:{port}"


def _default_perception_url(grasp_url: str) -> str:
    """The server answering /detect, /segment and /detect_and_segment: the
    grasp server itself unless PERCEPTION_SERVER_URL points at a separate
    perception-only instance (server.py without --gripper-config)."""
    return os.environ.get("PERCEPTION_SERVER_URL", grasp_url)


def _default_planner_url(grasp_url: str) -> str:
    """The server answering /plan_motion, /attach_object and /detach_object:
    the grasp server itself unless PLANNER_SERVER_URL points at a separate
    cuRobo-only instance (server.py --enable-curobo without --gripper-config;
    cuRobo 0.8 needs a newer torch than GraspGen's)."""
    return os.environ.get("PLANNER_SERVER_URL", grasp_url)


class GraspClient:
    """HTTP client for the grasp prediction server."""

    def __init__(
        self,
        url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self._url = (url or _default_grasp_url()).rstrip("/")
        self._perception_url = _default_perception_url(self._url).rstrip("/")
        self._planner_url = _default_planner_url(self._url).rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_npz(**arrays) -> bytes:
        """Serialize numpy arrays into an in-memory ``.npz`` blob."""
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _unpack_npz(content: bytes) -> dict[str, np.ndarray]:
        """Deserialize an ``.npz`` response body."""
        return dict(np.load(io.BytesIO(content), allow_pickle=True))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Check server health.  Returns ``{"status": "ok", "models": [...]}``."""
        resp = requests.get(
            f"{self._url}/health", timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def planner_health(self) -> dict:
        """Health of the server answering /plan_motion (see PLANNER_SERVER_URL)."""
        resp = requests.get(
            f"{self._planner_url}/health", timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def compute_grasp(self, *args, **kwargs) -> tuple[np.ndarray, float]:
        """The best grasp of :meth:`compute_grasps`: ``(grasp_pose (4, 4), confidence)``."""
        poses, confidences = self.compute_grasps(*args, **kwargs)
        return poses[0], float(confidences[0])

    def compute_grasps(
        self,
        point_cloud: np.ndarray,
        mask: np.ndarray,
        image_hw: tuple[int, int],
        focal_length_px: float,
        *,
        fy_px: Optional[float] = None,
        cx_px: Optional[float] = None,
        cy_px: Optional[float] = None,
        topdown_gravity: Optional[np.ndarray] = None,
        topdown_threshold: Optional[float] = None,
        enable_topdown_filter: Optional[bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict grasp poses for the masked object, best first.

        Args:
            point_cloud: (N, 3) float32 scene point cloud in camera frame.
            mask: (H, W) boolean mask indicating object pixels.
            image_hw: (height, width) of the image the mask corresponds to.
            focal_length_px: Camera focal length in pixels (fx).
            fy_px: Vertical focal length in pixels.  Defaults to
                ``focal_length_px`` if not provided.
            cx_px: Principal point X in pixels.  Defaults to ``w / 2``.
            cy_px: Principal point Y in pixels.  Defaults to ``h / 2``.
            topdown_gravity: Optional (3,) gravity vector for top-down
                filtering (server has a default if omitted).
            topdown_threshold: Optional dot-product cutoff in ``[0, 1]``;
                Contact-GraspNet candidates whose approach axis is
                less aligned with ``topdown_gravity`` than this are
                rejected.  ``0`` keeps everything; ``1`` accepts only
                perfectly vertical approaches.  Server default 0.3.
                Pass ~0.85 for a "near-top-down only" run.
            enable_topdown_filter: Optional bool.  When ``False`` the server
                skips the top-down filter entirely and returns GraspGen's best
                grasp at any approach angle (used for stack=False placements,
                where the object is dropped from above and grasp orientation
                does not matter).  ``None`` leaves the server default (on).

        Returns:
            Tuple of ``(grasp_poses, confidences)``: (K, 4, 4) homogeneous
            transforms in camera frame and their (K,) confidences in [0, 1],
            sorted by confidence, highest first.

        Raises:
            requests.HTTPError: on server-side failure (e.g. no valid grasp).
        """
        arrays: dict = {
            "point_cloud": np.asarray(point_cloud, dtype=np.float32),
            "mask": np.asarray(mask, dtype=np.uint8),
            "image_hw": np.array(image_hw, dtype=np.int32),
            "focal_length_px": np.array(focal_length_px, dtype=np.float64),
        }
        if fy_px is not None:
            arrays["fy_px"] = np.array(fy_px, dtype=np.float64)
        if cx_px is not None:
            arrays["cx_px"] = np.array(cx_px, dtype=np.float64)
        if cy_px is not None:
            arrays["cy_px"] = np.array(cy_px, dtype=np.float64)
        if topdown_gravity is not None:
            arrays["topdown_gravity"] = np.asarray(
                topdown_gravity, dtype=np.float32,
            )
        if topdown_threshold is not None:
            arrays["topdown_threshold"] = np.array(
                topdown_threshold, dtype=np.float64,
            )
        if enable_topdown_filter is not None:
            arrays["enable_topdown_filter"] = np.array(
                bool(enable_topdown_filter)
            )

        body = self._pack_npz(**arrays)

        resp = requests.post(
            f"{self._url}/compute_grasp",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout,
        )
        if not resp.ok:
            # Include the server's error detail so we can diagnose
            try:
                detail = resp.json().get("detail", resp.text[:500])
            except Exception:
                detail = resp.text[:500]
            raise RuntimeError(
                f"Grasp server returned {resp.status_code}: {detail}"
            )

        data = self._unpack_npz(resp.content)
        return data["grasp_poses"], data["grasp_confidences"]

    def plan_motion(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        *,
        n_steps: Optional[int] = None,
        scene_pc: Optional[np.ndarray] = None,
        disable_fingers: bool = False,
    ) -> tuple[Optional[np.ndarray], bool]:
        """Request a collision-free joint trajectory from the cuRobo endpoint.

        Args:
            q_start: (7,) start arm joint config.
            q_end: (7,) goal arm joint config (from caller's IK).
            n_steps: Optional target waypoint count (server resamples).
            scene_pc: Optional (N, 3) base-frame scene point cloud obstacle.
            disable_fingers: If True, the server disables the gripper
                finger/hand collision links for this plan so a top-down
                pre-grasp near the table isn't rejected (fingers reach ~10 cm
                below panda_hand).  Matches cuRobo's own plan_grasp approach.

        Returns:
            ``(waypoints, True)`` with ``waypoints`` (M, 7), or ``(None, False)``
            if the planner failed / is not loaded.  Never raises on planning
            failure — the caller surfaces it via a named planner label.
        """
        arrays: dict = {
            "q_start": np.asarray(q_start, dtype=np.float32),
            "q_end": np.asarray(q_end, dtype=np.float32),
        }
        if n_steps is not None:
            arrays["n_steps"] = np.array(int(n_steps), dtype=np.int32)
        if scene_pc is not None and len(scene_pc) > 0:
            arrays["scene_pc"] = np.asarray(scene_pc, dtype=np.float32)
        if disable_fingers:
            arrays["disable_fingers"] = np.array(True)

        body = self._pack_npz(**arrays)
        try:
            resp = requests.post(
                f"{self._planner_url}/plan_motion",
                data=body,
                headers={"Content-Type": "application/octet-stream"},
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            logger.warning(f"plan_motion request failed: {e}")
            return None, False

        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            logger.warning(
                f"plan_motion server returned {resp.status_code}: {detail}"
            )
            return None, False

        data = self._unpack_npz(resp.content)
        if not bool(data.get("success", np.array(False))):
            return None, False
        return data["waypoints"], True

    def attach_object(
        self,
        obj_pc: np.ndarray,
        q_hold: np.ndarray,
        *,
        num_spheres: int = 4,
    ) -> bool:
        """Attach a held-object cloud to the gripper link (cuRobo).

        Used before a collision-aware PLACE approach so the carried object
        moves with the hand instead of acting as a fixed world obstacle.

        Args:
            obj_pc: (N, 3) held-object cloud in the robot base frame.
            q_hold: (7,) arm config at which the object is held.
            num_spheres: sphere-hull resolution (server caps at 4).

        Returns:
            True if attach succeeded.  Never raises on failure — returns
            False so the caller can surface a named label / decide to abort.
        """
        if obj_pc is None or len(obj_pc) == 0:
            return False
        arrays: dict = {
            "obj_pc": np.asarray(obj_pc, dtype=np.float32),
            "q_hold": np.asarray(q_hold, dtype=np.float32),
            "num_spheres": np.array(int(num_spheres), dtype=np.int32),
        }
        body = self._pack_npz(**arrays)
        try:
            resp = requests.post(
                f"{self._planner_url}/attach_object",
                data=body,
                headers={"Content-Type": "application/octet-stream"},
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            logger.warning(f"attach_object request failed: {e}")
            return False
        if not resp.ok:
            logger.warning(
                f"attach_object server returned {resp.status_code}"
            )
            return False
        data = self._unpack_npz(resp.content)
        return bool(data.get("success", np.array(False)))

    def detach_object(self) -> bool:
        """Detach the held object on the cuRobo planner.

        Safe to call even if nothing was attached.  Never raises — returns
        False on transport error.
        """
        try:
            resp = requests.post(
                f"{self._planner_url}/detach_object",
                data=self._pack_npz(),
                headers={"Content-Type": "application/octet-stream"},
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            logger.warning(f"detach_object request failed: {e}")
            return False
        if not resp.ok:
            logger.warning(
                f"detach_object server returned {resp.status_code}"
            )
            return False
        data = self._unpack_npz(resp.content)
        return bool(data.get("success", np.array(False)))

    def detect(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> list[dict]:
        """Detect objects using GroundingDINO on the grasp server.

        Args:
            image: (H, W, 3) uint8 RGB image.
            text_prompt: Object description (should end with ".").

        Returns:
            List of detections sorted by score descending. Each dict has
            ``score`` (float), ``label`` (str), ``box`` ([x1,y1,x2,y2]).
            Empty list if nothing detected.

        Raises:
            requests.HTTPError: if GDino not loaded (501) or server error.
        """
        body = self._pack_npz(
            image=np.asarray(image, dtype=np.uint8),
            text_prompt=np.array(text_prompt, dtype=object),
        )

        resp = requests.post(
            f"{self._perception_url}/detect",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout,
        )
        resp.raise_for_status()

        data = self._unpack_npz(resp.content)
        n = int(data["num_detections"])
        if n == 0:
            return []

        boxes = data["boxes"]      # (N, 4)
        scores = data["scores"]    # (N,)
        labels = data["labels"]    # (N,)

        detections = []
        for i in range(n):
            detections.append({
                "score": float(scores[i]),
                "label": str(labels[i]),
                "box": [int(boxes[i][j]) for j in range(4)],
            })
        return detections

    def detect_and_segment(
        self,
        image: np.ndarray,
        text_prompt: str,
        point: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
        """Detect and segment an object using SAM3 (single model).

        Replaces the two-step GDino detect → SAM2 segment flow with one
        SAM3 call on the grasp server.

        Args:
            image: (H, W, 3) uint8 RGB image.
            text_prompt: Description of the target object.

        Returns:
            Tuple of ``(mask, score, box)`` where *mask* is (H, W) bool,
            *score* is float, and *box* is (x1, y1, x2, y2) in pixel coords.

        With ``point`` (normalised ``(x, y)``) the detection whose mask
        covers it, among all of the phrase's, is returned.

        Raises:
            requests.HTTPError: if SAM3 is not loaded (501) or no detection
                (covers the point) (422).
        """
        arrays = dict(
            image=np.asarray(image, dtype=np.uint8),
            text_prompt=np.array(text_prompt, dtype=object),
        )
        if point is not None:
            arrays["point"] = np.asarray(point, dtype=np.float64)
        body = self._pack_npz(**arrays)

        resp = requests.post(
            f"{self._perception_url}/detect_and_segment",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout,
        )
        resp.raise_for_status()

        data = self._unpack_npz(resp.content)
        mask = data["mask"].astype(bool)
        score = float(data["score"])
        box = tuple(int(x) for x in data["box"])
        return mask, score, box

    def segment(
        self,
        image: np.ndarray,
        point_x: float,
        point_y: float,
        *,
        box: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Segment the object at a normalised 2-D point (and optional box).

        Args:
            image: (H, W, 3) uint8 RGB image.
            point_x: Normalised x-coordinate [0, 1].
            point_y: Normalised y-coordinate [0, 1].
            box: Optional (x1, y1, x2, y2) bounding box in **pixel** coords.
                 When provided SAM2 uses both the point and box for a
                 higher-quality mask.

        Returns:
            Tuple of ``(mask, iou_score)`` where *mask* is (H, W) bool and
            *iou_score* is a float in [0, 1].

        Raises:
            requests.HTTPError: if SAM2 is not loaded on the server (501).
        """
        arrays: dict = {
            "image": np.asarray(image, dtype=np.uint8),
            "point_x": np.array(point_x, dtype=np.float64),
            "point_y": np.array(point_y, dtype=np.float64),
        }
        if box is not None:
            arrays["box"] = np.array(box, dtype=np.int32)

        body = self._pack_npz(**arrays)

        resp = requests.post(
            f"{self._perception_url}/segment",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout,
        )
        resp.raise_for_status()

        data = self._unpack_npz(resp.content)
        mask = data["mask"].astype(bool)
        iou = float(data["iou_score"])
        return mask, iou
