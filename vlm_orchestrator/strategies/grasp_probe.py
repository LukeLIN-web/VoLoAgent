# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``grasp_probe`` mode: measure the grasp tool on its own.

Grasps every scene object once per episode, with no VLM and no VLA, smallest
visible object first: the objects tasks move are mostly small, and containers
or shelves would otherwise use up the episode; objects the grasp camera cannot
see go last. Each attempt starts from wherever the previous one left the scene
(the grasp tool releases first). Targets are scene-object names, so the grasp tool
runs with ``--grasp-seg-mode gt_sim`` and its held-and-lifted verdict decides
DONE / FAILED. Every attempt is logged to ``rewrites.jsonl`` as a
``grasp_probe_start`` entry and, once the tool finishes, a ``grasp_probe``
entry with the tool status and reason; an episode that ends mid-grasp leaves
the start entry alone. Once every object is probed it asks the eval client to
end the episode (``SessionState.episode_done``).
"""

from __future__ import annotations

import dataclasses
import logging

from vlm_orchestrator.perception.gt_segmentation import gt_object_mask

from .tool_chain import ToolChainConfig, ToolChainDecision, ToolChainStrategy

logger = logging.getLogger(__name__)


class GraspProbeStrategy(ToolChainStrategy):
    """Tool-chain loop whose decisions come from a fixed queue, not a VLM."""

    def __init__(self, ctx, config: ToolChainConfig, **kwargs):
        # Every scene object is one tool call; the tool-chain caps would
        # otherwise force a (VLM) replan after a few grasps.
        unbounded = dataclasses.replace(
            config, max_tools_per_subgoal=10**9, max_tools_per_episode=10**9,
        )
        super().__init__(ctx, unbounded, **kwargs)
        self._queue: list[str] = []
        self._pending: str | None = None

    def _decompose_and_setup(self, obs, state, instruction, image):
        state.subgoals = ["grasp every scene object"]
        state.current_subgoal_idx = 0
        state.rewritten_instruction = instruction
        size = {
            name: int(gt_object_mask(obs, name, front=self._use_front_camera).sum())
            for name in obs["gt_seg/obj_body_id"]
        }
        self._queue = sorted(size, key=lambda n: (size[n] == 0, size[n]))
        self._pending = None
        logger.info(f"grasp_probe: {len(self._queue)} objects: {self._queue}")

    def _vlm_step(self, obs, state, now_image):
        if self._pending is not None:
            finished = (
                state.tool_chain_last_tool == "grasp"
                and state.tool_chain_last_args.get("target") == self._pending
            )
            state.log({
                "type": "grasp_probe",
                "target": self._pending,
                "status": state.tool_chain_last_status if finished else "NOT_STARTED",
                "reason": state.tool_chain_last_reason if finished else "",
            })
            state.tool_chain_last_tool = None
            self._pending = None
        if not self._queue:
            state.episode_done = True
            return ToolChainDecision(
                subgoal_action="abort", tool="noop",
                reason="probed every scene object",
            )
        self._pending = self._queue.pop(0)
        state.log({"type": "grasp_probe_start", "target": self._pending})
        return ToolChainDecision(
            subgoal_action="continue", tool="grasp",
            args={"target": self._pending}, reason="grasp probe",
        )
