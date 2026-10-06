# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""grasp_probe: queue order, per-attempt logging, and the end-of-episode signal."""

from __future__ import annotations

import numpy as np

from vlm_orchestrator.strategies.base import SessionState
from vlm_orchestrator.strategies.grasp_probe import GraspProbeStrategy


def _probe() -> GraspProbeStrategy:
    probe = GraspProbeStrategy.__new__(GraspProbeStrategy)  # no proxy context needed
    probe._use_front_camera = True
    return probe


def _obs() -> dict:
    ids = np.zeros((10, 20), dtype=np.uint8)
    ids[0:5, 0:10] = 1    # bin: 50 px
    ids[6:7, 0:2] = 2     # block: 2 px
    ids[8:10, 15:20] = 3  # cup: 10 px
    return {
        "gt_seg/body_ids_front": ids,
        "gt_seg/obj_body_id": {"bin": 1, "block": 2, "cup": 3, "hidden": 4},
    }


def test_smallest_visible_object_first_and_invisible_last():
    probe, state = _probe(), SessionState()
    probe._decompose_and_setup(_obs(), state, "task", None)
    assert probe._queue == ["block", "cup", "bin", "hidden"]


def test_each_attempt_is_logged_and_the_episode_ends_after_the_last():
    probe, state = _probe(), SessionState()
    probe._decompose_and_setup(_obs(), state, "task", None)
    probe._queue = ["block"]

    d = probe._vlm_step({}, state, None)
    assert (d.tool, d.args) == ("grasp", {"target": "block"})
    assert not state.episode_done

    state.tool_chain_last_tool = "grasp"
    state.tool_chain_last_args = {"target": "block"}
    state.tool_chain_last_status, state.tool_chain_last_reason = "FAILED", "Failed: x"
    d = probe._vlm_step({}, state, None)
    assert (d.subgoal_action, d.tool) == ("abort", "noop")
    assert state.episode_done
    probe_logs = [e for e in state.log_entries if e["type"].startswith("grasp_probe")]
    assert [(e["type"], e["target"]) for e in probe_logs] == [
        ("grasp_probe_start", "block"), ("grasp_probe", "block"),
    ]
    assert probe_logs[-1]["status"] == "FAILED"
