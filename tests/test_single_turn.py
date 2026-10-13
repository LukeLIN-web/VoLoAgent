# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tool_chain_single_turn: one plan call, calls popped in order whatever the outcome, the prompt."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vlm_orchestrator.strategies.base import SessionState
from vlm_orchestrator.strategies.single_turn import (
    SingleTurnToolChainStrategy,
    parse_single_turn_plan,
)
from vlm_orchestrator.strategies.tool_chain import TOOL_CHAIN_SYSTEM_PROMPT

PLAN = json.dumps({"calls": [
    {"tool": "grasp", "args": {"target": "red block", "point": [0.25, 0.5]}},
    {"tool": "place", "args": {"destination": "in the blue bowl", "held_object_hint": "red block",
                               "point": [0.75, 0.5]}},
], "reason": "block into bowl"})


def _strategy(raws: list[str]) -> SingleTurnToolChainStrategy:
    s = SingleTurnToolChainStrategy.__new__(SingleTurnToolChainStrategy)   # no proxy context needed
    s._use_front_camera, s._stack_mode_enabled, s._env_mode = True, True, "robolab"
    s._point_call, s._master_points = False, True
    s._grasp_seg_mode, s._place_seg_mode = "gt_sim", "gt_sim"
    s.calls = []

    def vlm_call(system_prompt, user_content, max_tokens=None):
        s.calls.append((system_prompt, user_content, max_tokens))
        return raws.pop(0)
    s._vlm_call = vlm_call
    return s


def _obs() -> dict:
    ids = np.zeros((10, 20), dtype=np.uint8)
    ids[:, :10], ids[:, 10:] = 1, 2
    return {"gt_seg/body_ids_front": ids, "gt_seg/obj_body_id": {"block": 1, "bowl": 2}}


def test_plan_calls_parse_as_tool_chain_calls():
    plan = parse_single_turn_plan(PLAN, frozenset({"grasp", "place"}))
    assert [(d.subgoal_action, d.tool) for d in plan] == [("continue", "grasp"), ("continue", "place")]
    assert plan[0].args["point"] == (0.25, 0.5)


@pytest.mark.parametrize("raw", [
    '{"calls": [{"tool": "noop", "args": {}}]}',
    '{"plan": []}',
    '{"calls": [{"tool": "grasp", "args": {"target": "red block"}}]}',   # no point
    "not json",
])
def test_malformed_plans_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_single_turn_plan(raw, frozenset({"grasp", "place"}))


def test_one_plan_call_then_the_calls_in_order_whatever_the_outcome():
    s, state = _strategy(["oops", PLAN]), SessionState()
    state.original_instruction, state.subgoals = "put the block in the bowl", ["block into bowl"]
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    assert s._vlm_step(_obs(), state, image) is None            # malformed: asked again
    first = s._vlm_step(_obs(), state, image)
    assert (first.tool, first.args["target"]) == ("grasp", "red block")
    state.tool_chain_last_tool, state.tool_chain_last_status = "grasp", "FAILED"
    assert s._vlm_step(_obs(), state, image).tool == "place"
    assert s._vlm_step(_obs(), state, image) is None and state.tool_chain_task_done
    assert len(s.calls) == 2

    plans = [e for e in state.log_entries if e["type"] == "single_turn_plan"]
    assert len(plans) == 1 and plans[0]["point_objects"] == ["block", "bowl"]
    assert [e["type"] for e in state.log_entries if e["type"].startswith("single_turn")] == [
        "single_turn_plan_parse_error", "single_turn_plan", "single_turn_plan_done",
    ]


def test_prompt_keeps_the_tool_chain_tool_text_and_one_image():
    s, state = _strategy([PLAN]), SessionState()
    state.original_instruction, state.subgoals = "task", ["a", "b"]
    s._vlm_step(_obs(), state, np.zeros((4, 4, 3), dtype=np.uint8))
    system_prompt, user_content, max_tokens = s.calls[0]
    tools = TOOL_CHAIN_SYSTEM_PROMPT[TOOL_CHAIN_SYSTEM_PROMPT.index("  grasp(target: str)"):
                                     TOOL_CHAIN_SYSTEM_PROMPT.index("  noop\n")].rstrip()
    assert tools in system_prompt
    assert "NOW" not in system_prompt and "subgoal_action" not in system_prompt
    assert "STACK CONTROL" in system_prompt and "POINTS:" in system_prompt
    assert [c["type"] for c in user_content] == ["text", "image_url"]
    assert max_tokens > 512


def test_point_call_is_refused():
    with pytest.raises(ValueError):
        SingleTurnToolChainStrategy(None, None, point_call=True)




def test_points_go_to_the_tools_as_written_and_a_new_episode_plans_again():
    s, state = _strategy([PLAN, PLAN]), SessionState()
    state.original_instruction, state.subgoals = "task", ["a"]
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    moved = {"gt_seg/body_ids_front": np.full((10, 20), 2, dtype=np.uint8),   # the block left; all bowl now
             "gt_seg/obj_body_id": {"block": 1, "bowl": 2}}
    assert s._vlm_step(_obs(), state, image).args["point"] == (0.25, 0.5)
    assert s._vlm_step(moved, state, image).args["point"] == (0.75, 0.5)
    state.single_turn_plan = None   # what _on_new_episode sets
    assert s._vlm_step(_obs(), state, image).tool == "grasp" and len(s.calls) == 2

def test_libero_is_refused():
    with pytest.raises(ValueError):
        SingleTurnToolChainStrategy(None, None, env_mode="libero")
