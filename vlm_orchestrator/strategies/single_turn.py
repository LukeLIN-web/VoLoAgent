# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tool_chain_single_turn`` mode: the tool-chain loop without feedback.

The master writes every grasp / place call once, on the scene before anything
has run, and never sees a result: no NOW image after a tool, no tool status,
no replan. The calls then run in order; a FAILED call is followed by the next
one as written. The subgoal decomposition at episode start is the same call as
in ``tool_chain``. Tools and their grounding sources are unchanged: a call's
point is read by its tool on the frame the tool perceives, as in ``tool_chain``,
so it goes stale for an object an earlier call moved.

``rewrites.jsonl`` gets one ``single_turn_plan`` entry with the calls and, on
RoboLab, the scene object each point names on the image the master saw
(``point_objects``). The ``point_object`` of a ``tool_chain_step`` entry is read
on the frame at execution time.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time

from vlm_orchestrator.perception.gt_segmentation import POINT_SNAP_PX, gt_ids_available, gt_object_at
from vlm_orchestrator.vlm import encode_image_b64, parse_json

from .tool_chain import (
    TOOL_CHAIN_SYSTEM_PROMPT,
    _TOOL_CHAIN_FRONT_CAM_NOTE,
    _TOOL_CHAIN_STACK_NOTE,
    ToolChainDecision,
    ToolChainStrategy,
    _tool_chain_point_note,
    parse_tool_chain_response,
)

logger = logging.getLogger(__name__)

# The grasp / place definitions and phrasing rules of the tool_chain prompt, verbatim.
_TOOLS_START, _TOOLS_END = "  grasp(target: str)\n", "  noop\n"
assert TOOL_CHAIN_SYSTEM_PROMPT.count(_TOOLS_START) == 1 and TOOL_CHAIN_SYSTEM_PROMPT.count(_TOOLS_END) == 1
_TOOLS = TOOL_CHAIN_SYSTEM_PROMPT[
    TOOL_CHAIN_SYSTEM_PROMPT.index(_TOOLS_START):TOOL_CHAIN_SYSTEM_PROMPT.index(_TOOLS_END)
].rstrip() + "\n"

SINGLE_TURN_SYSTEM_PROMPT = """\
You are driving a robot arm.  Write the whole plan at once: every
tool call needed to finish the task, in the order they should run.
The calls run one after another exactly as written.  You will not
see any result, and the plan cannot be changed once it is written.

Inputs you receive:
  - the overall task instruction
  - the planned subgoal list
  - an image of the scene before anything has run

TOOLS:

""" + _TOOLS + """
OUTPUT: a single JSON object, exactly these two top-level keys, no
markdown, no commentary outside the JSON:
  {
    "calls":  [ {"tool": "grasp" | "place", "args": { <tool-specific> }}, ... ],
    "reason": "<one short sentence describing the plan>"
  }

Example:
  {"calls": [
     {"tool": "grasp", "args": {"target": "red block"}},
     {"tool": "place", "args": {"destination": "in the blue bowl",
                                "held_object_hint": "red block"}}],
   "reason": "Put the red block into the blue bowl."}
"""

# A plan holds every call of the episode, not one.
PLAN_MAX_TOKENS = 2048


def parse_single_turn_plan(raw: str, point_tools: frozenset[str] = frozenset()) -> list[ToolChainDecision]:
    """The plan's calls as ``continue`` decisions, each validated as a tool_chain call."""
    try:
        data = parse_json(raw)
    except Exception as exc:
        raise ValueError(f"could not parse the plan as JSON: {exc!s}; raw[:200]={raw[:200]!r}") from exc
    calls = data.get("calls") if isinstance(data, dict) else None
    if not isinstance(calls, list):
        raise ValueError(f"the plan needs a `calls` list; got {type(calls).__name__}")
    reason = str(data.get("reason", "")).strip()
    plan = []
    for i, call in enumerate(calls):
        if not isinstance(call, dict) or call.get("tool") not in ("grasp", "place"):
            raise ValueError(f"calls[{i}] must be a grasp or place call; got {call!r}")
        decision = parse_tool_chain_response(json.dumps({
            "subgoal_action": "continue", "tool": call["tool"],
            "args": call.get("args") or {}, "reason": reason,
        }), point_tools)
        decision.raw_response = raw
        plan.append(decision)
    return plan


class SingleTurnToolChainStrategy(ToolChainStrategy):
    """Tool-chain loop whose calls come from one plan the master writes before anything runs."""

    def __init__(self, ctx, config, **kwargs):
        if kwargs.get("point_call"):
            raise ValueError(
                "tool_chain_single_turn: --master-point-call asks the master at every tool; "
                "the points come with the plan (--master-points)"
            )
        # The per-subgoal cap forces a replan, which asks the master on a new image.
        super().__init__(ctx, dataclasses.replace(config, max_tools_per_subgoal=10**9), **kwargs)
        self._plan: list[ToolChainDecision] | None = None

    def _on_new_episode(self, obs, state, prompt):
        obs, state = super()._on_new_episode(obs, state, prompt)
        self._plan = None
        return obs, state

    def _vlm_step(self, obs, state, now_image):
        if self._plan is None:
            self._plan = self._write_plan(obs, state, now_image)
            if self._plan is None:
                return None   # logged; asked again next cycle, nothing has run yet
        if not self._plan:
            state.tool_chain_task_done = True
            state.log({"type": "single_turn_plan_done"})
            return None
        return self._plan.pop(0)

    def _system_prompt(self) -> str:
        prompt = SINGLE_TURN_SYSTEM_PROMPT
        if self._use_front_camera:
            prompt += _TOOL_CHAIN_FRONT_CAM_NOTE
        if self._stack_mode_enabled:
            prompt += _TOOL_CHAIN_STACK_NOTE
        if self._point_tools:
            note = _tool_chain_point_note(self._point_tools)
            assert note.count(", in the NOW image.") == 1
            prompt += note.replace(", in the NOW image.", ", in the image.")
        return prompt

    def _write_plan(self, obs, state, image) -> list[ToolChainDecision] | None:
        lines = [f"Task: {state.original_instruction or ''}", "", "Subgoals:"]
        lines += [f"  [{i}] {sg}" for i, sg in enumerate(state.subgoals)] or ["  (no subgoals)"]
        lines += ["", "Return JSON only. No markdown. No commentary."]
        user_content = [
            {"type": "text", "text": "\n".join(lines)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_b64(image)}"}},
        ]
        t0 = time.time()
        try:
            raw = self._vlm_call(self._system_prompt(), user_content, max_tokens=PLAN_MAX_TOKENS)
        except Exception as e:
            logger.error(f"tool_chain_single_turn: plan call failed: {e}; will retry next cycle")
            state.log({"type": "single_turn_plan_vlm_error", "error": repr(e)})
            return None
        elapsed = round(time.time() - t0, 2)
        try:
            plan = parse_single_turn_plan(raw, self._point_tools)
        except ValueError as e:
            logger.warning(f"tool_chain_single_turn: malformed plan ({e}); will retry next cycle")
            state.log({"type": "single_turn_plan_parse_error", "error": str(e), "raw": raw[:500],
                       "vlm_latency_s": elapsed})
            return None
        entry = {
            "type": "single_turn_plan",
            "calls": [{"tool": d.tool, "args": d.args} for d in plan],
            "raw": raw[:2000],
            "vlm_latency_s": elapsed,
        }
        if self._env_mode == "robolab" and gt_ids_available(obs, front=self._use_front_camera):
            entry["point_objects"] = [
                gt_object_at(obs, d.args["point"], front=self._use_front_camera, radius_px=POINT_SNAP_PX)
                if d.args.get("point") is not None else None
                for d in plan
            ]
        state.log(entry)
        logger.info(f"tool_chain_single_turn: plan of {len(plan)} calls ({elapsed}s)")
        return plan
