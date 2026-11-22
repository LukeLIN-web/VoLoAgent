# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VLMFailureHandler."""

import time

import numpy as np
import pytest

from vlm_orchestrator.failure_handlers.base import (
    ACTION_CONTINUE,
    ACTION_GRASP,
    ACTION_NEXT,
    ACTION_REPLAN,
    HandlerResult,
    STATUS_COMPLETE,
    STATUS_FAILURE,
    STATUS_IN_PROGRESS,
)
from vlm_orchestrator.failure_handlers.vlm import (
    VLMFailureHandler,
    _ACTION_SETS,
    _build_system_prompt,
    _parse_vlm_detection,
)


# ======================================================================
# Prompt generation
# ======================================================================


class TestBuildSystemPrompt:
    def test_replan_mode(self):
        prompt = _build_system_prompt(_ACTION_SETS["replan"])
        assert '"next"' in prompt
        assert '"continue"' in prompt
        assert '"replan"' in prompt
        assert '"grasp_tool"' not in prompt

    def test_replan_grasp_mode(self):
        prompt = _build_system_prompt(_ACTION_SETS["replan_grasp"])
        assert '"next"' in prompt
        assert '"continue"' in prompt
        assert '"replan"' in prompt
        assert '"grasp_tool"' in prompt

    def test_grasp_mode(self):
        prompt = _build_system_prompt(_ACTION_SETS["grasp"])
        assert '"next"' in prompt
        assert '"continue"' in prompt
        assert '"replan"' not in prompt
        assert '"grasp_tool"' in prompt

    def test_stack_note_gated_on_stack_mode(self):
        """The place_stack guidance is advertised only when stack mode is
        enabled AND place_tool is an available action — never a no-op arg."""
        # replan_tools exposes place_tool.
        on = _build_system_prompt(
            _ACTION_SETS["replan_tools"], stack_mode_enabled=True,
        )
        off = _build_system_prompt(
            _ACTION_SETS["replan_tools"], stack_mode_enabled=False,
        )
        assert '"place_tool"' in on and '"place_tool"' in off
        assert "place_stack" in on
        assert "place_stack" not in off
        # No place_tool action → no stack note even with the switch on.
        no_place = _build_system_prompt(
            _ACTION_SETS["replan"], stack_mode_enabled=True,
        )
        assert "place_stack" not in no_place


# ======================================================================
# Parsing
# ======================================================================


class TestParseVLMDetection:
    def test_valid_complete(self):
        raw = '{"status": "complete", "action": "next", "reason": "done"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_COMPLETE
        assert r.action == ACTION_NEXT
        assert r.reason == "done"

    def test_valid_failure_replan(self):
        raw = '{"status": "failure", "action": "replan", "reason": "dropped"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_FAILURE
        assert r.action == ACTION_REPLAN

    def test_valid_in_progress(self):
        raw = '{"status": "in_progress", "action": "continue", "reason": "ok"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_IN_PROGRESS
        assert r.action == ACTION_CONTINUE

    def test_valid_grasp_tool(self):
        raw = '{"status": "failure", "action": "grasp_tool", "grasp_target": "red block", "reason": "wrong object"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan_grasp"])
        assert r.action == ACTION_GRASP
        assert r.grasp_target == "red block"

    def test_place_stack_parsed(self):
        raw = ('{"status": "failure", "action": "place_tool", '
               '"place_destination": "on the red block", '
               '"place_stack": true, "reason": "stack on top"}')
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan_tools"])
        assert r.place_destination == "on the red block"
        assert r.place_stack is True

    def test_place_stack_defaults_false(self):
        raw = ('{"status": "failure", "action": "place_tool", '
               '"place_destination": "in the white bowl"}')
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan_tools"])
        assert r.place_stack is False

    def test_invalid_json_defaults_to_continue(self):
        raw = "not json at all"
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_IN_PROGRESS
        assert r.action == ACTION_CONTINUE
        assert r.reason == "parse_failure"

    def test_invalid_status_defaults(self):
        raw = '{"status": "exploded", "action": "next"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_IN_PROGRESS

    def test_unavailable_action_defaults_to_continue(self):
        # grasp_tool not available in "replan" mode
        raw = '{"status": "failure", "action": "grasp_tool"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.action == ACTION_CONTINUE

    def test_markdown_code_fence(self):
        raw = '```json\n{"status": "complete", "action": "next", "reason": "moved"}\n```'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_COMPLETE
        assert r.action == ACTION_NEXT

    def test_point_tool_action_keeps_its_point(self):
        raw = ('{"status": "failure", "action": "grasp_tool", "reason": "wrong", '
               '"grasp_target": "red block", "grasp_point": [0.4, 0.6]}')
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan_tools"], frozenset({"grasp"}))
        assert r.action == ACTION_GRASP
        assert r.grasp_point == (0.4, 0.6)

    def test_point_tool_action_without_point_continues(self):
        raw = ('{"status": "failure", "action": "place_tool", "reason": "wrong", '
               '"place_destination": "in the bowl"}')
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan_tools"], frozenset({"place"}))
        assert r.action == ACTION_CONTINUE
        assert r.place_point is None

    def test_prompt_asks_for_points_only_for_point_tools(self):
        grasp_only = _build_system_prompt(_ACTION_SETS["replan_tools"], point_tools=frozenset({"grasp"}))
        assert '"grasp_point"' in grasp_only and '"place_point"' not in grasp_only
        assert '"grasp_point"' not in _build_system_prompt(_ACTION_SETS["replan_tools"])


# ======================================================================
# HandlerResult validation
# ======================================================================


class TestHandlerResult:
    def test_valid_result(self):
        r = HandlerResult(status="complete", action="next")
        assert r.status == STATUS_COMPLETE

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="Invalid status"):
            HandlerResult(status="exploded", action="next")

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="Invalid action"):
            HandlerResult(status="complete", action="dance")


# ======================================================================
# VLMFailureHandler lifecycle
# ======================================================================


class _FakeState:
    """Minimal state stub for handler tests."""
    def __init__(self):
        self.subgoals = ["pick up red block", "place in bin"]
        self.current_subgoal_idx = 0
        self.original_instruction = "put the red block in the bin"
        self.initial_image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.initial_extra_images = None
        self.infer_count = 10
        self.episode_id = 1
        self.episode_step = 0  # bumped manually in tests to drive cadence
        self.vlm_check_result = None
        self.vlm_check_infer_step = None
        self.log_entries = []
        self.metric_entries = []

    def tick(self):
        """Advance both counters by one chunk worth of step (=1 in tests)."""
        self.infer_count += 1
        self.episode_step += 1

    def log(self, entry):
        """Match SessionState.log signature so the handler can record events."""
        entry.setdefault("timestamp", time.time())
        entry.setdefault("episode_id", self.episode_id)
        self.log_entries.append(entry)


class TestVLMFailureHandler:
    def _make_handler(self, vlm_response, check_interval=5, **kwargs):
        """Create handler with a mock VLM that returns fixed response, or
        the next of a list of responses."""
        def mock_vlm_call(system_prompt, user_content):
            if isinstance(vlm_response, list):
                return vlm_response.pop(0)
            return vlm_response

        def mock_build_check(text, init_img, cur_img, extra, **kw):
            return [{"type": "text", "text": text}]

        def mock_get_image(obs):
            return np.zeros((64, 64, 3), dtype=np.uint8)

        def mock_get_extra(obs):
            return None

        return VLMFailureHandler(
            vlm_call_fn=mock_vlm_call,
            image_builder_fn=mock_build_check,
            get_vlm_image_fn=mock_get_image,
            get_extra_images_fn=mock_get_extra,
            check_interval=check_interval,
            recovery_mode="replan",
            **kwargs,
        )

    NEXT = '{"status": "complete", "action": "next", "reason": "done"}'
    CONTINUE = '{"status": "in_progress", "action": "continue", "reason": "moving"}'
    REPLAN = '{"status": "failure", "action": "replan", "reason": "dropped"}'

    def _run(self, handler, state, steps):
        """step() once per episode step: {step: action} of the replies passed on."""
        passed = {}
        for _ in range(steps):
            state.tick()
            result = handler.step({}, state)
            if result is not None:
                passed[state.episode_step] = result.action
                if result.action != ACTION_CONTINUE:
                    handler.on_subgoal_advanced({}, state, 0)
        return passed

    def _replies(self, state):
        return [
            (e["outcome"], e["call_step"], e["episode_step"])
            for e in state.log_entries if e["type"] == "vlm_reply"
        ]

    def test_next_needs_the_configured_number_of_next_replies_in_a_row(self):
        handler = self._make_handler(
            [self.NEXT, self.CONTINUE, self.NEXT, self.NEXT, self.NEXT],
            next_confirmations=2,
        )
        state = _FakeState()
        handler.on_episode_start({}, state)
        # checks at 5 (next), 10 (continue), 15 (next), 20 (next -> passed on)
        assert self._run(handler, state, 24) == {20: ACTION_NEXT}
        assert self._replies(state) == [("unconfirmed", 5, 5), ("unconfirmed", 15, 15)]
        # the streak starts over after the subgoal advanced
        assert self._run(handler, state, 5) == {}
        assert self._replies(state)[-1] == ("unconfirmed", 25, 25)

    def test_default_logs_no_reply_events(self):
        handler = self._make_handler([self.CONTINUE, self.NEXT])
        state = _FakeState()
        handler.on_episode_start({}, state)
        assert self._run(handler, state, 10) == {10: ACTION_NEXT}
        assert self._replies(state) == []

    def test_reply_takes_effect_on_its_due_step_while_later_checks_go_out(self):
        latency = [12, 3]  # the second check's reply lands first
        handler = self._make_handler(
            [self.CONTINUE, self.REPLAN, self.NEXT, self.NEXT],
            reply_due_fn=lambda: state.episode_step + latency.pop(0),
        )
        state = _FakeState()
        handler.on_episode_start({}, state)
        # checks at 5 (due 17) and 10 (due 13): the replan lands at 13 and
        # the reply still in flight, judged on an older frame, is dropped
        assert self._run(handler, state, 17) == {13: ACTION_REPLAN}
        assert self._replies(state) == [("applied", 10, 13), ("dropped", 5, 13)]

    def test_overlapping_next_replies_confirm_each_other(self):
        handler = self._make_handler(
            [self.NEXT, self.NEXT, self.NEXT],
            next_confirmations=2,
            reply_due_fn=lambda: state.episode_step + 12,
        )
        state = _FakeState()
        handler.on_episode_start({}, state)
        # checks at 5, 10, 15 land at 17, 22, 27: advance at 22, third dropped
        assert self._run(handler, state, 27) == {22: ACTION_NEXT}
        assert self._replies(state) == [
            ("unconfirmed", 5, 17), ("applied", 10, 22), ("dropped", 15, 22),
        ]

    def test_returns_none_before_interval(self):
        handler = self._make_handler('{"status": "complete", "action": "next"}')
        state = _FakeState()
        handler.on_episode_start({}, state)

        # Steps 1-4: should return None (not time yet)
        for _ in range(4):
            state.tick()
            assert handler.step({}, state) is None

    def test_returns_result_at_interval(self):
        handler = self._make_handler(
            '{"status": "complete", "action": "next", "reason": "block moved"}'
        )
        state = _FakeState()
        handler.on_episode_start({}, state)

        # Steps 1-4: not time yet
        for _ in range(4):
            state.tick()
            handler.step({}, state)

        # Step 5: should check and return result
        state.tick()
        result = handler.step({}, state)
        assert result is not None
        assert result.status == STATUS_COMPLETE
        assert result.action == ACTION_NEXT

    def test_in_progress_continue_returns_none(self):
        """in_progress + continue = nothing to report."""
        handler = self._make_handler(
            '{"status": "in_progress", "action": "continue", "reason": "ok"}'
        )
        state = _FakeState()
        handler.on_episode_start({}, state)

        for _ in range(5):
            state.tick()
            result = handler.step({}, state)
        # in_progress + continue is filtered to None
        assert result is None
        # But check result is stored on state
        assert state.vlm_check_result is not None
        assert state.vlm_check_result["status"] == STATUS_IN_PROGRESS

    def test_failure_replan_returns_result(self):
        handler = self._make_handler(
            '{"status": "failure", "action": "replan", "reason": "stuck"}'
        )
        state = _FakeState()
        handler.on_episode_start({}, state)

        for _ in range(5):
            state.tick()
            result = handler.step({}, state)
        assert result is not None
        assert result.status == STATUS_FAILURE
        assert result.action == ACTION_REPLAN

    def test_on_subgoal_advanced_resets_counter(self):
        handler = self._make_handler(
            '{"status": "complete", "action": "next"}'
        )
        state = _FakeState()
        handler.on_episode_start({}, state)

        # 3 steps in
        for _ in range(3):
            state.tick()
            handler.step({}, state)

        # Advance resets counter
        handler.on_subgoal_advanced({}, state, 1)

        # Now need 5 more steps to trigger
        for i in range(4):
            state.tick()
            assert handler.step({}, state) is None
        state.tick()
        result = handler.step({}, state)
        assert result is not None

    def test_invalid_recovery_mode(self):
        with pytest.raises(ValueError, match="Invalid recovery_mode"):
            VLMFailureHandler(
                vlm_call_fn=lambda s, u: "",
                image_builder_fn=lambda *a, **k: [],
                get_vlm_image_fn=lambda o: None,
                get_extra_images_fn=lambda o: None,
                recovery_mode="nonexistent",
            )

    def test_no_subgoals_returns_none(self):
        handler = self._make_handler('{"status": "complete", "action": "next"}')
        state = _FakeState()
        state.subgoals = []
        handler.on_episode_start({}, state)

        for _ in range(10):
            assert handler.step({}, state) is None
