# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VLMFailureHandler."""

import base64
import io
import json
import time

import numpy as np
import pytest
from PIL import Image

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
    def _make_handler(self, vlm_response, check_interval=5):
        """Create handler with a mock VLM that returns fixed response."""
        def mock_vlm_call(system_prompt, user_content):
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
        )

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


class TestRecoveryContext:
    @staticmethod
    def make_handler(response='{"status":"in_progress","action":"continue"}'):
        from vlm_orchestrator.strategies.base import SessionState
        from vlm_orchestrator.strategies.subgoal_base import SubgoalBaseStrategy

        calls = []

        def call(system, content):
            calls.append(content)
            return response

        handler = VLMFailureHandler(
            vlm_call_fn=call,
            image_builder_fn=SubgoalBaseStrategy._build_check_message,
            get_vlm_image_fn=lambda obs: obs["rgb"],
            get_extra_images_fn=lambda obs: [obs["wrist"]],
            check_interval=5,
            recovery_mode="replan_tools",
            primary_label="Front camera",
            extra_labels=["Wrist camera"],
        )
        state = SessionState(original_instruction="Put the red block in the bowl")
        # Production startup resets the handler BEFORE decomposition.
        handler.on_episode_start({}, state)
        state.subgoals = [state.original_instruction]
        state.rewritten_instruction = state.subgoals[0]
        obs = {
            "rgb": np.full((16, 16, 3), 17, dtype=np.uint8),
            "wrist": np.full((16, 16, 3), 42, dtype=np.uint8),
        }
        return handler, state, obs, calls

    def test_previous_rgb_is_copied_and_survives_same_subgoal_tool_callback(self):
        handler, state, obs, calls = self.make_handler()
        state.episode_step = 5
        assert handler.step(obs, state) is None
        assert not any(p.get("text", "").startswith("PREVIOUS") for p in calls[0])

        # Camera buffers may be reused; history must keep the old pixels.
        obs["rgb"][:] = 200
        obs["wrist"][:] = 210
        handler.on_subgoal_advanced(obs, state, 0)
        state.episode_step = 10
        assert handler.step(obs, state) is None
        previous = []
        for i, part in enumerate(calls[1]):
            if part.get("text", "").startswith("PREVIOUS"):
                assert "step 5" in part["text"]
                assert state.subgoals[0] in part["text"]
                url = calls[1][i + 1]["image_url"]["url"]
                pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",")[1]))))
                previous.append((part["text"], pixels))
        assert len(previous) == 2
        assert "Front camera" in previous[0][0]
        assert "Wrist camera" in previous[1][0]
        assert np.allclose(previous[0][1], 17, atol=1)
        assert np.allclose(previous[1][1], 42, atol=1)

    @pytest.mark.parametrize("extra_view", [False, True])
    def test_three_camera_monitor_respects_image_limit_and_pairing(self, extra_view):
        from vlm_orchestrator.strategies.base import SessionState, StrategyContext
        from vlm_orchestrator.strategies.subgoal_base import SubgoalBaseStrategy

        # Deployment uses Front / External / Wrist. Also exercise an already
        # full eight-image BEFORE/NOW message: history must then be omitted.
        ctx = StrategyContext(
            vlm=None, front_image_key="front", image_key="external",
            extra_image_keys=["wrist"] + (["aux"] if extra_view else []),
        )
        keys = ["front", "external", "wrist"] + (["aux"] if extra_view else [])

        def observation(offset):
            return {key: np.full((16, 16, 3), offset + i * 10, dtype=np.uint8)
                    for i, key in enumerate(keys)}

        before, previous, now = [observation(offset) for offset in (10, 60, 110)]
        state = SessionState(
            original_instruction="Put the block in the bowl",
            subgoals=["Put the block in the bowl"],
            initial_image=ctx.get_vlm_image(before),
            initial_extra_images=ctx.get_vlm_extra_images(before),
        )
        calls = []

        def call(system, content):
            calls.append(content)
            return '{"action":"continue"}'

        primary_label, extra_labels = ctx.vlm_camera_labels
        handler = VLMFailureHandler(
            vlm_call_fn=call,
            image_builder_fn=SubgoalBaseStrategy._build_check_message,
            get_vlm_image_fn=ctx.get_vlm_image,
            get_extra_images_fn=ctx.get_vlm_extra_images,
            check_interval=5, primary_label=primary_label, extra_labels=extra_labels,
        )
        handler.on_episode_start(before, state)
        state.episode_step = 5
        handler.step(previous, state)
        assert sum(p["type"] == "image_url" for p in calls[0]) == 2 * len(keys)
        state.episode_step = 10
        handler.step(now, state)

        baseline = SubgoalBaseStrategy._build_check_message(
            "", state.initial_image, ctx.get_vlm_image(now),
            ctx.get_vlm_extra_images(now), state.initial_extra_images,
            primary_label=primary_label, extra_labels=extra_labels,
        )
        # Preserve every existing BEFORE/NOW label and encoded image verbatim.
        assert calls[1][:len(baseline) - 1] == baseline[:-1]
        pairs = []
        for i, part in enumerate(calls[1]):
            if part["type"] == "image_url":
                assert calls[1][i - 1]["type"] == "text"
                pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(
                    part["image_url"]["url"].split(",")[1],
                ))))
                pairs.append((calls[1][i - 1]["text"], pixels))
        assert len(pairs) == 8
        expected = [(f"{when} — {label}", offset + i * 10)
                    for when, offset in [("BEFORE", 10), ("NOW", 110)]
                    for i, label in enumerate([primary_label] + extra_labels)]
        if not extra_view:
            expected += [("PREVIOUS — Front camera", 60), ("PREVIOUS — Wrist camera", 80)]
        for (label, pixels), (expected_label, value) in zip(pairs, expected):
            assert label.startswith(expected_label)
            assert np.allclose(pixels, value, atol=1)

    @pytest.mark.parametrize("reset", ["episode", "subgoal"])
    def test_context_does_not_cross_episode_or_subgoal(self, reset):
        handler, state, obs, calls = self.make_handler()
        state.log({"type": "grasp_tool_failed", "reason": "old motion failure"})
        state.episode_step = 5
        handler.step(obs, state)
        state.log_entries.clear()
        if reset == "episode":
            handler.on_episode_start(obs, state)
        else:
            state.subgoals = ["Put the blue block in the bowl"]
            handler.on_subgoal_advanced(obs, state, 0)
        state.episode_step = 10
        handler.step(obs, state)
        text = "\n".join(p.get("text", "") for p in calls[-1])
        assert "PREVIOUS" not in text
        assert "old motion failure" not in text

    def test_only_deployment_rgb_and_whitelisted_tool_feedback_are_used(self):
        handler, state, obs, calls = self.make_handler()

        class RGBOnlyObservation(dict):
            def __getitem__(self, key):
                assert key in ("rgb", "wrist"), f"Forbidden observation read: {key}"
                return super().__getitem__(key)

            def get(self, *args):
                raise AssertionError("Recovery must use its supplied RGB callbacks")

            def __iter__(self):
                raise AssertionError("Recovery must not enumerate raw observations")

        obs = RGBOnlyObservation(obs, gt_state=object(), success=True)
        state.metric_entries = [{"type": "evaluator_sentinel", "success": True}]
        state.log_entries = [
            {"type": "grasp_tool_done", "target_object": "red block",
             "success": True, "gt_state": "evaluator_sentinel"},
            {"type": "evaluator_sentinel", "reason": "evaluator_sentinel"},
        ]
        # Completion can occur before the next scheduled monitor.
        state.episode_step = 2
        assert handler.step(obs, state) is None
        assert calls == []
        state.log_entries.clear()
        state.episode_step = 5
        assert handler.step(obs, state) is None
        text = "\n".join(p.get("text", "") for p in calls[-1])
        assert "grasp_tool_done" in text and "red block" in text
        assert "evaluator_sentinel" not in text and "gt_state" not in text
        assert '"success"' not in text
        assert state.current_subgoal_idx == 0 and not state.episode_done

    @pytest.mark.parametrize("instruction", [None, "", "  ", 7, False, [], {}, "x" * 241])
    def test_invalid_refinement_does_not_reach_vla(self, instruction):
        handler, state, obs, _ = self.make_handler(json.dumps({
            "action": "continue", "instruction": instruction,
        }))
        state.episode_step = 5
        assert handler.step(obs, state) is None
        assert state.rewritten_instruction == state.subgoals[0]

    @pytest.mark.parametrize("action", ["unknown", "grasp_tool", [], {}])
    def test_rejected_action_cannot_smuggle_a_refinement(self, action):
        result = _parse_vlm_detection(json.dumps({
            "action": action, "instruction": "Move the red block",
        }), _ACTION_SETS["replan"])
        assert result.action == ACTION_CONTINUE
        assert result.instruction is None

    def test_refinement_renews_then_expires_before_tool_handoff(self):
        handler, state, obs, calls = self.make_handler()
        replies = iter([
            '{"action":"continue","instruction":"  Grasp the red block  "}',
            '{"action":"continue","instruction":"Lift the red block"}',
            '{"action":"place_tool","place_destination":"in the bowl"}',
        ])
        handler._vlm_call = lambda *args: next(replies)
        for step, expected in [(5, "Grasp the red block"), (10, "Lift the red block")]:
            state.episode_step = step
            result = handler.step(obs, state)
            assert result.action == ACTION_CONTINUE and result.instruction == expected
            state.rewritten_instruction = result.instruction
        state.episode_step = 15
        result = handler.step(obs, state)
        assert result.action == "place_tool"
        assert result.instruction == state.subgoals[0]
        assert result.place_destination == "in the bowl"

    def test_expiry_does_not_overwrite_an_external_instruction_change(self):
        handler, state, obs, calls = self.make_handler(
            '{"action":"continue","instruction":"Grasp the red block"}'
        )
        state.episode_step = 5
        result = handler.step(obs, state)
        state.rewritten_instruction = result.instruction
        state.rewritten_instruction = "Operator instruction"
        handler._vlm_call = lambda *args: '{"action":"continue"}'
        state.episode_step = 10
        assert handler.step(obs, state) is None
        assert state.rewritten_instruction == "Operator instruction"
