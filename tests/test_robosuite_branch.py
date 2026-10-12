"""The robosuite (``--env libero``) branch of tool_chain, as RoboCasa365 runs it (agent-harness #28):
the master sees only the NOW image, a refused resend costs no budget and escalates to a replan and
then an abort, and a tool ends by taking the arm and base back to the episode-start pose."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np

from vlm_orchestrator.grasp.tool import LiberoHoming
from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.tool_chain import ToolChainConfig, ToolChainStrategy
from vlm_orchestrator.vlm import PassthroughVLM


def _img():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _state() -> SessionState:
    s = SessionState()
    s.episode_id = 1
    s.original_instruction = "put cube in bowl"
    s.subgoals = ["pick cube", "place cube in bowl"]
    s.current_subgoal_idx = 0
    s.subgoals_ordered = True
    s.initial_image = _img()
    s.tool_chain_active = True
    return s


def _obs() -> dict:
    return {"observation/gripper_position": np.array([0.0]), "observation/ee_pos": np.array([0.4, 0.0, 0.5])}


def _strategy(env_mode: str) -> ToolChainStrategy:
    s = ToolChainStrategy(StrategyContext(vlm=PassthroughVLM()), ToolChainConfig(max_identical_calls=2),
                          env_mode=env_mode, master_points=True)
    s.ctx.get_vlm_image = lambda obs: _img()
    s._activate_grasp_for_tool_chain = MagicMock()
    return s


GRASP = {"subgoal_action": "continue", "tool": "grasp", "args": {"target": "red block", "point": [0.5, 0.5]}, "reason": "go"}


class TestMasterInput:
    def test_robosuite_master_sees_only_the_now_image(self):
        s = _strategy("libero")
        s._vlm_call = MagicMock(return_value=json.dumps(GRASP))
        s._on_step(_obs(), _state())
        system_prompt, user_content = s._vlm_call.call_args[0]
        assert "BEFORE" not in system_prompt
        assert "the tools see this same image" in system_prompt
        assert sum(1 for c in user_content if c["type"] == "image_url") == 1

    def test_robolab_master_still_sees_before_and_now(self):
        s = _strategy("robolab")
        s._vlm_call = MagicMock(return_value=json.dumps(GRASP))
        s._on_step(_obs(), _state())
        system_prompt, user_content = s._vlm_call.call_args[0]
        assert "BEFORE image" in system_prompt
        assert sum(1 for c in user_content if c["type"] == "image_url") == 2


class TestRefusal:
    def _failed_state(self) -> SessionState:
        state = _state()
        state.tool_chain_last_tool, state.tool_chain_last_status = "grasp", "FAILED"
        state.tool_chain_last_reason = "Failed: nothing there"
        state.tool_chain_last_sig = ("grasp", "red block", (0.5, 0.5))
        state.tool_chain_repeats = 1   # the resend that was executed and failed again
        return state

    def test_refusals_cost_nothing_then_replan_then_abort(self):
        s = _strategy("libero")
        s._vlm_call = MagicMock(return_value=json.dumps(GRASP))
        state = self._failed_state()
        with patch.object(s, "_recycle", return_value=True) as recycle:
            s._on_step(_obs(), state)                       # refusal 1
            assert not s._activate_grasp_for_tool_chain.called
            assert (state.tool_chain_tool_calls, state.tool_chain_subgoal_calls) == (0, 0)
            assert "REFUSED" in state.tool_chain_last_reason
            assert state.tool_chain_refusals == 1 and not recycle.called
            s._on_step(_obs(), state)                       # refusal 2 → forced replan
            assert recycle.called and state.tool_chain_refusal_replanned
            assert not state.tool_chain_aborted
            s._on_step(_obs(), state)                       # refusal 3 → abort
        assert state.tool_chain_aborted
        assert not s._activate_grasp_for_tool_chain.called
        assert (state.tool_chain_tool_calls, state.tool_chain_subgoal_calls) == (0, 0)

    def test_a_different_call_resets_the_refusals(self):
        s = _strategy("libero")
        state = self._failed_state()
        s._vlm_call = MagicMock(return_value=json.dumps(GRASP))
        s._on_step(_obs(), state)
        assert state.tool_chain_refusals == 1
        other = dict(GRASP, args={"target": "blue block", "point": [0.2, 0.2]})
        s._vlm_call = MagicMock(return_value=json.dumps(other))
        s._on_step(_obs(), state)
        assert s._activate_grasp_for_tool_chain.called
        assert state.tool_chain_refusals == 0 and state.tool_chain_tool_calls == 1

    def test_robolab_never_refuses(self):
        s = _strategy("robolab")
        s._vlm_call = MagicMock(return_value=json.dumps(GRASP))
        state = self._failed_state()
        state.tool_chain_repeats = 5
        s._on_step(_obs(), state)
        assert s._activate_grasp_for_tool_chain.called
        assert state.tool_chain_tool_calls == 1


class TestLiberoHoming:
    """Perfect tracking: a 7-D chunk moves the EE by its first five rows, a 9-D chunk drives the base
    2 cm toward home."""

    def _obs(self, ee, base):
        return {"observation/ee_pos": np.array(ee, float), "observation/ee_pos_home": np.array([0.3, 0.0, 0.6]),
                "observation/base_pos": np.array([*base, 0.0]), "observation/base_yaw": np.array([0.0]),
                "observation/base_home": np.array([0.0, 0.0])}

    def test_not_wanted_without_a_home(self):
        assert not LiberoHoming.wanted({"observation/ee_pos": np.zeros(3)})

    def test_up_then_base_then_across(self):
        ee, base = np.array([0.6, 0.1, 0.3]), np.array([0.2, 0.0])
        h, kinds = LiberoHoming(), []
        for _ in range(200):
            chunk = h.chunk(self._obs(ee, base), -1.0, 10)
            if chunk is None:
                break
            if chunk.shape[1] == 9:
                kinds.append("base")
                base = base + np.clip(-base, -0.02, 0.02)
            else:
                d = chunk[:5, :3].sum(axis=0) * 0.05
                kinds.append("up" if abs(d[0]) + abs(d[1]) < 1e-9 else "across")
                ee = ee + d
        assert chunk is None
        order = [k for i, k in enumerate(kinds) if i == 0 or kinds[i - 1] != k]
        assert order == ["up", "base", "across"]
        assert np.allclose(ee, [0.3, 0.0, 0.6], atol=0.03) and np.linalg.norm(base) <= 0.03

    def test_a_stalled_segment_is_skipped(self):
        obs = self._obs([0.6, 0.1, 0.3], [0.0, 0.0])   # the EE never moves
        h, n = LiberoHoming(), 0
        while h.chunk(obs, -1.0, 10) is not None:
            n += 1
            assert n < 100
        assert n > 0
