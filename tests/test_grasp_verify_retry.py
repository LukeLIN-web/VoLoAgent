"""Retry on the grasp check at gripper closure (--retry-trigger closure, strategies/grasp_verify.py)."""

import json

import numpy as np
import pytest

from vlm_orchestrator.grasp.tool import PERCEPTION_JOINTS
from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.retry import CAP
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy

TASK = "Put the banana in the bowl"
SUBGOAL = "Place the banana in the bowl"


def _reply(held):
    return json.dumps({"held": held, "object": "apple", "reason": "red round fruit"})


class _Master:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, system_prompt, user_content, max_tokens=None):
        self.calls.append((system_prompt, user_content))
        return self.reply


class _Handler:
    def __init__(self, due=False):
        self._due = due

    def due(self, state):
        return self._due

    def on_subgoal_advanced(self, obs, state, idx):
        pass


def _strategy(reply, trigger="closure", due=False):
    s = SubgoalStrategy(StrategyContext(vlm=None), SubgoalConfig(), retry_trigger=trigger, retry_action="home",
                        motion_planner="linear")
    s._failure_handler = _Handler(due)
    s._vlm_call = m = _Master(reply)
    return s, m


def _state():
    s = SessionState()
    s.original_instruction, s.subgoals, s.current_subgoal_idx, s.rewritten_instruction = TASK, [SUBGOAL], 0, SUBGOAL
    s.episode_step = 0
    return s


def _obs(grip, z):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    return {
        "prompt": TASK,
        "observation/gripper_position": np.array([grip]),
        "observation/ee_pos": np.array([0.4, 0.0, z]),
        "observation/joint_position": PERCEPTION_JOINTS + 0.5,
        "observation/front_image_left_raw": img,
        "observation/exterior_image_1_left_raw": img,
        "observation/wrist_image_left_raw": img,
    }


def _tick(s, state, grip, z):
    state.episode_step += 8
    _, state, busy = s._tick_retry(_obs(grip, z), state)
    return busy


def _checks(state):
    return [e for e in state.log_entries if e["type"] == "grasp_verify"]


def test_closed_but_not_lifted_is_not_checked():
    s, m = _strategy(_reply("other"))
    state = _state()
    for z in (0.20, 0.21, 0.22, 0.225):
        assert not _tick(s, state, 0.9, z)
    assert not m.calls and not _checks(state)


def test_lift_checks_once_per_closed_segment():
    s, m = _strategy(_reply("target"))
    state = _state()
    _tick(s, state, 0.0, 0.10)                   # open: no segment
    _tick(s, state, 0.9, 0.20)                   # closes at z0 = 0.20
    assert not m.calls
    assert not _tick(s, state, 0.9, 0.235)       # lifted 3.5 cm: one check
    assert len(m.calls) == 1
    _tick(s, state, 0.9, 0.30)                   # same segment: no second check
    assert len(m.calls) == 1
    _tick(s, state, 0.05, 0.30)                  # opens: segment ends
    _tick(s, state, 0.9, 0.30)
    _tick(s, state, 0.9, 0.335)
    assert len(m.calls) == 2
    system_prompt, content = m.calls[0]
    assert sum(c["type"] == "image_url" for c in content) == 3
    assert f'Overall task: "{TASK}"' in content[-1]["text"] and f'Current subgoal: "{SUBGOAL}"' in content[-1]["text"]
    e = _checks(state)[0]
    assert (e["verdict"], e["object"], e["reason"], e["retried"], e["retry_count"]) == (
        "target", "apple", "red round fruit", False, 0)
    assert not state.retry_home_chunks


def _grasp_and_lift(s, state):
    _tick(s, state, 0.9, 0.20)
    return _tick(s, state, 0.9, 0.235)


def test_other_retries_until_the_cap():
    s, m = _strategy(_reply("other"))
    state = _state()
    for n in range(1, CAP + 1):
        assert _grasp_and_lift(s, state)
        assert state.retry_home_chunks and s._retry.count == n
        retry = [e for e in state.log_entries if e["type"] == "retry"][-1]
        assert (retry["trigger"], retry["reason"], retry["count"]) == ("closure", "other", n)
        assert _checks(state)[-1]["retried"] and _checks(state)[-1]["retry_count"] == n
        while state.retry_home_chunks:                 # the proxy serves the home chunks: no check meanwhile
            assert _tick(s, state, 0.9, 0.40)
            state.retry_home_chunks.pop(0)
        assert not _tick(s, state, 0.0, 0.40)          # re-issued as written
        assert state.rewritten_instruction == SUBGOAL
    assert len(m.calls) == CAP
    assert not _grasp_and_lift(s, state)               # third other: checked, not retried
    e = _checks(state)[-1]
    assert len(m.calls) == CAP + 1 and e["verdict"] == "other" and not e["retried"] and e["retry_count"] == CAP
    assert not state.retry_home_chunks


@pytest.mark.parametrize("reply,verdict", [
    (_reply("target"), "target"), (_reply("empty"), "empty"), (_reply("unsure"), "unsure"), ("no json", "unsure"),
])
def test_other_verdicts_do_not_retry(reply, verdict):
    s, m = _strategy(reply)
    state = _state()
    assert not _grasp_and_lift(s, state)
    assert len(m.calls) == 1 and _checks(state)[0]["verdict"] == verdict
    assert not state.retry_home_chunks and s._retry.count == 0


def test_missing_inputs_skip_the_check():
    s, m = _strategy(_reply("other"))
    state = _state()
    obs = _obs(0.9, 0.20)
    del obs["observation/ee_pos"]
    s._tick_retry(obs, state)
    s._tick_retry(_obs(0.9, 0.20), state)
    obs = _obs(0.9, 0.24)
    del obs["observation/gripper_position"]
    s._tick_retry(obs, state)
    assert not m.calls


def test_no_check_without_closure():
    s, m = _strategy(_reply("other"), trigger="gt", due=False)
    state = _state()
    assert not _grasp_and_lift(s, state)
    assert not m.calls and not _checks(state)
    s, _ = _strategy(_reply("other"), trigger="none")
    assert s._retry is None
