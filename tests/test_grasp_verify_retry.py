"""Retry on the grasp check at gripper closure (--retry-trigger closure, strategies/grasp_verify.py)."""

import json

import numpy as np
import pytest

from vlm_orchestrator.grasp.tool import PERCEPTION_JOINTS
from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.retry import CAP, EPISODE_CAP, Retry
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy

TASK = "Put the banana in the bowl"
SUBGOAL = "Place the banana in the bowl"
SUBGOAL2 = "Place the apple in the bowl"


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

    def on_episode_start(self, obs, state):
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


def _retry_cycle(s, state, n, total):
    """An ``other`` check that retries (the n-th on this subgoal, the total-th this episode), then home and re-issue."""
    subgoal = state.rewritten_instruction
    assert _grasp_and_lift(s, state)
    assert state.retry_home_chunks and (s._retry.count, s._retry.episode_count) == (n, total)
    retry = [e for e in state.log_entries if e["type"] == "retry"][-1]
    assert (retry["trigger"], retry["reason"], retry["count"]) == ("closure", "other", n)
    e = _checks(state)[-1]
    assert e["retried"] and (e["retry_count"], e["episode_retry_count"]) == (n, total) and "cap" not in e
    while state.retry_home_chunks:                     # the proxy serves the home chunks: no check meanwhile
        assert _tick(s, state, 0.9, 0.40)
        state.retry_home_chunks.pop(0)
    assert not _tick(s, state, 0.0, 0.40)              # re-issued as written
    assert state.rewritten_instruction == subgoal


def _blocked(s, state, n, total, cap):
    assert not _grasp_and_lift(s, state)               # checked, not retried
    e = _checks(state)[-1]
    assert e["verdict"] == "other" and not e["retried"] and e["cap"] == cap
    assert (e["retry_count"], e["episode_retry_count"]) == (n, total)
    assert not state.retry_home_chunks
    _tick(s, state, 0.0, 0.40)                         # fingers open: the segment ends


def test_other_retries_until_the_cap():
    s, m = _strategy(_reply("other"))
    state = _state()
    for n in range(1, CAP + 1):
        _retry_cycle(s, state, n, n)
    assert len(m.calls) == CAP
    _blocked(s, state, CAP, CAP, "subgoal")            # third other on the subgoal
    assert len(m.calls) == CAP + 1


def _advance(state, subgoals, idx):
    """What advancing or a replan/recycle leaves for the retry: a new subgoal list, index and instruction."""
    state.subgoals, state.current_subgoal_idx, state.rewritten_instruction = subgoals, idx, subgoals[idx]


def test_episode_cap_spans_subgoals():
    s, _ = _strategy(_reply("other"))
    state = _state()
    state.subgoals = [SUBGOAL, SUBGOAL2]
    _retry_cycle(s, state, 1, 1)
    _retry_cycle(s, state, 2, 2)
    _advance(state, [SUBGOAL, SUBGOAL2], 1)            # the next subgoal: its own per-subgoal cap
    _retry_cycle(s, state, 1, EPISODE_CAP)
    _blocked(s, state, 1, EPISODE_CAP, "episode")


def test_episode_cap_survives_replans_and_rewrites():
    s, _ = _strategy(_reply("other"))
    state = _state()
    _retry_cycle(s, state, 1, 1)
    _advance(state, [SUBGOAL2], 0)                     # replan: same index, rewritten subgoal
    _retry_cycle(s, state, 1, 2)
    _advance(state, [SUBGOAL, SUBGOAL2], 0)            # recycle: a new list
    _retry_cycle(s, state, 1, 3)
    _advance(state, [SUBGOAL2], 0)
    _blocked(s, state, 0, EPISODE_CAP, "episode")


def test_new_episode_clears_the_episode_cap():
    s, _ = _strategy(_reply("other"))
    state = _state()
    state.subgoals = [SUBGOAL, SUBGOAL2]
    for n, idx in ((1, 0), (2, 0), (1, 1)):
        _advance(state, [SUBGOAL, SUBGOAL2], idx)
        _retry_cycle(s, state, n, s._retry.episode_count + 1)
    _blocked(s, state, 1, EPISODE_CAP, "episode")
    s._on_new_episode({"prompt": TASK}, state, TASK)   # no image: no decomposition
    assert s._retry.episode_count == 0
    _advance(state, [SUBGOAL], 0)
    _retry_cycle(s, state, 1, 1)


def test_gt_and_timer_ignore_the_episode_cap():
    for trigger in ("gt", "timer"):
        r = Retry(trigger, "inplace")
        r.episode_count = EPISODE_CAP
        state = SessionState()
        state.current_subgoal_idx, state.rewritten_instruction, state.subgoals = 0, SUBGOAL, [SUBGOAL]
        gt = {   # test_retry.py's _gt(): nothing held or lifted, so gt says "late" at check 3; timer fires at check 2
            "scene_objects": ["banana", "bowl", "apple"],
            "objects": {n: {"pos": [0, 0, 0], "lifted": False} for n in ("banana", "bowl", "apple")},
            "subtask": {"conditions": [{"object": "banana", "condition_idx": 0, "satisfied": False}],
                        "object_completed": {"banana": False}},
            "robot": {"grasped_object": None, "objects_in_contact": []},
            "episode_max_steps": 160,
        }
        reasons = []
        for step in (80, 160, 240):
            state.episode_step = step
            reasons.append(r.on_check(state, gt))
            r.last_action = "continue"
        assert reasons == ([None, None, "late"] if trigger == "gt" else [None, "timer", "timer"]), (trigger, reasons)


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
