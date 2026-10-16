"""Retry of the current subgoal at a due monitor check (strategies/retry.py)."""

import numpy as np

from vlm_orchestrator.grasp.tool import GRIPPER_OPEN, PERCEPTION_JOINTS, RELEASE_HOLD_STEPS
from vlm_orchestrator.motion import build_motion_planner
from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.retry import CAP, Retry, gt_reason, home_chunks
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy

SUBGOAL = "Place the banana in the bowl"


def _gt(held=None, lifted=False, done=False, contact=(), budget=900):
    return {
        "scene_objects": ["banana", "bowl", "apple"],
        "objects": {n: {"pos": [0, 0, 0], "lifted": lifted and n == "banana"} for n in ("banana", "bowl", "apple")},
        "subtask": {"conditions": [{"object": "banana", "condition_idx": 0, "satisfied": done}],
                    "object_completed": {"banana": done}},
        "robot": {"grasped_object": held, "objects_in_contact": list(contact)},
        "episode_max_steps": budget,
    }


def _state(step, idx=0, text=SUBGOAL):
    s = SessionState()
    s.episode_step, s.current_subgoal_idx, s.rewritten_instruction, s.subgoals = step, idx, text, [text]
    return s


def test_gt_reason_strata():
    assert gt_reason(_gt(held="banana"), SUBGOAL, set(), 5) is None             # holding the target
    assert gt_reason(_gt(held="apple"), SUBGOAL, set(), 2) == "wrong_object"
    assert gt_reason(_gt(held="apple", contact=["banana"]), SUBGOAL, set(), 2) is None   # touching the target
    assert gt_reason(_gt(), SUBGOAL, {"banana"}, 2) == "dropped"
    assert gt_reason(_gt(done=True), SUBGOAL, {"banana"}, 4) is None             # done
    assert gt_reason(_gt(), SUBGOAL, set(), 2) is None and gt_reason(_gt(), SUBGOAL, set(), 3) == "late"
    assert gt_reason(_gt(lifted=True), SUBGOAL, set(), 4) is None                # lifted once, never held at a check


def test_cooldown_cap_and_check_count():
    r = Retry("gt", "home")
    assert r.on_check(_state(80), _gt()) is None                  # check 1
    r.last_action = "replan"
    assert r.on_check(_state(160), _gt()) is None and r.k == 1    # a non-continue decision restarts the count
    r.last_action = "continue"
    assert r.on_check(_state(240), _gt()) is None and r.k == 2
    r.last_action = "continue"
    assert r.on_check(_state(320), _gt()) == "late"
    for i in range(CAP):
        r.fired(320 + 80 * i)
    for step in (400, 480, 560, 640):
        assert r.on_check(_state(step), _gt()) is None            # cap reached on this subgoal
    assert r.on_check(_state(720, idx=1), _gt()) is None and r.count == 0   # a new subgoal resets the cap


def test_dropped_needs_a_hold_at_an_earlier_check_since_the_last_retry():
    r = Retry("gt", "inplace")
    r.on_check(_state(80), _gt(held="banana"))
    r.last_action = "continue"
    assert r.on_check(_state(160), _gt()) == "dropped"
    r.fired(160)
    r.on_check(_state(240), _gt())
    r.last_action = "continue"
    assert r.on_check(_state(320), _gt(lifted=True)) is None      # the hold was before the retry


def test_timer_half_the_budget_since_the_subgoal_or_retry():
    r = Retry("timer", "home")
    for step in (80, 160, 240, 320, 400):
        assert r.on_check(_state(step), _gt(held="banana")) is None
        r.last_action = "continue"
    assert r.on_check(_state(480), _gt(held="banana")) == "timer"   # 480 >= 450, whatever the ground truth says
    r.fired(480)
    r.on_check(_state(560), _gt())
    r.last_action = "continue"
    assert r.on_check(_state(640), _gt()) is None


def test_home_chunks_release_then_lift():
    q = PERCEPTION_JOINTS + 0.5
    chunks = home_chunks(np.r_[q, 0.9], build_motion_planner("linear"))
    rows = np.concatenate(chunks)
    assert all(c.shape == (8, 8) for c in chunks)
    assert np.allclose(rows[:RELEASE_HOLD_STEPS, :7], q) and np.all(rows[:, 7] == GRIPPER_OPEN)
    assert np.allclose(rows[-1, :7], PERCEPTION_JOINTS)
    assert len(home_chunks(PERCEPTION_JOINTS, build_motion_planner("linear"))) == 2   # already home: release only


class _DueHandler:
    def __init__(self):
        self.advanced = 0

    def due(self, state):
        return True

    def on_subgoal_advanced(self, obs, state, idx):
        self.advanced += 1


def test_tick_retry_goes_home_then_reissues():
    s = SubgoalStrategy(StrategyContext(vlm=None), SubgoalConfig(), retry_trigger="gt", retry_action="home",
                        motion_planner="linear")
    s._failure_handler = h = _DueHandler()
    state = _state(80)
    obs = {"prompt": SUBGOAL, "observation/joint_position": PERCEPTION_JOINTS + 0.5, "gt_state": _gt(held="apple")}
    s._tick_retry(obs, state)                                    # check 1: cooldown
    s._retry.last_action = "continue"
    state.episode_step = 160
    _, state, busy = s._tick_retry(obs, state)
    assert busy and state.retry_home_chunks and s._retry_reissue
    assert [e for e in state.log_entries if e["type"] == "retry"][0]["reason"] == "wrong_object"
    while state.retry_home_chunks:
        assert s._tick_retry(obs, state)[2]
        state.retry_home_chunks.pop(0)
    _, state, busy = s._tick_retry(obs, state)
    assert not busy and not s._retry_reissue and h.advanced == 1 and state.rewritten_instruction == SUBGOAL


def test_default_is_vola_loop():
    s = SubgoalStrategy(StrategyContext(vlm=None), SubgoalConfig())
    assert s._retry is None and s._max_recycles == 3
