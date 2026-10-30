# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""--grasp-noun-phrase (SAM3 gets the target's noun phrase) and
--pick-with-grasp-tool (each subgoal starts with the grasp tool picking its
object)."""

import sys

import numpy as np
import pytest

from vlm_orchestrator import cli
from vlm_orchestrator.grasp.tool import GraspPhase, subgoal_pick_target, target_noun_phrase
from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy

# The subgoal tests' fakes; imported as a module so pytest does not collect
# TestableSubgoalStrategy a second time from here.
import test_subgoal as subgoal_fakes


# ---- target_noun_phrase ----


@pytest.mark.parametrize("text, phrase", [
    ("the yellow lemon on the table near the grey bin", "yellow lemon"),
    ("the orange orange near the grey bin", "orange orange"),
    ("Place the lemon on the plate, then in the grey bin", "lemon"),
    ("Pick each lime and place it in the grey bin", "lime"),
    ("the orange pumpkin and the small orange fruit on the table near the grey bin",
     "orange pumpkin"),
    ("the dark green avocado", "dark green avocado"),
    ("lemon", "lemon"),
    ("yellow-orange orange near the grey bin", "yellow-orange orange"),
    ("Place each of the four green limes into the grey bin", "green limes"),
])
def test_target_noun_phrase(text, phrase):
    assert target_noun_phrase(text) == phrase


@pytest.mark.parametrize("subgoal, target", [
    ("Place the yellow banana on the beige plate", "yellow banana"),
    ("Pick the orange near the grey bin and place it on the plate", "orange"),
    ("Pick up the red block and put it in the bin", "red block"),
    ("Open the drawer and put the lemon in it", None),
    ("Open the drawer", None),
])
def test_subgoal_pick_target(subgoal, target):
    assert subgoal_pick_target(subgoal) == target


# ---- CLI ----


class _Parsed(Exception):
    """Raised in place of building the strategy, carrying the parsed args."""


def _parse(monkeypatch, tmp_path, *argv):
    def capture(args):
        raise _Parsed(args)

    monkeypatch.setattr(cli, "_build_strategy", capture)
    monkeypatch.setattr(
        sys, "argv", ["vlm_orchestrator.cli", "--log-dir", str(tmp_path), *argv],
    )
    with pytest.raises(_Parsed) as parsed:
        cli.main()
    return parsed.value.args[0]


def test_flags_default_off(monkeypatch, tmp_path):
    args = _parse(monkeypatch, tmp_path, "--mode", "subgoal")
    assert args.grasp_noun_phrase is False
    assert args.pick_with_grasp_tool is False


def test_flags_parse_with_subgoal(monkeypatch, tmp_path):
    args = _parse(
        monkeypatch, tmp_path, "--mode", "subgoal",
        "--grasp-noun-phrase", "--pick-with-grasp-tool",
    )
    assert args.grasp_noun_phrase is True
    assert args.pick_with_grasp_tool is True


@pytest.mark.parametrize("flag", ["--pick-with-grasp-tool", "--grasp-noun-phrase"])
def test_flag_rejected_outside_subgoal(monkeypatch, tmp_path, capsys, flag):
    monkeypatch.setattr(
        sys, "argv",
        ["vlm_orchestrator.cli", "--log-dir", str(tmp_path),
         "--mode", "tool_chain", flag],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert flag in capsys.readouterr().err


# ---- --pick-with-grasp-tool on the subgoal strategy ----


class _FakeGraspExecutor:
    """Stands in for GraspToolExecutor and records the targets it is given."""

    status_message = ""
    _grasp_confidence = 0.0

    def __init__(self, starts: bool = True):
        self.targets: list[str] = []
        self._starts = starts
        self.phase = GraspPhase.IDLE

    @property
    def is_active(self) -> bool:
        return self.phase not in (GraspPhase.IDLE, GraspPhase.DONE, GraspPhase.FAILED)

    def start(self, target_object, obs, state, stack=True, point=None):
        self.targets.append(target_object)
        self.phase = GraspPhase.RELEASING if self._starts else GraspPhase.FAILED

    def reset(self):
        self.phase = GraspPhase.IDLE


class _PickStrategy(subgoal_fakes.TestableSubgoalStrategy):
    def __init__(self, mock_vlm, **kwargs):
        SubgoalStrategy.__init__(
            self, StrategyContext(vlm=mock_vlm),
            SubgoalConfig(check_mode="vlm", check_interval=1, subgoal_timeout=100),
            **kwargs,
        )
        self._mock = mock_vlm


def _first_step(subgoals, *, pick, starts=True, gripper=None):
    """Decompose "task" into *subgoals* on the episode's first step."""
    mock = subgoal_fakes.MockSubgoalVLM()
    mock.decompose_responses["task"] = subgoals
    strategy = _PickStrategy(mock, pick_with_grasp_tool=pick)
    state = SessionState()
    executor = _FakeGraspExecutor(starts=starts)
    state.grasp_tool_executor = executor
    obs = subgoal_fakes._make_obs("task")
    if gripper is not None:
        obs["observation/gripper_position"] = np.array([gripper])
    strategy.process(obs, state)
    return strategy, state, executor, mock


def _picks(state):
    return [e for e in state.log_entries if e["type"] == "pick_with_grasp_tool"]


def test_first_subgoal_starts_with_pick():
    strategy, state, executor, _ = _first_step(
        ["Place the yellow lemon in the grey bin"], pick=True,
    )
    assert executor.targets == ["yellow lemon"]
    assert state.grasp_tool_active
    assert state.grasp_tool_target == "yellow lemon"
    assert [(e["target_object"], e["subgoal_idx"], e["started"]) for e in _picks(state)] == [
        ("yellow lemon", 0, True),
    ]
    assert strategy._grasp_attempts == 0


def test_no_pick_when_off():
    _, state, executor, _ = _first_step(
        ["Place the yellow lemon in the grey bin"], pick=False,
    )
    assert executor.targets == []
    assert not state.grasp_tool_active
    assert _picks(state) == []


def test_no_pick_without_grasp_verb():
    _, state, executor, _ = _first_step(["Open the drawer"], pick=True)
    assert executor.targets == []
    assert not state.grasp_tool_active


def test_no_pick_with_closed_gripper():
    _, state, executor, _ = _first_step(
        ["Place the yellow lemon in the grey bin"], pick=True, gripper=0.5,
    )
    assert executor.targets == []
    assert not state.grasp_tool_active


def test_pick_that_does_not_start_hands_over_to_vla():
    _, state, executor, _ = _first_step(
        ["Place the yellow lemon in the grey bin"], pick=True, starts=False,
    )
    assert executor.targets == ["yellow lemon"]
    assert not state.grasp_tool_active
    assert [e["started"] for e in _picks(state)] == [False]


def test_pick_takes_the_object_not_it():
    _, _, executor, _ = _first_step(
        ["Pick the orange and place it in the grey bin"], pick=True,
    )
    assert executor.targets == ["orange"]


def test_next_subgoal_starts_with_pick_after_the_first_ends():
    strategy, state, executor, mock = _first_step(
        ["Place the yellow lemon in the grey bin", "Place the orange in the grey bin"],
        pick=True,
    )
    mock.check_responses = [{"action": "next"}]
    executor.phase = GraspPhase.DONE
    for _ in range(2):
        state.infer_count += 1
        state.episode_step += 1
        strategy.process(subgoal_fakes._make_obs("task"), state)
    assert state.current_subgoal_idx == 1
    assert executor.targets == ["yellow lemon", "orange"]
    assert [(e["subgoal_idx"], e["started"]) for e in _picks(state)] == [(0, True), (1, True)]
