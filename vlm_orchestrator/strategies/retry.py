"""Retry of the current subgoal at a monitor check (agent-harness 「先决：真值重来」).

At a monitor check that is due, before the master is asked, a trigger may fire; when it does the master is not
asked at that check and the current subgoal is re-issued, as written, either in place (the VLA is simply re-prompted)
or after the arm goes home (gripper opened, then the grasp tool's lift to ``PERCEPTION_JOINTS``, the pose every
episode starts from). The triggers act only at the checks the 80-step monitor already makes:

- ``gt``: the ground-truth strata of agent-harness ``regrasp_read.py`` at that check — ``dropped`` (a target of the
  subgoal was in the gripper at an earlier check on this subgoal, is not now and is not done), ``late`` (no target of
  the subgoal has ever been lifted this episode, from the ``LATE_CHECKS``-th check on), plus VoLo's own
  ``wrong_object_picked`` rule (``metrics1_gt_rules.py``: the gripper holds an object that is no target and touches
  no undone target);
- ``timer``: ``TIMER_FRAC`` of the episode's time budget has passed since the subgoal started or was last retried.

Both share the cooldown: never at the first check after a subgoal starts or is retried. The check count restarts, as
in ``regrasp_read.py``, after a decision that is not continue.

- ``closure`` (agent-harness #137) acts on every chunk instead: once per closed-gripper segment, when the flange has
  risen ``LIFT_M`` since the fingers closed, the master is asked whether the held object is the subgoal's target
  (``grasp_verify.py``); it fires on ``other``. No cooldown.

All triggers share the cap: at most ``CAP`` retries per subgoal.
"""
from __future__ import annotations

import numpy as np

from vlm_orchestrator.diagnostics.metrics1_gt_rules import current_subtask_targets
from vlm_orchestrator.grasp.tool import (
    GRIPPER_OPEN, INTERP_STEPS_LIFT, PERCEPTION_JOINTS, RELEASE_HOLD_STEPS,
)
from vlm_orchestrator.gt_metrics.detectors import _target_objects_from_gt

from .grasp_verify import GRIP_CLOSED, LIFT_M

TRIGGERS = ("none", "gt", "timer", "closure")
ACTIONS = ("inplace", "home")
LATE_CHECKS = 3     # regrasp_read.py LATE (VoLoAgent 1a38752 REGRASP_AFTER_CHECKS)
MIN_CHECKS = 2      # cooldown
CAP = 2             # retries per subgoal
TIMER_FRAC = 0.5    # three in four of GPT-5.5's successful collection episodes on these openings end within half the budget
CHUNK = 8


def gt_reason(gt: dict | None, subgoal: str, held_before: set, k: int) -> str | None:
    if not gt:
        return None
    sub = gt.get("subtask") or {}
    completed = sub.get("object_completed") or {}
    targets = _target_objects_from_gt(gt, text_hint=subgoal)   # the targets gt_state_check records for this subgoal
    undone = [t for t in targets if not completed.get(t)]
    robot = gt.get("robot") or {}
    held = robot.get("grasped_object")
    if held is not None:
        any_target = set(targets) | set(current_subtask_targets(gt)) | set(completed)
        touching = set(robot.get("objects_in_contact") or [])
        return "wrong_object" if held not in any_target and not touching & set(undone) else None
    objects = gt.get("objects") or {}
    if not undone or any(not isinstance(objects.get(t), dict) for t in undone):
        return None
    if any(t in held_before for t in undone):
        return "dropped"
    if any(objects[t].get("lifted") for t in undone):
        return None
    return "late" if k >= LATE_CHECKS else None


def home_chunks(joints: np.ndarray, planner) -> list[np.ndarray]:
    """The grasp tool's preamble as 8-step chunks: hold with the gripper open for RELEASE_HOLD_STEPS, then its lift
    to PERCEPTION_JOINTS (same planner, steps and skip distance). Raises RuntimeError when the lift has no plan."""
    joints = np.asarray(joints, dtype=np.float64).reshape(-1)[:7]
    rows = [np.r_[joints, GRIPPER_OPEN]] * RELEASE_HOLD_STEPS
    if np.linalg.norm(joints - PERCEPTION_JOINTS) >= 0.1:
        plan = planner.plan_segment(joints, PERCEPTION_JOINTS, n_steps=INTERP_STEPS_LIFT, scene_pc=None, phase="lift")
        if not plan.success or plan.waypoints is None:
            raise RuntimeError(f"no lift plan ({plan.label}): {plan.detail}")
        rows += [np.r_[np.asarray(q, dtype=np.float64)[:7], GRIPPER_OPEN] for q in plan.waypoints]
    rows += [rows[-1]] * (-len(rows) % CHUNK)
    return [np.stack(rows[i:i + CHUNK]) for i in range(0, len(rows), CHUNK)]


class Retry:
    """Per-proxy-session bookkeeping of the checks on the current subgoal; ``reset`` at every episode start."""

    def __init__(self, trigger: str, action: str):
        assert trigger in TRIGGERS and action in ACTIONS, (trigger, action)
        self.trigger, self.action = trigger, action
        self.reset()

    def reset(self) -> None:
        self.key = None
        self.k = 0
        self.held: set = set()
        self.count = 0
        self.clock = 0
        self.last_check = 0
        self.last_action = None
        self.end_segment()

    def end_segment(self) -> None:
        """closure: forget the closed-gripper segment (fingers opened, a tool took over, or a retry fired)."""
        self.z0 = None
        self.verified = False

    def on_chunk(self, state, grip: float, z: float) -> bool:
        """closure, every chunk: track the closed-gripper segment from the gripper position (0 open, 1 closed) and
        the flange height; True once per segment, when the flange has risen LIFT_M since the fingers closed."""
        key = (state.current_subgoal_idx, state.rewritten_instruction)
        if key != self.key:   # a new subgoal: its own cap
            self.key, self.count = key, 0
        if grip <= GRIP_CLOSED:
            self.end_segment()
            return False
        if self.z0 is None:
            self.z0 = z
        if self.verified or z - self.z0 < LIFT_M:
            return False
        self.verified = True
        return True

    def on_check(self, state, gt: dict | None) -> str | None:
        """A monitor check is due: update the bookkeeping, return why to retry now (None: ask the master)."""
        key = (state.current_subgoal_idx, state.rewritten_instruction)
        if key != self.key:   # the subgoal changed at the previous check (or the episode started)
            self.key, self.k, self.held, self.count, self.clock = key, 0, set(), 0, self.last_check
        elif self.last_action not in (None, "continue"):
            self.k = 0
        self.k += 1
        self.last_check = state.episode_step
        self.last_action = None
        held_before = set(self.held)
        held = ((gt or {}).get("robot") or {}).get("grasped_object")
        if held:
            self.held.add(held)
        if self.trigger == "none" or self.count >= CAP or self.k < MIN_CHECKS:
            return None
        if self.trigger == "timer":
            budget = (gt or {}).get("episode_max_steps")
            return "timer" if budget and state.episode_step - self.clock >= TIMER_FRAC * budget else None
        return gt_reason(gt, key[1] or "", held_before, self.k)

    def fired(self, step: int) -> None:
        self.count += 1
        self.k = 0
        self.held = set()
        self.clock = step
        self.end_segment()
