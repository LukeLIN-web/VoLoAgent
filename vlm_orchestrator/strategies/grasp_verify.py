"""Grasp check at the gripper-closure event (agent-harness #137).

When the fingers have closed and the flange has since risen 3 cm (the
``_route_lifted`` rule of the place-route branch), the master is asked one
question: is the object in the gripper the one the current subgoal names?
The wording follows VoLo's ``grasp_tool`` rule ("the gripper holds an object
whose identity differs from the subgoal's target"); identity is judged on the
front and external cameras, the wrist camera only shows what is held
(RPent's strict_hybrid_guide, rule 1b). Only ``other`` triggers the
release-and-lift and re-issue; ``target``, ``empty`` and ``unsure`` continue.
Retries are capped at 2 per subgoal and 3 per episode (``retry.CAP``, ``retry.EPISODE_CAP``).
"""
from __future__ import annotations

import json
import re

GRIP_CLOSED = 0.1
LIFT_M = 0.03
VERDICTS = ("target", "other", "empty", "unsure")

SYSTEM_PROMPT = """You are checking a robot arm right after it closed its gripper and lifted.

You see three current images: the front camera and the external camera show the whole scene, the wrist camera looks down from the gripper.

Decide what the gripper is holding, compared with the target object of the current subgoal:
- "target": the gripper holds the object the current subgoal asks to move.
- "other": the gripper holds an object whose identity differs from the subgoal's target.
- "empty": the gripper holds nothing.
- "unsure": you cannot tell.

Judge the identity of the held object on the front and external cameras; the wrist camera is close up and similar-looking objects are easy to confuse there.

Reply with JSON only: {"held": "target" | "other" | "empty" | "unsure", "object": "<what is in the gripper>", "reason": "<one sentence>"}"""


def build_messages(task: str, subgoal: str, front_url: str, external_url: str, wrist_url: str) -> list[dict]:
    """Chat messages for one check; the urls are data: urls (live) or file paths (offline replay)."""
    content = []
    for label, url in (("Front camera", front_url), ("External camera", external_url), ("Wrist camera", wrist_url)):
        content.append({"type": "text", "text": f"NOW — {label}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": f'Overall task: "{task}"\nCurrent subgoal: "{subgoal}"\n\nWhat is the gripper holding?'})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def parse_verdict(text: str) -> str:
    """The ``held`` field of the reply; anything unparsable counts as ``unsure``."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        held = json.loads(m.group(0)).get("held") if m else None
    except (json.JSONDecodeError, AttributeError):
        held = None
    return held if held in VERDICTS else "unsure"
