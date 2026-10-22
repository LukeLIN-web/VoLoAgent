# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""reference_hook: the sentence π0.5 gets with the reference image (--reference-prompt, agent-harness #77)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vlm_orchestrator.perception.reference_image import GENERIC_PROMPT, REFERENCE_KEY
from vlm_orchestrator.strategies import reference_hook as rh
from vlm_orchestrator.strategies.base import SessionState, StrategyContext

SQUARE = np.array([[10.0, 10.0], [60.0, 10.0], [60.0, 60.0], [10.0, 60.0]])
OTHER = SQUARE + 100.0


def _hook(monkeypatch, prompt_mode, polygons):
    """A gt-mode hook whose polygons come from ``polygons`` (a list, one (target, destination) per computation)."""
    hook = rh.ReferenceHook(StrategyContext(vlm=None), "gt", prompt=prompt_mode)
    calls = []

    def fake_gt(obs, prompt, record):
        calls.append(prompt)
        return polygons[len(calls) - 1]

    monkeypatch.setattr(hook, "_gt", fake_gt)
    return hook, calls


def _obs(prompt):
    return {"prompt": prompt, rh.EXTERIOR_224: np.full((224, 224, 3), 200, np.uint8)}


def _state(tmp_path):
    return SessionState(episode_id=3, infer_count=0, episode_log_dir=str(tmp_path))


def test_generic_swaps_the_sentence_while_both_polygons_are_drawn(monkeypatch, tmp_path):
    hook, calls = _hook(monkeypatch, "generic", [(SQUARE, OTHER)])
    state = _state(tmp_path)
    obs = _obs("put the apple in the bowl")
    out = hook(obs, state)
    assert out["prompt"] == GENERIC_PROMPT and REFERENCE_KEY in out
    assert obs["prompt"] == "put the apple in the bowl" and REFERENCE_KEY not in obs   # the strategy's obs is untouched
    state.infer_count = 1
    again = hook(_obs("put the apple in the bowl"), state)   # same subgoal: cached, not recomputed
    assert again["prompt"] == GENERIC_PROMPT and calls == ["put the apple in the bowl"]
    (line,) = [json.loads(x) for x in open(tmp_path / "reference.jsonl")]
    assert line["prompt"] == "put the apple in the bowl" and line["vla_prompt"] == GENERIC_PROMPT


@pytest.mark.parametrize("polygons", [(SQUARE, None), (None, OTHER)])
def test_generic_keeps_the_subgoal_with_one_polygon(monkeypatch, tmp_path, polygons):
    hook, _ = _hook(monkeypatch, "generic", [polygons])
    out = hook(_obs("put the apple on the table"), _state(tmp_path))
    assert out["prompt"] == "put the apple on the table" and REFERENCE_KEY in out
    (line,) = [json.loads(x) for x in open(tmp_path / "reference.jsonl")]
    assert line["vla_prompt"] == "put the apple on the table"


def test_generic_without_an_image_leaves_the_request_alone(monkeypatch, tmp_path):
    hook, _ = _hook(monkeypatch, "generic", [(None, None)])
    obs = _obs("open the drawer")
    assert hook(obs, _state(tmp_path)) is obs


def test_subgoal_mode_keeps_the_subgoal(monkeypatch, tmp_path):
    hook, _ = _hook(monkeypatch, "subgoal", [(SQUARE, OTHER)])
    out = hook(_obs("put the apple in the bowl"), _state(tmp_path))
    assert out["prompt"] == "put the apple in the bowl" and REFERENCE_KEY in out


def test_a_new_subgoal_recomputes_image_and_sentence(monkeypatch, tmp_path):
    hook, calls = _hook(monkeypatch, "generic", [(SQUARE, OTHER), (SQUARE, None)])
    state = _state(tmp_path)
    assert hook(_obs("put the apple in the bowl"), state)["prompt"] == GENERIC_PROMPT
    state.infer_count = 5
    assert hook(_obs("put the pear on the table"), state)["prompt"] == "put the pear on the table"
    assert calls == ["put the apple in the bowl", "put the pear on the table"]


def test_tool_steps_are_left_alone(monkeypatch, tmp_path):
    hook, calls = _hook(monkeypatch, "generic", [(SQUARE, OTHER)])
    state = _state(tmp_path)
    state.tool_chain_active = True
    obs = _obs("put the apple in the bowl")
    assert hook(obs, state) is obs and calls == []


@pytest.mark.parametrize("share, expected", [(0.3, "put the cube on the wooden table"), (0.05, GENERIC_PROMPT)])
def test_generic_keeps_the_subgoal_when_the_destination_mask_is_the_table(monkeypatch, tmp_path, share, expected):
    hook, _ = _hook(monkeypatch, "generic", [(SQUARE, OTHER)])
    drawn = hook._gt

    def master_like(obs, prompt, record):   # the master's hull_of logs the destination mask's share of the front frame
        record["destination"] = {"phrase": "wooden table", "mask_share": share}
        return drawn(obs, prompt, record)

    monkeypatch.setattr(hook, "_gt", master_like)
    out = hook(_obs("put the cube on the wooden table"), _state(tmp_path))
    assert out["prompt"] == expected and REFERENCE_KEY in out
    (line,) = [json.loads(x) for x in open(tmp_path / "reference.jsonl")]
    assert line["vla_prompt"] == expected
