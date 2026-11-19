# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unpaused sim: latency table, master calls held until their due step, release and abort."""

from __future__ import annotations

import asyncio
import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_orchestrator.nopause import GT_RESOLVER_PROMPT, EpisodeEnded, Gate, LatencyTable, hold_chunk
from vlm_orchestrator.vlm import api
from vlm_orchestrator.vlm.api import chat_create, set_call_hook, set_call_log_dir

HZ = 15.0
PLAN = "You are a vision-language assistant helping a robot arm plan"
MONITOR = "You are monitoring a robot arm"
RECYCLE = "You check whether a robot finished"
REPLY = SimpleNamespace(
    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
    usage=None,
)
CLIENT = SimpleNamespace(
    chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: REPLY)),
)
ROW_FIELDS = {
    "call_index", "kind", "call_step", "latency_s",
    "delay_steps", "due_step", "release_step",
}


def master(system_prompt: str):
    return chat_create(CLIENT, model="qwen3.5-9b", messages=[
        {"role": "system", "content": system_prompt + " tabletop tasks."},
        {"role": "user", "content": "now"},
    ])


def rows(log_dir):
    return [json.loads(line) for line in (log_dir / "nopause.jsonl").read_text().splitlines()]


@pytest.fixture(autouse=True)
def isolated_api(monkeypatch):
    monkeypatch.setattr(api, "_call_log_dir", None)
    monkeypatch.setattr(api, "_call_hook", None)


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "latency.json"
    path.write_text(json.dumps({
        "kinds": {"plan": PLAN, "monitor": MONITOR, "recycle": RECYCLE},
        "latency_s": {"plan": [2.31, 3.07, 4.5], "monitor": [1.2, 1.9], "recycle": [0.73]},
        "source": "ignored",
    }))
    return LatencyTable(path)


def test_kind_names_each_prefix_and_rejects_others(table):
    assert table.kind(PLAN + " tabletop tasks.") == "plan"
    assert table.kind(MONITOR + "'s progress on a manipulation task.") == "monitor"
    assert table.kind(RECYCLE + " a pick-and-place task") == "recycle"
    with pytest.raises(ValueError):
        table.kind("Robot progress checker. Decide if the current step is done.")


def test_draw_is_reproducible_and_leaves_global_rngs_alone(table):
    py_state, np_state = random.getstate(), np.random.get_state()
    keys = [f"put the apple in the bowl|3|{i}" for i in range(20)]
    draws = [table.draw("plan", key) for key in keys]
    assert draws == [table.draw("plan", key) for key in keys]
    assert set(draws) <= set(table.latency_s["plan"])
    assert random.getstate() == py_state
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), np_state))


def test_without_hook_reply_is_returned_unchanged():
    assert master(PLAN) is REPLY


def test_held_call_is_released_on_the_new_frame(table, tmp_path):
    def process(obs, state):
        master(PLAN)
        return obs, state

    async def main():
        gate = Gate(table, HZ, asyncio.get_running_loop())
        set_call_hook(gate.hook)
        set_call_log_dir(tmp_path)
        gate.start_episode("put the apple in the bowl|0")
        state = SimpleNamespace(episode_step=16, episode_log_dir=str(tmp_path))

        assert await gate.step(process, {"frame": 2}, state) is None
        delay = math.ceil(table.draw("plan", "put the apple in the bowl|0|0") * HZ)
        assert gate.due == 16 + delay
        assert len((tmp_path / "vlm_calls.jsonl").read_text().splitlines()) == 1

        state.episode_step = gate.due - 1
        assert await gate.step(process, {"frame": 3}, state) is None

        state.episode_step = gate.due + 5
        obs, returned = await gate.step(process, {"frame": 4}, state)
        assert obs == {"frame": 4} and returned is state
        assert gate.due is None
        return delay, state.episode_step

    delay, release_step = asyncio.run(main())
    [row] = rows(tmp_path)
    assert set(row) == ROW_FIELDS
    assert row["kind"] == "plan" and row["call_index"] == 0
    assert row["call_step"] == 16 and row["delay_steps"] == delay
    assert row["due_step"] == 16 + delay and row["release_step"] == release_step
    assert row["latency_s"] == table.draw("plan", "put the apple in the bowl|0|0")


def test_second_call_is_held_from_the_release_step(table, tmp_path):
    def process(obs, state):
        master(MONITOR)
        master(RECYCLE)
        return obs, state

    async def main():
        gate = Gate(table, HZ, asyncio.get_running_loop())
        set_call_hook(gate.hook)
        gate.start_episode("stack the cubes|1")
        state = SimpleNamespace(episode_step=40, episode_log_dir=str(tmp_path))

        assert await gate.step(process, {"frame": 0}, state) is None
        state.episode_step = gate.due + 3
        assert await gate.step(process, {"frame": 1}, state) is None
        recycle_delay = math.ceil(table.draw("recycle", "stack the cubes|1|1") * HZ)
        assert gate.due == state.episode_step + recycle_delay

        state.episode_step = gate.due
        obs, _ = await gate.step(process, {"frame": 2}, state)
        assert obs == {"frame": 2}

    asyncio.run(main())
    monitor, recycle = rows(tmp_path)
    assert (monitor["kind"], recycle["kind"]) == ("monitor", "recycle")
    assert (monitor["call_index"], recycle["call_index"]) == (0, 1)
    assert recycle["call_step"] == monitor["release_step"]


def reply(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=None,
    )


def run_one_call(table, tmp_path, monkeypatch, system_prompt, content, instant_next):
    """One master call at step 40: (whether process() returned on that step, the logged row)."""
    monkeypatch.setattr(CLIENT.chat.completions, "create", lambda **kwargs: reply(content))

    def process(obs, state):
        master(system_prompt)
        return obs, state

    async def main():
        gate = Gate(table, HZ, asyncio.get_running_loop(), instant_next=instant_next)
        set_call_hook(gate.hook)
        gate.start_episode("stack the cubes|0")
        state = SimpleNamespace(episode_step=40, episode_log_dir=str(tmp_path))
        first = await gate.step(process, {"frame": 1}, state)
        if first is None:
            state.episode_step = gate.due
            assert await gate.step(process, {"frame": 2}, state) is not None
        assert gate.due is None
        return first is not None

    same_step = asyncio.run(main())
    [row] = rows(tmp_path)
    assert set(row) == ROW_FIELDS
    return same_step, row


def test_instant_next_lets_a_monitor_next_reply_through_on_its_call_step(table, tmp_path, monkeypatch):
    same_step, row = run_one_call(
        table, tmp_path, monkeypatch, MONITOR,
        '{"status": "complete", "action": "next", "reason": "placed"}', True,
    )
    assert same_step
    assert row["latency_s"] == table.draw("monitor", "stack the cubes|0|0")
    assert (row["delay_steps"], row["due_step"], row["release_step"]) == (0, 40, 40)


@pytest.mark.parametrize("system_prompt, content, instant_next", [
    (MONITOR, '{"action": "next"}', False),
    (MONITOR, '{"status": "in_progress", "action": "continue"}', True),
    (MONITOR, "not json, next", True),
    (PLAN, '{"action": "next"}', True),
    (RECYCLE, '{"action": "next"}', True),
])
def test_every_other_reply_is_held_for_its_table_latency(
    table, tmp_path, monkeypatch, system_prompt, content, instant_next,
):
    same_step, row = run_one_call(table, tmp_path, monkeypatch, system_prompt, content, instant_next)
    kind = table.kind(system_prompt + " tabletop tasks.")
    delay = math.ceil(table.draw(kind, "stack the cubes|0|0") * HZ)
    assert not same_step
    assert (row["delay_steps"], row["due_step"], row["release_step"]) == (delay, 40 + delay, 40 + delay)


def test_abort_discards_the_reply_and_ends_process(table, tmp_path):
    ended = []

    def process(obs, state):
        try:
            master(PLAN)
        except EpisodeEnded:
            ended.append(True)
            raise
        return obs, state

    async def main():
        gate = Gate(table, HZ, asyncio.get_running_loop())
        set_call_hook(gate.hook)
        gate.start_episode("open the drawer|0")
        state = SimpleNamespace(episode_step=8, episode_log_dir=str(tmp_path))

        assert await gate.step(process, {"frame": 1}, state) is None
        await gate.abort()
        assert gate.due is None

        gate.start_episode("open the drawer|1")
        assert await gate.step(lambda o, s: (o, s), {"frame": 9}, state) == ({"frame": 9}, state)

    asyncio.run(main())
    assert ended == [True]
    [row] = rows(tmp_path)
    assert set(row) == ROW_FIELDS | {"discarded"}
    assert row["release_step"] is None and row["discarded"] is True


def test_master_call_outside_process_fails_the_next_step(table, tmp_path):
    async def main():
        gate = Gate(table, HZ, asyncio.get_running_loop())
        set_call_hook(gate.hook)
        with pytest.raises(RuntimeError):
            master(PLAN)
        state = SimpleNamespace(episode_step=0, episode_log_dir=str(tmp_path))
        with pytest.raises(RuntimeError):
            await gate.step(lambda o, s: (o, s), {}, state)

    asyncio.run(main())


def test_gt_resolver_call_outside_process_is_not_held(table, tmp_path):
    async def main():
        gate = Gate(table, HZ, asyncio.get_running_loop())
        set_call_hook(gate.hook)
        reply = chat_create(CLIENT, model="qwen3.5-9b", messages=[
            {"role": "system", "content": GT_RESOLVER_PROMPT}, {"role": "user", "content": "x"}])
        assert reply is REPLY and gate.fault is None and gate.call_index == 0
        state = SimpleNamespace(episode_step=0, episode_log_dir=str(tmp_path))
        assert await gate.step(lambda o, s: (o, s), {}, state) == ({}, state)

    asyncio.run(main())


def test_unknown_kind_swallowed_by_the_strategy_still_fails_the_step(table, tmp_path):
    def process(obs, state):
        try:
            master("Robot progress checker.")
        except Exception:
            pass
        return obs, state

    async def main():
        gate = Gate(table, HZ, asyncio.get_running_loop())
        set_call_hook(gate.hook)
        state = SimpleNamespace(episode_step=0, episode_log_dir=str(tmp_path))
        with pytest.raises(ValueError):
            await gate.step(process, {}, state)

    asyncio.run(main())


def test_hold_chunk_keeps_the_joints_with_the_gripper_open():
    joints = np.arange(7) * 0.1
    chunk = hold_chunk({"observation/joint_position": joints, "observation/gripper_position": [1.0]})
    assert chunk.shape == (8, 8)
    assert (chunk[:, :7] == joints).all() and (chunk[:, 7] == 0.0).all()
