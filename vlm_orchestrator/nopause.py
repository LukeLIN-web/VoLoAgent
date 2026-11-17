# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unpaused simulation: the sim keeps stepping while the master is called.

The strategy runs in a worker thread.  :meth:`Gate.hook`, installed as the
``chat_create`` call hook, holds each master reply until the proxy's sim clock
reaches the step the reply is due (the call's step plus its latency, drawn from
a measured table, in sim steps); until then the proxy keeps serving action
chunks without the strategy.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import zlib

import numpy as np

from vlm_orchestrator.gt_metrics.target_resolver import _SYSTEM_PROMPT as GT_RESOLVER_PROMPT


class LatencyTable:
    """Measured master latencies per call kind; a call's kind is named by the
    prefix its system prompt starts with."""

    def __init__(self, path):
        with open(path) as f:
            table = json.load(f)
        self.kinds = table["kinds"]
        self.latency_s = table["latency_s"]

    def kind(self, system_prompt: str) -> str:
        matches = [k for k, prefix in self.kinds.items()
                   if system_prompt.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(
                f"system prompt matches latency kinds {matches}: "
                f"{system_prompt[:80]!r}"
            )
        return matches[0]

    def draw(self, kind: str, key: str) -> float:
        return random.Random(zlib.crc32(key.encode())).choice(
            self.latency_s[kind]
        )


class EpisodeEnded(BaseException):
    """Raised in a held master call whose episode ended before its reply was due."""


def hold_chunk(obs: dict) -> np.ndarray:
    """Eight steps holding the arm's joints with the gripper open."""
    return np.tile(np.r_[obs["observation/joint_position"][:7], 0.0], (8, 1))


class Gate:
    """Runs ``strategy.process`` in a worker thread that :meth:`hook` suspends
    after every master reply until :meth:`step` sees the reply's due step.

    A failure inside the hook is also kept in ``fault`` and raised by the next
    :meth:`step`, since strategies swallow exceptions around master calls.
    """

    def __init__(self, table: LatencyTable, hz: float, loop):
        self.table = table
        self.hz = hz
        self.loop = loop
        self.due: int | None = None
        self.fault: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._obs: dict | None = None
        self._state = None
        self._signal = None
        self._release = threading.Event()
        self._ended = False
        self.start_episode("")

    def start_episode(self, key: str) -> None:
        self.episode_key = key
        self.call_index = 0
        self.prompt_in_force: str | None = None

    async def step(self, process, obs: dict, state):
        """``process(obs, state)``'s result, or None while a reply is in flight
        (``due`` then names its step)."""
        if self.fault is not None:
            raise self.fault
        if self.due is None:
            self._obs, self._state = dict(obs), state
            self._thread = threading.Thread(
                target=self._work, args=(process,), daemon=True,
            )
            return await self._settle(self._thread.start)
        if state.episode_step < self.due:
            return None
        self._obs.clear()
        self._obs.update(obs)
        return await self._settle(self._release.set)

    async def abort(self) -> None:
        """Drop the reply in flight: its call raises EpisodeEnded and process() ends."""
        if self.due is None:
            return
        self._ended = True
        await self._settle(self._release.set)
        self._ended = False

    async def _settle(self, go):
        self._signal = self.loop.create_future()
        go()
        result = await self._signal
        if self.fault is not None:
            raise self.fault
        return result

    def _work(self, process) -> None:
        try:
            result = process(self._obs, self._state)
        except EpisodeEnded:
            result = None
        except BaseException as e:
            self.loop.call_soon_threadsafe(self._signal.set_exception, e)
            return
        self.loop.call_soon_threadsafe(self._signal.set_result, result)

    def hook(self, messages) -> None:
        if messages[0]["content"] == GT_RESOLVER_PROMPT:
            return  # the GT metrics' resolver feeds no action, so its reply is not delayed
        try:
            self._hold(messages)
        except EpisodeEnded:
            raise
        except BaseException as e:
            self.fault = e
            raise

    def _hold(self, messages) -> None:
        if threading.current_thread() is not self._thread:
            raise RuntimeError("master call outside the gated strategy.process()")
        kind = self.table.kind(messages[0]["content"])
        latency = self.table.draw(kind, f"{self.episode_key}|{self.call_index}")
        delay = math.ceil(latency * self.hz)
        call_step = self._state.episode_step
        row = {
            "call_index": self.call_index, "kind": kind,
            "call_step": call_step, "latency_s": latency,
            "delay_steps": delay, "due_step": call_step + delay,
        }
        self.call_index += 1
        log_path = os.path.join(self._state.episode_log_dir, "nopause.jsonl")

        self._release.clear()
        self.due = row["due_step"]
        self.loop.call_soon_threadsafe(self._signal.set_result, None)
        self._release.wait()
        self.due = None

        if self._ended:
            row.update(release_step=None, discarded=True)
        else:
            row["release_step"] = self._state.episode_step
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if self._ended:
            raise EpisodeEnded
