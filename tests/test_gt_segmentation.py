# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ground-truth segmentation providers and grasp tool integration."""

from __future__ import annotations

import numpy as np
import pytest

from vlm_orchestrator.perception.gt_segmentation import (
    DMControlSegProvider,
    GTSegProvider,
    IsaacLabSegProvider,
    RobosuiteSegProvider,
    create_seg_provider,
    gt_object_mask,
)
from vlm_orchestrator.grasp.tool import GraspSegMode, GraspToolExecutor


# ======================================================================
# Mock simulators
# ======================================================================


class MockMujocoModel:
    """Minimal MuJoCo model mock for robosuite / dm_control."""

    def __init__(self, bodies: dict[str, int], geoms: dict[int, int] | None = None):
        """
        bodies: {name: body_id}
        geoms: {geom_id: body_id}
        """
        self._bodies = bodies
        self._id_to_name = {v: k for k, v in bodies.items()}
        self.nbody = max(bodies.values()) + 1 if bodies else 0
        # geom → body mapping
        self._geoms = geoms or {}
        self.ngeom = max(self._geoms.keys()) + 1 if self._geoms else 0
        self.geom_bodyid = np.zeros(max(self.ngeom, 1), dtype=int)
        for gid, bid in self._geoms.items():
            self.geom_bodyid[gid] = bid

    def body_id2name(self, body_id):
        return self._id_to_name.get(body_id, "")

    def body_name2id(self, name):
        if name in self._bodies:
            return self._bodies[name]
        raise KeyError(f"body '{name}' not found")

    def id2name(self, idx, entity_type):
        if entity_type == "body":
            return self._id_to_name.get(idx, "")
        return ""

    def get_joint_qpos_addr(self, name):
        return 0

    def joint_id2name(self, idx):
        return ""

    def site_name2id(self, name):
        return 0

    def site_id2name(self, idx):
        return ""


class MockSim:
    """Minimal robosuite sim mock."""

    def __init__(self, bodies: dict[str, int], seg_image: np.ndarray):
        self.model = MockMujocoModel(bodies)
        self._seg_image = seg_image
        self.data = type("Data", (), {
            "body_xpos": np.zeros((self.model.nbody, 3)),
            "body_xquat": np.zeros((self.model.nbody, 4)),
            "qpos": np.zeros(20),
            "site_xpos": np.zeros((1, 3)),
        })()

    def render(self, camera_name, height, width, segmentation=False):
        if segmentation:
            return self._seg_image
        return np.zeros((height, width, 3), dtype=np.uint8)


class MockRobosuiteEnv:
    """Minimal robosuite env mock."""

    def __init__(self, bodies: dict[str, int], seg_image: np.ndarray):
        self.sim = MockSim(bodies, seg_image)
        self.obj_body_id = bodies


class MockPhysics:
    """Minimal dm_control physics mock."""

    def __init__(
        self,
        bodies: dict[str, int],
        geoms: dict[int, int],
        seg_image: np.ndarray,
    ):
        self.model = MockMujocoModel(bodies, geoms)
        self._seg_image = seg_image
        self.data = type("Data", (), {
            "xpos": np.zeros((self.model.nbody, 3)),
            "xquat": np.zeros((self.model.nbody, 4)),
        })()

    def render(self, height, width, camera_id=0, segmentation=False):
        if segmentation:
            return self._seg_image
        return np.zeros((height, width, 3), dtype=np.uint8)


class MockDMControlEnv:
    """Minimal VLABench env mock."""

    def __init__(self, physics: MockPhysics):
        self.physics = physics


# ======================================================================
# Tests: RobosuiteSegProvider (LIBERO, RoboCasa)
# ======================================================================


class TestRobosuiteSegProvider:
    def _make_env(self, target_body_id: int = 5):
        """Create a mock env with a known segmentation image.

        The real robosuite renderer returns geom IDs in channel 1.
        ``render_mask`` converts geom IDs → body IDs via
        ``model.geom_bodyid``.  We use geom_ids = 10 (→ body 5)
        and 12 (→ body 7) to match the body IDs.
        """
        h, w = 256, 256
        bodies = {
            "red_block": 5,
            "blue_cup": 7,
            "table": 2,
        }
        # geom_id → body_id mapping
        geoms = {
            10: target_body_id,  # red_block geom
            12: 7,               # blue_cup geom
            3: 2,                # table geom
        }

        # Segmentation image: (H, W, 2) — channel 1 = geom_id
        # Place target object in a 50x50 pixel region
        seg = np.zeros((h, w, 2), dtype=np.int32)
        seg[:, :, 1] = -1  # background
        seg[100:150, 80:130, 1] = 10  # red_block geom_id
        seg[50:70, 200:220, 1] = 12   # blue_cup geom_id
        # robosuite renders upside down, so flip
        seg = seg[::-1]

        env = MockRobosuiteEnv(bodies, seg)
        # Set up geom→body mapping on the model
        env.sim.model = MockMujocoModel(bodies, geoms)
        return env

    def test_exact_match(self):
        env = self._make_env(target_body_id=5)
        seg = RobosuiteSegProvider(env, camera_name="agentview", height=256, width=256)

        mask = seg.render_mask("red_block")
        assert mask.shape == (256, 256)
        assert mask.dtype == bool
        assert mask.sum() == 50 * 50  # 50x50 region

    def test_fuzzy_match(self):
        env = self._make_env(target_body_id=5)
        # Add a body with _main suffix
        env.sim.model._bodies["red_block_main"] = 5
        env.sim.model._id_to_name[5] = "red_block_main"
        env.obj_body_id["red_block_main"] = 5

        seg = RobosuiteSegProvider(env, camera_name="agentview", height=256, width=256)
        mask = seg.render_mask("red_block")
        assert mask.sum() == 50 * 50

    def test_object_not_found(self):
        env = self._make_env()
        seg = RobosuiteSegProvider(env, camera_name="agentview", height=256, width=256)

        with pytest.raises(ValueError, match="not found"):
            seg.render_mask("nonexistent_object")

    def test_available_objects(self):
        env = self._make_env()
        seg = RobosuiteSegProvider(env, camera_name="agentview", height=256, width=256)
        objects = seg.available_objects()
        assert "red_block" in objects
        assert "blue_cup" in objects

    def test_render_mask_for_obs(self):
        env = self._make_env()
        seg = RobosuiteSegProvider(env, camera_name="agentview", height=256, width=256)

        wire_obs = {"observation/image": np.zeros((256, 256, 3))}
        seg.render_mask_for_obs("red_block", wire_obs)

        assert "gt_seg/mask" in wire_obs
        assert "gt_seg/target" in wire_obs
        assert wire_obs["gt_seg/target"] == "red_block"
        assert wire_obs["gt_seg/mask"].sum() == 50 * 50


# ======================================================================
# Tests: DMControlSegProvider (VLABench)
# ======================================================================


class TestDMControlSegProvider:
    def _make_env(self):
        h, w = 256, 256
        bodies = {"target_ball": 3, "container": 4, "world": 0}
        # geom 10, 11 belong to body 3 (target_ball)
        # geom 20 belongs to body 4 (container)
        geoms = {10: 3, 11: 3, 20: 4}

        # Segmentation: channel 0 = geom_id
        seg = np.full((h, w, 2), -1, dtype=np.int32)
        seg[60:100, 60:100, 0] = 10   # geom 10 → body 3
        seg[100:120, 60:80, 0] = 11   # geom 11 → body 3
        seg[150:200, 150:200, 0] = 20  # geom 20 → body 4

        physics = MockPhysics(bodies, geoms, seg)
        return MockDMControlEnv(physics)

    def test_multi_geom_body(self):
        env = self._make_env()
        seg = DMControlSegProvider(env, camera_id=2, height=256, width=256)

        mask = seg.render_mask("target_ball")
        assert mask.shape == (256, 256)
        # Two geom regions: 40*40 + 20*20 = 1600 + 400 = 2000
        assert mask.sum() == 2000

    def test_container(self):
        env = self._make_env()
        seg = DMControlSegProvider(env, camera_id=2, height=256, width=256)

        mask = seg.render_mask("container")
        assert mask.sum() == 50 * 50  # 50x50 region

    def test_fuzzy_match(self):
        env = self._make_env()
        seg = DMControlSegProvider(env, camera_id=2, height=256, width=256)
        # "ball" should fuzzy-match "target_ball"
        mask = seg.render_mask("ball")
        assert mask.sum() == 2000


# ======================================================================
# Tests: Factory
# ======================================================================


class TestFactory:
    def test_create_libero(self):
        env = MockRobosuiteEnv({"obj": 1}, np.zeros((256, 256, 2), dtype=np.int32))
        provider = create_seg_provider(env, "libero", camera_name="agentview")
        assert isinstance(provider, RobosuiteSegProvider)

    def test_create_robocasa(self):
        env = MockRobosuiteEnv({"obj": 1}, np.zeros((256, 256, 2), dtype=np.int32))
        provider = create_seg_provider(env, "robocasa", camera_name="agentview")
        assert isinstance(provider, RobosuiteSegProvider)

    def test_create_vlabench(self):
        physics = MockPhysics({"obj": 1}, {}, np.zeros((256, 256, 2), dtype=np.int32))
        env = MockDMControlEnv(physics)
        provider = create_seg_provider(env, "vlabench", camera_id=2)
        assert isinstance(provider, DMControlSegProvider)

    def test_unknown_env(self):
        with pytest.raises(ValueError, match="Unknown env_type"):
            create_seg_provider(None, "unknown_sim")


# ======================================================================
# Tests: GraspTool GT_SIM integration
# ======================================================================


class TestGraspToolGTSIM:
    def test_seg_mode_enum(self):
        assert GraspSegMode.GT_SIM == "gt_sim"
        assert GraspSegMode("gt_sim") == GraspSegMode.GT_SIM

    @staticmethod
    def _obs():
        """LIBERO-style id image: -1 background, two objects and the table."""
        body_ids = np.full((256, 256), -1, dtype=np.int32)
        body_ids[40:80, 60:120] = 5     # red_block
        body_ids[150:180, 100:140] = 7  # blue_cup
        body_ids[200:256, :] = 2        # table
        return {
            "gt_seg/body_ids": body_ids,
            "gt_seg/obj_body_id": {"red_block": 5, "blue_cup": 7, "table": 2},
        }

    def test_mask_is_exactly_the_named_object(self):
        obs = self._obs()
        mask = gt_object_mask(obs, "blue_cup", front=False)
        assert np.array_equal(mask, obs["gt_seg/body_ids"] == 7)

    def test_missing_id_image_raises(self):
        with pytest.raises(RuntimeError, match="gt_sim needs"):
            gt_object_mask({}, "red_block", front=False)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="not a scene object"):
            gt_object_mask(self._obs(), "red", front=False)

    def test_mask_to_bbox(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:60] = True
        bbox = GraspToolExecutor._mask_to_bbox(mask)
        assert bbox == (30, 20, 60, 40)

    def test_mask_to_bbox_empty(self):
        mask = np.zeros((100, 100), dtype=bool)
        bbox = GraspToolExecutor._mask_to_bbox(mask)
        assert bbox == (0, 0, 0, 0)

    def test_scene_object_name_is_taken_as_is(self):
        executor = GraspToolExecutor(seg_mode=GraspSegMode.GT_SIM, env_mode="libero")
        assert executor._resolve_gt_target(self._obs(), "red_block") == "red_block"

    def test_phrase_resolves_through_the_masters_point(self):
        executor = GraspToolExecutor(seg_mode=GraspSegMode.GT_SIM, env_mode="libero")
        obs = self._obs()
        with pytest.raises(RuntimeError, match="gave no point"):
            executor._resolve_gt_target(obs, "the blue cup")
        executor._target_point = (120 / 256, 165 / 256)  # on the cup
        assert executor._resolve_gt_target(obs, "the blue cup") == "blue_cup"
        executor._target_point = (125 / 256, 60 / 256)  # a few px beside the block
        assert executor._resolve_gt_target(obs, "red block") == "red_block"
        executor._target_point = (10 / 256, 10 / 256)  # far from every object
        with pytest.raises(RuntimeError, match="not on a scene object"):
            executor._resolve_gt_target(obs, "red block")

    def test_grasp_verdict_needs_held_and_lifted(self):
        executor = GraspToolExecutor(seg_mode=GraspSegMode.GT_SIM, env_mode="libero")

        def obs(z, contacts):
            return {"gt_state": {
                "objects": {"red_block": {"pos": [0.3, 0.0, z]}},
                "robot": {"objects_in_contact_raw": contacts},
            }}

        assert executor._judge_grasp(obs(0.15, ["red_block"])) is None  # no target resolved
        executor._gt_name, executor._target_z0 = "red_block", 0.10
        assert executor._judge_grasp(obs(0.15, ["red_block"]))[0] is True
        assert executor._judge_grasp(obs(0.11, ["red_block"]))[0] is False
        assert executor._judge_grasp(obs(0.15, []))[0] is False
