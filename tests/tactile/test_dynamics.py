"""Verify the native visual/pose bridge with standalone USD and a fake solver."""

from types import SimpleNamespace
import unittest
import sys
from unittest.mock import patch

import numpy as np
from transforms3d.quaternions import quat2mat

from env.tactile.dynamics import ArticulationRenderBridge, UipcSceneBody, _native_actor_view, _restore_attribute_opinion

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


class SolverBody:
    def __init__(self, position):
        self.matrix = np.eye(4)[None]
        self.matrix[0, :3, 3] = position
        self.geo_slot_list = [
            SimpleNamespace(
                geometry=lambda: SimpleNamespace(transforms=lambda: SimpleNamespace(view=lambda: self.matrix))
            )
        ]
        self.velocity_clears = 0

    def write_pose_to_sim(self, pose, *, zero_velocity):
        pose = pose.detach().cpu().numpy()[0]
        self.matrix[0, :3, 3] = pose[:3]
        self.matrix[0, :3, :3] = quat2mat(pose[3:])
        if zero_velocity:
            self.velocity_clears += 1


class NativeActorView:
    def __init__(self, paths, poses, *, stage):
        self.prim_paths = paths
        self.poses = np.asarray(poses, dtype=np.float32)
        self.velocities = np.zeros((len(paths), 6), dtype=np.float32)
        self.writes = []
        self.stage = stage
        self.kinematic = self._kinematic_flags()
        self.invalid_velocity_writes = []
        self.velocity_write_failures = 0
        self.velocity_write_indices = []

    def _kinematic_flags(self):
        flags = []
        for path in self.prim_paths:
            prim = self.stage.GetPrimAtPath(path)
            flags.append(bool(prim and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                              and UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()))
        return np.asarray(flags)

    def _refresh_kinematic_state(self):
        flags = self._kinematic_flags()
        # PhysX's dynamic-to-kinematic transition sleeps and clears velocity.
        self.velocities[flags & ~self.kinematic] = 0
        self.kinematic = flags

    def get_transforms(self):
        self._refresh_kinematic_state()
        return self.poses.copy()

    def get_velocities(self):
        self._refresh_kinematic_state()
        return self.velocities.copy()

    def set_transforms(self, poses, indices):
        self._refresh_kinematic_state()
        self.poses[np.asarray(indices)] = poses
        self.writes.append(np.asarray(poses).copy())

    def set_velocities(self, velocities, indices):
        self._refresh_kinematic_state()
        indices = np.asarray(indices)
        if np.any(self.kinematic[np.asarray(indices)]):
            # The installed backend logs the rejected PxRigidDynamic setters
            # without necessarily raising in Python. Model that no-op too.
            self.invalid_velocity_writes.append(np.asarray(indices).copy())
            return
        if self.velocity_write_failures:
            self.velocity_write_failures -= 1
            raise RuntimeError("native velocity write failed")
        velocities = np.asarray(velocities)
        assert velocities.shape == self.velocities.shape  # full-view buffer
        self.velocity_write_indices.append(indices.copy())
        self.velocities[indices] = velocities[indices]


@unittest.skipIf(Usd is None, "Standalone USD bindings are not on PYTHONPATH")
class ArticulationRenderBridgeTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        base = UsdGeom.Xform.Define(self.stage, "/World/envs/env_0/robot0")
        base.AddTranslateOp().Set(Gf.Vec3d(0, -.55, .765))
        base.AddRotateZOp().Set(90.)
        self.paths = {
            "hand": str(base.GetPath()) + "/hand",
            "finger": str(base.GetPath()) + "/hand/finger",
            "case": str(base.GetPath()) + "/hand/finger/case",
            "WristCamera": str(base.GetPath()) + "/WristCamera",
        }
        for name, path in self.paths.items():
            body = UsdGeom.Xform.Define(self.stage, path)
            body.AddTranslateOp().Set(Gf.Vec3d(.02, .03, .04))
            if name == "case":
                body.SetResetXformStack(True)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        # A link without a collider must still carry its optical descendants.
        wrist = UsdGeom.Xform.Define(self.stage, self.paths["WristCamera"] + "/optics")
        wrist.AddTranslateOp().Set(Gf.Vec3d(.01, 0, .02))
        self.camera = UsdGeom.Camera.Define(self.stage, str(wrist.GetPath()) + "/Camera")
        self.body_names = ["WristCamera", "case", "finger", "hand"]
        self.poses = np.array([[[.1 + .02 * i, .2 + .01 * i, 1.0, 0, 0, .38268343, .92387953]
                               for i in range(len(self.body_names))]])
        import torch

        self.robot = SimpleNamespace(
            cfg=SimpleNamespace(prim_path="{ENV_REGEX_NS}/robot0"), body_names=self.body_names,
            root_physx_view=SimpleNamespace(get_link_transforms=lambda: torch.tensor(self.poses)),
        )
        self.bridge = ArticulationRenderBridge(self.robot, self.stage)

    def test_all_links_and_camera_follow_same_measured_pose_across_updates(self):
        from transforms3d.quaternions import quat2mat

        def pose_matrix(pose):
            result = np.eye(4)
            result[:3, 3] = pose[:3]
            result[:3, :3] = quat2mat([pose[6], *pose[3:6]])
            return result

        original_camera_local = np.asarray(UsdGeom.Xformable(self.camera.GetPrim().GetParent()).GetLocalTransformation()).T
        original_body_flags = {name: self.stage.GetPrimAtPath(path).GetAppliedSchemas() for name, path in self.paths.items()}
        for move in (np.zeros(3), np.array([-.05, .03, -.02])):
            self.poses[:, :, :3] += move
            with patch("env.tactile.dynamics._sync_fabric_link_transforms") as fabric:
                self.bridge.sync()
            worlds = fabric.call_args.args[0]
            for index, name in enumerate(self.body_names):
                expected = pose_matrix(self.poses[0, index])
                prim = self.stage.GetPrimAtPath(self.paths[name])
                actual = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                np.testing.assert_allclose(actual, expected, atol=1e-7)
                np.testing.assert_allclose(worlds[self.paths[name]], expected, atol=1e-7)
                self.assertEqual(prim.GetAppliedSchemas(), original_body_flags[name])
            expected_camera = pose_matrix(self.poses[0, 0]) @ original_camera_local
            actual_camera = np.asarray(self.camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            np.testing.assert_allclose(actual_camera, expected_camera, atol=1e-7)
            self.assertTrue(UsdGeom.Xformable(self.stage.GetPrimAtPath(self.paths["case"])).GetResetXformStack())
        self.assertFalse(self.stage.GetPrimAtPath(self.paths["WristCamera"]).HasAPI(UsdPhysics.CollisionAPI))

    def test_fabric_hierarchy_receives_all_links_before_one_descendant_update(self):
        from env.tactile.dynamics import _sync_fabric_link_transforms
        from types import ModuleType

        events = []
        hierarchy = SimpleNamespace(
            set_world_xform=lambda path, matrix: events.append((str(path), np.asarray(matrix).T)),
            update_world_xforms=lambda: events.append("update"),
        )
        stage = SimpleNamespace(GetFabricId=lambda: 7, GetStageIdAsStageId=lambda: 11)
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: SimpleNamespace(get_stage_id=lambda: 11)

        def matrix4d(*values):
            # USDRT's binding accepts 16 scalars, unlike pxr.Gf's nested-list
            # constructor. A permissive numpy stand-in missed that mismatch.
            if len(values) != 16 or not all(isinstance(value, (int, float)) for value in values):
                raise TypeError("Matrix4d requires 16 numeric scalar arguments")
            return np.asarray(values, dtype=float).reshape(4, 4)

        usdrt = SimpleNamespace(
            Usd=SimpleNamespace(Stage=SimpleNamespace(Attach=lambda _: stage)),
            Sdf=SimpleNamespace(Path=str), Gf=SimpleNamespace(Matrix4d=matrix4d),
            hierarchy=SimpleNamespace(IFabricHierarchy=lambda: SimpleNamespace(get_fabric_hierarchy=lambda *_: hierarchy)),
        )
        worlds = {}
        for index, path in enumerate(self.paths.values()):
            world = np.eye(4)
            world[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            world[:3, 3] = [.12 + .03 * index, -.23 + .02 * index, 1.04 - .01 * index]
            worlds[path] = world
        with self.assertRaises(TypeError):
            matrix4d(np.eye(4).tolist())
        with patch.dict(sys.modules, {"omni.usd": usd, "usdrt": usdrt}):
            _sync_fabric_link_transforms(worlds)
        self.assertEqual([event[0] for event in events[:-1]], list(worlds))
        for path, actual in events[:-1]:
            np.testing.assert_array_equal(actual, worlds[path])
        self.assertEqual(events[-1], "update")


@unittest.skipIf(Usd is None, "Standalone USD bindings are not on PYTHONPATH")
class NativePoseBridgeTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        parent = UsdGeom.Xform.Define(self.stage, "/World")
        parent.AddTranslateOp().Set(Gf.Vec3d(0.3, 0.1, 0))
        self.root = UsdGeom.Xform.Define(self.stage, "/World/object")
        self.root.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.2, 0.7))
        self.root.AddScaleOp().Set(Gf.Vec3d(2, 3, 4))
        self.physics = UsdPhysics.RigidBodyAPI.Apply(self.root.GetPrim())
        self.physics.CreateRigidBodyEnabledAttr(True)
        collider = UsdGeom.Cube.Define(self.stage, "/World/object/collision")
        self.collision = UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        self.collision.CreateCollisionEnabledAttr(True)
        self.wrapper = SimpleNamespace(stage=self.stage)
        self.wrapper.set_dynamics_backend = lambda value: setattr(self.wrapper, "backend", value)
        self.binding = SimpleNamespace(scene_object=SimpleNamespace(wrapper=self.wrapper, prim_path="/World/object"),
                                       initial_pose=(.4, .3, .7, 1, 0, 0, 0))
        self.body = SolverBody([0.4, 0.3, 0.7])
        self.native = NativeActorView(["/World/object"], [[.4, .3, .7, 0, 0, 0, 1]], stage=self.stage)
        self.simulation = SimpleNamespace(is_valid=True)
        self.bridge = self.make_bridge()

    def make_bridge(self):
        with patch("env.tactile.dynamics._native_actor_view", return_value=(self.simulation, self.native)):
            return UipcSceneBody(self.binding, self.body, SimpleNamespace(retrieve=lambda: None), "cpu")

    def tearDown(self):
        self.bridge.close()

    def test_initial_correction_preserves_native_actor_offsets_and_close_state(self):
        self.bridge.close()
        child = UsdGeom.Xform.Define(self.stage, "/World/object/button")
        child.AddTranslateOp().Set(Gf.Vec3d(.44, .3, .7))
        child.SetResetXformStack(True)
        UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
        initial = [[.44, .3, .7, 0, 0, 0, 1], [.4, .3, .7, 0, 0, 0, 1]]
        self.native = NativeActorView(["/World/object/button", "/World/object"], initial, stage=self.stage)
        self.native.velocities[:] = [.1, .2, .3, .4, .5, .6]
        original_velocities = self.native.velocities.copy()
        self.body.matrix[0, :3, 3] += [.0002, 0, .00004]
        self.bridge = self.make_bridge()
        np.testing.assert_allclose(self.native.poses[1, :3], [.4002, .3, .70004], atol=1e-7)
        np.testing.assert_allclose(self.native.poses[0, :3], [.4402, .3, .70004], atol=1e-7)
        np.testing.assert_array_equal(self.native.velocities, np.zeros_like(original_velocities))
        self.bridge.close()
        np.testing.assert_allclose(self.native.poses, initial, atol=1e-7)
        np.testing.assert_array_equal(self.native.velocities, original_velocities)

    def test_effective_pose_does_not_hide_native_preparation_mismatch(self):
        self.bridge.close()
        self.body.matrix[0, :3, 3] = [.401, .3, .7]
        self.native.poses[0, :3] = [.401, .3, .7]
        with self.assertRaisesRegex(RuntimeError, "Native/UIPC root pose mismatch"):
            self.make_bridge()

    def test_handoff_uses_kinematic_transition_and_close_restores_velocity(self):
        self.assertEqual(self.native.invalid_velocity_writes, [])
        self.bridge.close()
        self.native.velocities[:] = [.1, .2, .3, .4, .5, .6]
        original = self.native.velocities.copy()
        self.bridge = self.make_bridge()
        self.assertTrue(self.physics.GetKinematicEnabledAttr().Get())
        np.testing.assert_array_equal(self.native.get_velocities(), np.zeros_like(original))
        self.assertEqual(self.native.invalid_velocity_writes, [])
        self.bridge.close()
        self.bridge.close()
        np.testing.assert_array_equal(self.native.get_velocities(), original)
        self.assertFalse(self.physics.GetKinematicEnabledAttr().Get())
        self.assertEqual(self.native.invalid_velocity_writes, [])

    def test_close_skips_original_kinematic_root(self):
        self.bridge.close()
        self.physics.CreateKinematicEnabledAttr(True)
        self.native = NativeActorView(["/World/object"], [[.4, .3, .7, 0, 0, 0, 1]], stage=self.stage)
        original = self.native.poses.copy()
        self.bridge = self.make_bridge()
        # Focus this regression on close, independently of the old ctor error.
        self.native.invalid_velocity_writes.clear()
        self.bridge.close()
        self.bridge.close()
        self.assertTrue(self.physics.GetKinematicEnabledAttr().Get())
        np.testing.assert_array_equal(self.native.poses, original)
        self.assertEqual(self.native.velocity_write_indices, [])
        self.assertEqual(self.native.invalid_velocity_writes, [])

    def test_close_skips_kinematic_child_and_retries_dynamic_velocity_failure(self):
        self.bridge.close()
        child = UsdGeom.Xform.Define(self.stage, "/World/object/button")
        child.AddTranslateOp().Set(Gf.Vec3d(.44, .3, .7))
        child.SetResetXformStack(True)
        child_physics = UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
        child_physics.CreateKinematicEnabledAttr(True)
        initial = [[.44, .3, .7, 0, 0, 0, 1], [.4, .3, .7, 0, 0, 0, 1]]
        self.native = NativeActorView(["/World/object/button", "/World/object"], initial, stage=self.stage)
        self.native.velocities[1] = [.1, .2, .3, .4, .5, .6]
        original = self.native.velocities.copy()
        self.bridge = self.make_bridge()
        self.native.invalid_velocity_writes.clear()
        self.native.velocity_write_failures = 1
        close_error = None
        try:
            self.bridge.close()
        except RuntimeError as error:
            close_error = str(error)
        self.assertEqual(self.native.invalid_velocity_writes, [])
        self.assertEqual(close_error, "native velocity write failed")
        self.assertFalse(self.bridge._closed)
        self.bridge.close()
        self.bridge.close()
        self.assertTrue(child_physics.GetKinematicEnabledAttr().Get())
        self.assertFalse(self.physics.GetKinematicEnabledAttr().Get())
        np.testing.assert_allclose(self.native.poses, initial, atol=1e-7)
        np.testing.assert_array_equal(self.native.get_velocities(), original)
        self.assertEqual(len(self.native.velocity_write_indices), 1)
        np.testing.assert_array_equal(self.native.velocity_write_indices[0], [1])
        self.assertEqual(self.native.invalid_velocity_writes, [])

    def test_close_restores_sdf_collisions_before_dynamic_flags_and_velocities(self):
        self.bridge.close()
        meshes = [UsdGeom.Mesh.Define(self.stage, "/World/object/sdf_mesh")]
        child = UsdGeom.Xform.Define(self.stage, "/World/object/button")
        child.AddTranslateOp().Set(Gf.Vec3d(.44, .3, .7))
        child.SetResetXformStack(True)
        child_physics = UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
        child_physics.CreateKinematicEnabledAttr(False)
        meshes.append(UsdGeom.Mesh.Define(self.stage, "/World/object/button/sdf_mesh"))
        for mesh in meshes:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("sdf")
        initial = [[.44, .3, .7, 0, 0, 0, 1], [.4, .3, .7, 0, 0, 0, 1]]
        self.native = NativeActorView(["/World/object/button", "/World/object"], initial, stage=self.stage)
        original = self.stage.GetRootLayer().ExportToString()
        self.bridge = self.make_bridge()
        events, illegal = [], []

        def restore(saved):
            attribute = saved[0]
            before = attribute.Get()
            _restore_attribute_opinion(saved)
            if attribute.GetName() == "physics:collisionEnabled":
                owner = attribute.GetPrim()
                while not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                    owner = owner.GetParent()
                kinematic = UsdPhysics.RigidBodyAPI(owner).GetKinematicEnabledAttr().Get()
                # NpShape::setFlagsInternal rejects false->true triangle shapes
                # on dynamic actors, even when their mesh approximation is SDF.
                if (not before and attribute.Get() and attribute.GetPrim().IsA(UsdGeom.Mesh)
                        and not kinematic):
                    illegal.append(str(attribute.GetPath()))
                events.append("collision")
            elif attribute.GetName() == "physics:kinematicEnabled":
                events.append("kinematic")

        def velocities(data, indices):
            self.assertFalse(self.native._kinematic_flags().any())
            events.append("velocities")
            NativeActorView.set_velocities(self.native, data, indices)

        with patch("env.tactile.dynamics._restore_attribute_opinion", side_effect=restore), \
                patch.object(self.native, "set_velocities", side_effect=velocities):
            self.bridge.close()
            self.bridge.close()
        self.assertEqual(illegal, [])
        self.assertEqual(events, ["collision"] * 3 + ["kinematic"] * 2 + ["velocities"])
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), original)
        np.testing.assert_allclose(self.native.poses, initial, atol=1e-7)

    def test_native_queries_and_visuals_follow_the_same_solver_state(self):
        self.assertTrue(self.physics.GetRigidBodyEnabledAttr().Get())
        self.assertTrue(self.physics.GetKinematicEnabledAttr().Get())
        self.assertFalse(self.collision.GetCollisionEnabledAttr().Get())
        self.body.matrix[0, :3, 3] = [0.7, 0.2, 0.8]
        self.bridge.sync_visual()
        world = np.asarray(self.root.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        np.testing.assert_allclose(world[:3, 3], self.bridge.get_world_pose()[0])
        np.testing.assert_allclose(self.native.writes[-1][0, :3], world[:3, 3])
        np.testing.assert_allclose(self.bridge.get_local_pose()[0], [0.4, 0.1, 0.8])
        np.testing.assert_allclose(np.diag(world)[:3], [2, 3, 4])

    def test_saved_local_pose_resets_the_solver_and_velocity(self):
        self.bridge.set_local_pose([0.2, 0.3, 0.4], [1, 0, 0, 0])
        np.testing.assert_allclose(self.body.matrix[0, :3, 3], [0.5, 0.4, 0.4], atol=1e-7)
        self.assertEqual(self.body.velocity_clears, 1)

    def test_close_restores_native_physics_and_original_transform_ops(self):
        self.bridge.close()
        self.bridge.close()
        self.assertIsNone(self.wrapper.backend)
        self.assertTrue(self.physics.GetRigidBodyEnabledAttr().Get())
        self.assertFalse(self.physics.GetKinematicEnabledAttr().HasAuthoredValueOpinion())
        self.assertTrue(self.collision.GetCollisionEnabledAttr().Get())
        self.assertEqual(list(self.root.GetXformOpOrderAttr().Get()), ["xformOp:translate", "xformOp:scale"])
        self.assertFalse(self.root.GetPrim().HasProperty("xformOp:transform:tactile"))

    def test_only_real_native_actors_follow_and_child_reset_restores_state(self):
        self.bridge.close()
        child = UsdGeom.Xform.Define(self.stage, "/World/object/button")
        child.AddTranslateOp().Set(Gf.Vec3d(.44, .3, .7))
        child.SetResetXformStack(True)
        child_api = UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
        child_api.CreateKinematicEnabledAttr(False)
        inactive = UsdGeom.Xform.Define(self.stage, "/World/object/not_a_live_actor")
        inactive.AddTranslateOp().Set(Gf.Vec3d(.01, 0, 0))
        UsdPhysics.RigidBodyAPI.Apply(inactive.GetPrim())
        initial = [[.44, .3, .7, 0, 0, 0, 1], [.4, .3, .7, 0, 0, 0, 1]]
        self.native = NativeActorView(["/World/object/button", "/World/object"], initial, stage=self.stage)
        self.native.velocities[0, :3] = [.1, .2, .3]
        self.bridge = self.make_bridge()
        self.assertEqual(len(self.bridge.native_actor_paths), 2)
        self.assertFalse(inactive.GetPrim().HasProperty("xformOp:transform:tactileFollower"))
        self.bridge.set_local_pose([.2, .3, .4], [2 ** -.5, 0, 0, 2 ** -.5])
        # 40 mm initial world offset rotates with the authoritative body.
        np.testing.assert_allclose(self.native.poses[0, :3], [.5, .44, .4], atol=1e-7)
        child_world = np.asarray(child.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        np.testing.assert_allclose(child_world[:3, 3], self.native.poses[0, :3], atol=1e-7)
        self.assertTrue(child.GetResetXformStack())
        self.bridge.set_local_pose([.1, .2, .7], [1, 0, 0, 0])
        np.testing.assert_allclose(self.native.poses, initial, atol=1e-7)
        self.bridge.close()
        np.testing.assert_allclose(self.native.poses, initial, atol=1e-7)
        np.testing.assert_allclose(self.native.velocities[0, :3], [.1, .2, .3])
        self.assertFalse(child_api.GetKinematicEnabledAttr().Get())
        self.assertTrue(child_api.GetKinematicEnabledAttr().HasAuthoredValueOpinion())
        self.assertEqual(list(child.GetXformOpOrderAttr().Get()), ["!resetXformStack!", "xformOp:translate"])
        self.assertFalse(child.GetPrim().HasProperty("xformOp:transform:tactileFollower"))

    def test_native_pose_device_follows_physx_not_contact_solver(self):
        import torch

        self.bridge._tensor_template = torch.zeros((1, 7), device="cpu")
        self.bridge.device = "cuda:99"  # a nonexistent solver device must not be used here
        result = self.bridge._native_tensor([[0] * 7])
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(self.bridge._native_tensor([0], indices=True).dtype, torch.int32)
        self.bridge._tensor_template = self.native.poses

    def test_actor_discovery_queries_only_enabled_rigid_candidates(self):
        queries = []
        simulation = SimpleNamespace(is_valid=True, create_rigid_body_view=lambda paths: queries.append(paths) or self.native)
        module = SimpleNamespace(SimulationManager=SimpleNamespace(get_physics_sim_view=lambda: simulation))
        with patch.dict(sys.modules, {"isaacsim.core.simulation_manager": module}):
            _, actual = _native_actor_view("/World/object", self.stage)
            self.assertEqual(queries[-1], ["/World/object"])
            self.assertIs(actual, self.native)
            for name, enabled in (("button", True), ("disabled", False)):
                prim = UsdGeom.Xform.Define(self.stage, f"/World/object/{name}").GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(enabled)
            _, actual = _native_actor_view("/World/object", self.stage)
            self.assertEqual(queries[-1], ["/World/object", "/World/object/button"])
            # This fake PhysX backend does not instantiate the child candidate.
            self.assertEqual(actual.prim_paths, ["/World/object"])

    def test_foreign_or_missing_native_actor_fails_before_authoring(self):
        self.bridge.close()
        original = self.stage.GetRootLayer().ExportToString()
        for paths in (["/World/object/button"], ["/World/object", "/World/object_other"]):
            self.native = NativeActorView(paths, [[.4, .3, .7, 0, 0, 0, 1]] * len(paths), stage=self.stage)
            with self.assertRaisesRegex(RuntimeError, "Unexpected native actor set"):
                self.make_bridge()
            self.assertEqual(self.stage.GetRootLayer().ExportToString(), original)

    def test_close_removes_temporary_opinions_over_referenced_source(self):
        self.bridge.close()
        original = self.stage.GetRootLayer().ExportToString()
        # A weaker layer models authored asset values; close must not leave
        # stronger copies of those values in the live scene's edit layer.
        from pxr import Sdf

        source = Sdf.Layer.CreateAnonymous("source.usda")
        source.ImportFromString(original)
        self.stage.GetRootLayer().Clear()
        self.stage.GetRootLayer().subLayerPaths = [source.identifier]
        self.bridge = self.make_bridge()
        self.bridge.close()
        # USD may retain inert over prim specs, but none may carry authored
        # transform or physics attribute opinions after the bridge is gone.
        for path in ("/World/object.xformOpOrder", "/World/object.physics:kinematicEnabled",
                     "/World/object/collision.physics:collisionEnabled"):
            self.assertIsNone(self.stage.GetRootLayer().GetAttributeAtPath(path))
        self.assertTrue(UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath("/World/object/collision")).GetCollisionEnabledAttr().Get())


if __name__ == "__main__":
    unittest.main()
