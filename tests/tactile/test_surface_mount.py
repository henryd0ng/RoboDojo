"""CPU USD checks for virtual pad cameras and clean geometry restoration."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from env.tactile.mount import _set_world_pose
from env.tactile.surface import mount_surface_camera, restore_surface_geometry, surface_optical_height

try:
    import numpy as np
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


def sensor_config(root="/World/envs/env_0/robot0", **overrides):
    values = dict(
        body_name="link7",
        gelpad_prim_path=f"{root}/link7/gelpad_left",
        sensor_prim_path=f"{root}/link7/gelsight_left",
        optical_field_mm=(20.0, 15.0),
        camera_focus_distance=0.028,
        camera_focal_length=1.05,
        camera_clipping_range=(0.024, 0.034),
    )
    return SimpleNamespace(**(values | overrides))


def make_gel(stage, path):
    gel = UsdGeom.Xform.Define(stage, path)
    gel.AddTranslateOp().Set(Gf.Vec3d(0.08, -0.024, 0.001))
    gel.AddRotateZOp().Set(30.0)
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/mesh")
    points = [(x, y, z) for z in (0.0, 0.004) for y in (-0.009, 0.009) for x in (-0.006, 0.006)]
    mesh.CreatePointsAttr().Set(points)
    mesh.CreateFaceVertexCountsAttr().Set([3, 3])
    mesh.CreateFaceVertexIndicesAttr().Set([4, 5, 7, 4, 7, 6])
    mesh.GetPrim().CreateAttribute("tet_points", Sdf.ValueTypeNames.Point3fArray, custom=True).Set(points)
    mesh.GetPrim().CreateAttribute("tet_indices", Sdf.ValueTypeNames.IntArray, custom=True).Set([0, 1, 3, 7])
    return gel, mesh


def unit_direction(matrix, direction):
    value = np.asarray(matrix.TransformDir(Gf.Vec3d(*direction)), dtype=float)
    return value / np.linalg.norm(value)


@unittest.skipIf(Usd is None, "NumPy and USD Python bindings are required for these optional checks")
class SurfaceMountTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.config = sensor_config()
        robot = UsdGeom.Xform.Define(self.stage, "/World/envs/env_0/robot0")
        robot.AddTranslateOp().Set(Gf.Vec3d(0.2, -0.3, 0.7))
        self.body = UsdGeom.Xform.Define(self.stage, "/World/envs/env_0/robot0/link7")
        self.body.AddRotateYOp().Set(25.0)
        UsdPhysics.RigidBodyAPI.Apply(self.body.GetPrim())
        self.gel, self.mesh = make_gel(self.stage, self.config.gelpad_prim_path)

    def test_camera_looks_outward_and_longer_axis_maps_to_width(self):
        pose = mount_surface_camera(self.stage, self.config)
        sensor = self.stage.GetPrimAtPath(self.config.sensor_prim_path)
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(f"{self.config.sensor_prim_path}/Camera"))
        gel_world = self.mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_world = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        outward = unit_direction(gel_world, (0, 0, 1))
        np.testing.assert_allclose(unit_direction(camera_world, (0, 0, -1)), outward, atol=1e-8)
        np.testing.assert_allclose(
            unit_direction(camera_world, (1, 0, 0)), unit_direction(gel_world, (0, 1, 0)), atol=1e-8
        )
        front = np.asarray(gel_world.Transform(Gf.Vec3d(0, 0, 0.004)), dtype=float)
        position = np.asarray(camera_world.ExtractTranslation(), dtype=float)
        np.testing.assert_allclose(position, front - outward * 0.028, atol=1e-8)
        self.assertEqual(sensor.GetAttribute("robodojo:gelWidthAxis").Get(), "y")
        np.testing.assert_allclose(sensor.GetAttribute("robodojo:mountPose").Get(), pose)
        self.assertAlmostEqual(np.linalg.norm(pose[3:]), 1.0)
        self.assertFalse(sensor.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertEqual(camera.GetProjectionAttr().Get(), UsdGeom.Tokens.orthographic)
        self.assertAlmostEqual(camera.GetHorizontalApertureAttr().Get(), 0.2, places=6)
        self.assertAlmostEqual(camera.GetVerticalApertureAttr().Get(), 0.15, places=6)
        np.testing.assert_allclose(sensor.GetAttribute("robodojo:cameraAperture").Get(), (0.2, 0.15))

    def test_object_optics_align_to_source_thickness(self):
        mesh = UsdGeom.Mesh(self.mesh)
        points = [(x, y, z / 4) for x, y, z in mesh.GetPointsAttr().Get()]
        mesh.GetPointsAttr().Set(points)
        mesh.GetPrim().GetAttribute("tet_points").Set(points)
        config = sensor_config(camera_clipping_range=(0.027, 0.034))
        mount_surface_camera(self.stage, config)
        self.assertAlmostEqual(surface_optical_height(self.stage, config), 0.001)
        with self.assertRaisesRegex(ValueError, "source pad thickness"):
            surface_optical_height(self.stage, self.config)

    def test_camera_inherits_finger_motion_and_mount_is_repeatable(self):
        first = mount_surface_camera(self.stage, self.config)
        second = mount_surface_camera(self.stage, self.config)
        np.testing.assert_allclose(first, second)
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(f"{self.config.sensor_prim_path}/Camera"))
        before = np.asarray(
            camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(), dtype=float
        )
        self.body.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.0, 0.0))
        after = np.asarray(
            camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(), dtype=float
        )
        self.assertGreater(np.linalg.norm(after - before), 0.09)
        self.assertFalse(camera.GetResetXformStack())

    def test_physical_span_orientation_includes_mesh_scale(self):
        self.mesh.AddScaleOp().Set(Gf.Vec3f(2.0, 0.5, 1.0))
        mount_surface_camera(self.stage, self.config)
        sensor = self.stage.GetPrimAtPath(self.config.sensor_prim_path)
        self.assertEqual(sensor.GetAttribute("robodojo:gelWidthAxis").Get(), "x")
        np.testing.assert_allclose(sensor.GetAttribute("robodojo:gelSpanMm").Get(), (24.0, 9.0, 4.0), atol=1e-5)

    def test_aperture_uses_stage_units(self):
        UsdGeom.SetStageMetersPerUnit(self.stage, 0.01)
        mount_surface_camera(self.stage, self.config)
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(f"{self.config.sensor_prim_path}/Camera"))
        self.assertAlmostEqual(camera.GetHorizontalApertureAttr().Get(), 20.0)
        self.assertAlmostEqual(camera.GetVerticalApertureAttr().Get(), 15.0)

    def test_incomplete_geometry_and_missing_physical_field_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "optical_field_mm"):
            mount_surface_camera(self.stage, sensor_config(optical_field_mm=None))
        UsdGeom.Mesh.Define(self.stage, f"{self.config.gelpad_prim_path}/extra")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            mount_surface_camera(self.stage, self.config)

    def _assert_world_pose_precision(self, precision, vector_type, quaternion_type, tolerance):
        path = "/World/envs/env_0/robot0/link7/pose_camera"
        camera = UsdGeom.Xform.Define(self.stage, path)
        translate = camera.AddTranslateOp(precision=precision)
        orient = camera.AddOrientOp(precision=precision)
        translate.Set(vector_type(-1, 2, 3))
        orient.Set(quaternion_type(1, vector_type(0, 0, 0)))

        quaternion = Gf.Rotation(Gf.Vec3d(1, 2, 3), 37).GetQuat()
        position = Gf.Vec3d(0.37, 0.11, 0.92)
        pose = (*position, quaternion.GetReal(), *quaternion.GetImaginary())
        expected = Gf.Matrix4d(1).SetRotate(quaternion)
        expected.SetTranslateOnly(position)
        _set_world_pose(self.stage, path, pose)
        first_local = camera.GetLocalTransformation()
        np.testing.assert_allclose(
            camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), expected,
            atol=tolerance, rtol=0,
        )

        _set_world_pose(self.stage, path, pose)
        np.testing.assert_allclose(camera.GetLocalTransformation(), first_local, atol=tolerance, rtol=0)
        np.testing.assert_allclose(
            camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), expected,
            atol=tolerance, rtol=0,
        )
        self.assertEqual(translate.GetPrecision(), precision)
        self.assertEqual(orient.GetPrecision(), precision)
        self.assertIsInstance(translate.Get(), vector_type)
        self.assertIsInstance(orient.Get(), quaternion_type)

    def test_world_pose_preserves_double_ops_under_transformed_parent(self):
        self._assert_world_pose_precision(UsdGeom.XformOp.PrecisionDouble, Gf.Vec3d, Gf.Quatd, 1e-8)

    def test_world_pose_preserves_float_ops_under_transformed_parent(self):
        self._assert_world_pose_precision(UsdGeom.XformOp.PrecisionFloat, Gf.Vec3f, Gf.Quatf, 2e-7)

    def test_restore_in_place_preserves_camera_body_and_prim_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "robot.usda"
            source = Usd.Stage.CreateNew(str(source_path))
            UsdGeom.SetStageMetersPerUnit(source, 1.0)
            robot = UsdGeom.Xform.Define(source, "/Robot")
            source.SetDefaultPrim(robot.GetPrim())
            UsdGeom.Xform.Define(source, "/Robot/link7")
            source_gel, source_mesh = make_gel(source, "/Robot/link7/gelpad_left")
            source.GetRootLayer().Save()
            original_nodes = np.asarray(source_mesh.GetPrim().GetAttribute("tet_points").Get(), dtype=float)
            original_transform = source_gel.GetLocalTransformation()

            runtime = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageMetersPerUnit(runtime, 1.0)
            runtime_root = "/World/envs/env_0/robot0"
            root = UsdGeom.Xform.Define(runtime, runtime_root)
            root.GetPrim().GetReferences().AddReference(str(source_path))
            body = UsdGeom.Xformable(runtime.GetPrimAtPath(f"{runtime_root}/link7"))
            body.AddTranslateOp().Set(Gf.Vec3d(0.3, -0.2, 0.8))
            mount_surface_camera(runtime, self.config)
            camera_before = runtime.GetPrimAtPath(self.config.sensor_prim_path).GetAttribute("robodojo:mountPose").Get()
            body_before = body.GetLocalTransformation()
            gel = UsdGeom.Xformable(runtime.GetPrimAtPath(self.config.gelpad_prim_path))
            gel.ClearXformOpOrder()
            gel.AddTransformOp(opSuffix="runtime").Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(9, 9, 9)))
            gel.SetResetXformStack(True)
            mesh = runtime.GetPrimAtPath(f"{self.config.gelpad_prim_path}/mesh")
            mesh.GetAttribute("points").Set([Gf.Vec3f(99, 99, 99)] * len(original_nodes))
            mesh.GetAttribute("tet_points").Set([Gf.Vec3f(99, 99, 99)] * len(original_nodes))

            resynced, changed = [], []

            def record_notice(notice, sender):
                resynced.extend(notice.GetResyncedPaths())
                changed.extend(notice.GetChangedInfoOnlyPaths())

            listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, record_notice, runtime)
            try:
                restore_surface_geometry(runtime, self.config.gelpad_prim_path, str(source_path), runtime_root)
            finally:
                listener.Revoke()
            # A property can be newly authored. A prim resync can invalidate
            # PhysX views even below a finger with collision disabled.
            self.assertTrue(changed or resynced)
            self.assertTrue(all(path.IsPropertyPath() for path in resynced), resynced)
            restored = runtime.GetPrimAtPath(f"{self.config.gelpad_prim_path}/mesh")
            np.testing.assert_allclose(restored.GetAttribute("points").Get(), original_nodes)
            np.testing.assert_allclose(restored.GetAttribute("tet_points").Get(), original_nodes)
            np.testing.assert_allclose(
                UsdGeom.Xformable(runtime.GetPrimAtPath(self.config.gelpad_prim_path)).GetLocalTransformation(),
                original_transform,
            )
            np.testing.assert_allclose(body.GetLocalTransformation(), body_before)
            np.testing.assert_allclose(
                runtime.GetPrimAtPath(self.config.sensor_prim_path).GetAttribute("robodojo:mountPose").Get(),
                camera_before,
            )
            self.assertTrue(runtime.GetPrimAtPath(f"{self.config.sensor_prim_path}/Camera").IsA(UsdGeom.Camera))

            incomplete = Usd.Stage.CreateInMemory()
            UsdGeom.Xform.Define(incomplete, self.config.gelpad_prim_path)
            with self.assertRaisesRegex(ValueError, "must already exist"):
                restore_surface_geometry(incomplete, self.config.gelpad_prim_path, str(source_path), runtime_root)


if __name__ == "__main__":
    unittest.main()
