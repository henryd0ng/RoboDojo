"""CPU-only scene identity and collision conversion checks; no Isaac application."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from env.tactile.config import load_tactile_config
from env.tactile.manager import RoboDojoTactileManager
from env.tactile.scene import (
    ScenePreparationError,
    _cached_mesh,
    collect_robot_colliders,
    collect_scene_objects,
    prepare_scene,
    split_scale_transform,
    tetrahedralize_surface,
    validate_tetrahedra,
)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


def wrapper(name, kind="rigid", mass=0.03, scale=(1, 1, 1)):
    return SimpleNamespace(instance_name=name, _prim_path=f"/World/envs/env_0/{name}",
                           mass=mass, scale=scale, usd_path=None, is_static=kind != "rigid")


def manager(rigids=(), geometry=(), table=None):
    records = {"Rigid": [{"inst_name": obj.instance_name} for obj in rigids],
               "Geometry": [{"inst_name": obj.instance_name} for obj in geometry]}
    return SimpleNamespace(num_envs=1,
        layout_manager=SimpleNamespace(get_layout_records=lambda env, kind: records.get(kind, [])),
        _rigid_and_dynamic_objects=[{obj.instance_name: obj for obj in rigids}],
        _geometry_objects=[{obj.instance_name: obj for obj in geometry}], _tables=[table])


def sensor_asset_directory():
    """Locate installed assets without importing TacEx's Isaac-dependent modules."""
    override = os.environ.get("TACEX_SENSOR_ASSETS")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        spec = importlib.util.find_spec("tacex_assets")
        if spec is None or not spec.origin:
            raise RuntimeError("Install tacex_assets or set TACEX_SENSOR_ASSETS to the GelSight_Mini directory")
        root = Path(spec.origin).resolve().parent / "data/Sensors/GelSight_Mini"
    missing = [name for name in ("Case.usd", "Gelpad_high_res.usd") if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing TacEx sensor assets under {root}: {', '.join(missing)}")
    return root


class SensorAssetLocationTests(unittest.TestCase):
    def test_explicit_directory_takes_precedence_without_loading_tacex(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("Case.usd", "Gelpad_high_res.usd"):
                (root / name).touch()
            with patch.dict(os.environ, {"TACEX_SENSOR_ASSETS": directory}), patch(
                "importlib.util.find_spec", side_effect=AssertionError("override must not inspect packages")
            ):
                self.assertEqual(sensor_asset_directory(), root)

    def test_installed_package_location_and_missing_asset_error(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "tacex_assets"
            root = package / "data/Sensors/GelSight_Mini"
            root.mkdir(parents=True)
            (root / "Case.usd").touch()
            with patch.dict(os.environ, {"TACEX_SENSOR_ASSETS": ""}), patch(
                "importlib.util.find_spec", return_value=SimpleNamespace(origin=str(package / "__init__.py"))
            ):
                with self.assertRaisesRegex(FileNotFoundError, "Gelpad_high_res.usd"):
                    sensor_asset_directory()
                (root / "Gelpad_high_res.usd").touch()
                self.assertEqual(sensor_asset_directory(), root)


class SceneCollectionTests(unittest.TestCase):
    def test_three_native_task_counts_and_dynamic_roles(self):
        cases = [
            (manager([wrapper("charger"), wrapper("socket", mass=0.05)]), {"charger", "socket"}, set()),
            (manager([wrapper(f"tube{i}") for i in range(3)], [wrapper("slot", "geometry")]),
             {"tube0", "tube1", "tube2"}, {"slot"}),
            (manager([wrapper("coin0", mass=0.005)],
                     [wrapper("vertical_coin_stand", "geometry"), wrapper("piggy_bank", "geometry")]),
             {"coin0"}, {"vertical_coin_stand", "piggy_bank"}),
        ]
        for scene, dynamic, static in cases:
            with self.subTest(dynamic=dynamic):
                scene._tables[0] = wrapper("table", "table")
                scene._rigid_and_dynamic_objects[0]["stale_previous_episode"] = wrapper("stale_previous_episode")
                objects = collect_scene_objects(scene)
                self.assertEqual({o.instance_name for o in objects if o.dynamic}, dynamic)
                self.assertEqual({o.instance_name for o in objects if not o.dynamic}, static | {"table"})
                self.assertEqual(len(objects), len(dynamic) + len(static) + 1)

    def test_effective_socket_mass_and_scale_are_preserved(self):
        socket = wrapper("socket", mass=0.05, scale=(2, 3, 4))
        scene = manager([socket])
        scene.layout_manager.get_layout_records(0, "Rigid")[0]["label"] = "target_socket"
        obj = collect_scene_objects(scene)[0]
        self.assertEqual(obj.mass, 0.05)
        self.assertTrue(obj.dynamic)
        self.assertEqual(obj.scale, (2, 3, 4))
        self.assertIs(obj.wrapper, socket)
        self.assertEqual(obj.layout_label, "target_socket")

    def test_missing_active_instance_is_not_silently_dropped(self):
        scene = manager([wrapper("tube0")])
        scene._rigid_and_dynamic_objects[0].clear()
        with self.assertRaisesRegex(ScenePreparationError, "not spawned"):
            collect_scene_objects(scene)

    def test_invalid_effective_mass_fails(self):
        with self.assertRaisesRegex(ScenePreparationError, "effective native mass"):
            collect_scene_objects(manager([wrapper("socket", mass=0)]))

    def test_articulations_are_not_flattened(self):
        scene = manager([wrapper("object")])
        scene.layout_manager.get_layout_records = lambda env, kind: [{"inst_name": "hinged"}] if kind == "Articulation" else []
        with self.assertRaisesRegex(ScenePreparationError, "Articulation"):
            collect_scene_objects(scene)


class CollisionNumericsTests(unittest.TestCase):
    def test_source_cache_hit_rejects_changed_boundary_and_small_volume_metadata_error(self):
        surface = trimesh.creation.box(extents=[.02, .02, .001])
        provenance = {"source_mesh_policy": {"method": "tetgen_plc", "version": 2}}
        with tempfile.TemporaryDirectory() as directory:
            result = _cached_mesh(surface, provenance, directory, False, .0003)
            cache_path = result[4]
            with np.load(cache_path, allow_pickle=False) as stored:
                original = {name: stored[name].copy() for name in stored.files}
            with patch("env.tactile.scene.tetrahedralize_surface", side_effect=AssertionError("cache miss")):
                repeated = _cached_mesh(surface, provenance, directory, False, .0003)
                self.assertEqual(repeated[3]["cache_status"], "hit")
                for index in range(3):
                    np.testing.assert_array_equal(repeated[index], result[index])
                for mutation in ("boundary", "volume"):
                    with self.subTest(mutation=mutation):
                        changed = {name: value.copy() for name, value in original.items()}
                        if mutation == "boundary":
                            # Translation preserves topology and volume, but changes contact geometry.
                            changed["points"][:, 0] += 1e-7
                            error = "preserve every original source boundary vertex"
                        else:
                            metrics = json.loads(str(changed["metrics"]))
                            metrics["volume_m3"] *= 1.01
                            changed["metrics"] = np.asarray(json.dumps(metrics))
                            # This 4e-9 m³ difference passes the legacy default atol=1e-8.
                            self.assertLess(abs(metrics["volume_m3"] - surface.volume), 1e-8)
                            error = "Source cache volume mismatch"
                        np.savez_compressed(cache_path, **changed)
                        with self.assertRaisesRegex(ScenePreparationError, error):
                            _cached_mesh(surface, provenance, directory, False, .0003)

    def test_old_cache_path_alias_and_float_pose_stretch_require_surface_revalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.usda"
            provenance = {"source_path": str(source), "source_sha256": "same-source-content",
                          "baked_scale": [1.0, 1.0, 1.0], "colliders": [{"prim": "/collision"}],
                          "source_watertight": True, "scale_roundoff_policy": "authored scale"}
            old_provenance = {**provenance, "source_path": str(source.parent / "alias/../source.usda"),
                              "baked_scale": [1 + 8e-8, 1 - 9e-8, 1 + 4e-8]}
            old_provenance.pop("scale_roundoff_policy")
            old_identity = {"format_version": 5, "provenance": old_provenance,
                            "allow_surface_repair": True, "max_surface_error": .0003}
            surface = trimesh.creation.box(extents=[.02, .03, .04])
            old_surface = surface.copy()
            old_surface.apply_scale(old_provenance["baked_scale"])
            points, tets, _, metrics = tetrahedralize_surface(old_surface)
            previous = Path(directory) / "legacy.npz"
            np.savez_compressed(previous, points=points, tets=tets, metrics=json.dumps(metrics),
                                identity=json.dumps(old_identity, sort_keys=True))
            with patch("env.tactile.scene.tetrahedralize_surface", side_effect=AssertionError("must reuse audited cache")):
                result = _cached_mesh(surface, provenance, directory, True, .0003)
                self.assertEqual(result[3]["cache_status"], "migrated_revalidated")
                self.assertEqual(result[3]["migrated_from_cache"], str(previous))
                self.assertLess(result[3]["tet_to_source_sampled_max_m"], 1e-8)
                self.assertTrue(result[4].exists())
                repeated = _cached_mesh(surface, provenance, directory, True, .0003)
                self.assertEqual(repeated[3]["cache_status"], "hit")
            np.testing.assert_array_equal(result[0], points)
            self.assertFalse(list(Path(directory).glob("tmp*.npz")))

    def test_old_cache_metadata_cannot_hide_changed_contact_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            provenance = {"source_path": str(Path(directory) / "source.usda"),
                          "source_sha256": "same-source-content", "baked_scale": [1., 1., 1.],
                          "colliders": [{"prim": "/collision"}], "source_watertight": True}
            identity = {"format_version": 5, "provenance": provenance,
                        "allow_surface_repair": True, "max_surface_error": .0003}
            source = trimesh.creation.box(extents=[.02, .03, .04])
            wrong = trimesh.creation.box(extents=[.08, .03, .04])
            points, tets, _, metrics = tetrahedralize_surface(wrong)
            np.savez_compressed(Path(directory) / "wrong.npz", points=points, tets=tets,
                                metrics=json.dumps(metrics), identity=json.dumps(identity, sort_keys=True))
            with patch("env.tactile.scene.tetrahedralize_surface", wraps=tetrahedralize_surface) as converter:
                result = _cached_mesh(source, provenance, directory, True, .0003)
            self.assertEqual(converter.call_count, 1)
            self.assertEqual(result[3]["cache_status"], "built")
            self.assertAlmostEqual(result[3]["volume_m3"], source.volume)

    def test_convex_conversion_preserves_volume_and_surface(self):
        surface = trimesh.creation.box(extents=[0.02, 0.03, 0.04])
        points, tets, boundary, metrics = tetrahedralize_surface(surface, max_surface_error=1e-8)
        self.assertAlmostEqual(metrics["volume_m3"], 0.02 * 0.03 * 0.04, places=12)
        determinants = np.linalg.det(points[tets[:, 1:]] - points[tets[:, 0, None]])
        self.assertTrue(np.all(determinants > 0))
        self.assertTrue(trimesh.Trimesh(points, boundary, process=False).is_watertight)
        self.assertLess(metrics["tet_to_source_sampled_max_m"], 1e-12)

    def test_open_surface_requires_explicit_repair(self):
        surface = trimesh.creation.box()
        surface.update_faces(np.arange(len(surface.faces) - 1))
        with self.assertRaisesRegex(ScenePreparationError, "allow_surface_repair"):
            tetrahedralize_surface(surface)

    def test_planar_repair_is_bounded_and_preserves_opening(self):
        # Two disconnected strips bound a 10 mm opening; thickening must not
        # replace them by their convex hull or close the opening.
        points = np.array([[-.02, -.01, 0], [-.005, -.01, 0], [-.005, .01, 0], [-.02, .01, 0],
                           [.005, -.01, 0], [.02, -.01, 0], [.02, .01, 0], [.005, .01, 0]])
        surface = trimesh.Trimesh(points, [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], process=False)
        vertices, _, boundary, metrics = tetrahedralize_surface(surface, allow_surface_repair=True)
        self.assertEqual(metrics["method"], "bounded_planar_thickening")
        self.assertAlmostEqual(metrics["repair"]["symmetric_thickness_m"], .0002)
        self.assertAlmostEqual(metrics["volume_m3"], 2 * .015 * .02 * .0002)
        converted = trimesh.Trimesh(vertices, boundary, process=False)
        hits = converted.ray.intersects_any([[0, 0, -.01], [.01, 0, -.01]], [[0, 0, 1]] * 2)
        self.assertEqual(hits.tolist(), [False, True])
        self.assertLessEqual(metrics["tet_to_source_sampled_max_m"], .000100001)

    def test_open_wall_repair_does_not_cap_the_missing_face(self):
        surface = trimesh.creation.box(extents=[.1] * 3)
        surface.update_faces(np.arange(len(surface.faces) - 1))
        points, _, boundary, metrics = tetrahedralize_surface(surface, allow_surface_repair=True, max_surface_error=.0003)
        self.assertEqual(metrics["method"], "bounded_open_surface_thickening")
        output = trimesh.Trimesh(points, boundary, process=False)
        # The omitted face's centre remains far from the thin repaired shell.
        missing = trimesh.creation.box(extents=[.1] * 3).triangles_center[-1]
        _, distance, _ = trimesh.proximity.closest_point(output, [missing])
        self.assertGreater(distance[0], .02)
        self.assertLess(metrics["volume_m3"], .1 ** 3 * .1)

    def test_compound_colliders_share_one_mesh_and_report_summed_volume(self):
        first = trimesh.creation.box(extents=[.02] * 3)
        second = first.copy()
        second.apply_translation([.01, 0, 0])  # author overlap is intentionally retained
        surface = trimesh.util.concatenate([first, second])
        surface.metadata["robodojo_collision_parts"] = [first, second]
        _, cells, _, metrics = tetrahedralize_surface(surface)
        self.assertEqual(metrics["method"], "author_collider_compound")
        self.assertEqual(len(cells), 24)
        self.assertAlmostEqual(metrics["volume_m3"], 2 * .02 ** 3)
        self.assertIn("inertia is not equivalent", metrics["mass_distribution_note"])

    def test_surface_audit_rejects_backend_that_fills_an_insertion_hole(self):
        ring = trimesh.creation.annulus(r_min=.01, r_max=.02, height=.01, sections=16)
        filled = trimesh.creation.cylinder(radius=.02, height=.01, sections=16)
        points, cells, _, _ = tetrahedralize_surface(filled)
        with patch("env.tactile.scene._wild_tetrahedra", return_value=(points, cells)):
            with self.assertRaisesRegex(ScenePreparationError, "Surface deviation"):
                tetrahedralize_surface(ring, max_surface_error=.0003)

    def test_static_triangle_skin_preserves_closed_surface_hole(self):
        ring = trimesh.creation.annulus(r_min=.01, r_max=.02, height=.01, sections=16)
        ring.metadata["robodojo_static_triangle_skin"] = True
        with self.assertRaisesRegex(ScenePreparationError, "explicit allow_surface_repair"):
            tetrahedralize_surface(ring)
        with patch("env.tactile.scene._wild_tetrahedra", side_effect=AssertionError("must retain source triangles")):
            points, cells, boundary, metrics = tetrahedralize_surface(ring, allow_surface_repair=True)
        self.assertEqual(metrics["method"], "bounded_static_triangle_skin")
        self.assertEqual(len(cells), 3 * len(ring.faces))
        self.assertLess(metrics["volume_m3"], ring.volume * .1)
        output = trimesh.Trimesh(points, boundary, process=False)
        hits = output.ray.intersects_any([[0, 0, -.02], [.015, 0, -.02]], [[0, 0, 1]] * 2)
        self.assertEqual(hits.tolist(), [False, True])
        self.assertLessEqual(metrics["tet_to_source_sampled_max_m"], .000100001)

    def test_static_triangle_skin_rejects_nonmanifold_edges_without_fallback(self):
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]]
        surface = trimesh.Trimesh(points, [[0, 1, 2], [1, 0, 3], [0, 1, 4]], process=False)
        surface.metadata["robodojo_static_triangle_skin"] = True
        with patch("env.tactile.scene._wild_tetrahedra", side_effect=AssertionError("no silent repair")):
            with self.assertRaisesRegex(ScenePreparationError, "non-manifold source edge"):
                tetrahedralize_surface(surface, allow_surface_repair=True)

    def test_degenerate_or_duplicate_tets_fail(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        with self.assertRaisesRegex(ScenePreparationError, "Degenerate"):
            validate_tetrahedra(points, [[0, 1, 2, 2]])
        with self.assertRaisesRegex(ScenePreparationError, "Duplicate"):
            validate_tetrahedra(points, [[0, 1, 2, 3], [3, 1, 2, 0]])

    def test_scale_is_removed_once_from_pose(self):
        matrix = np.array([[0, -3, 0, 0.4], [2, 0, 0, 0.2], [0, 0, 4, 0.8], [0, 0, 0, 1.]])
        pose, scale = split_scale_transform(matrix)
        np.testing.assert_allclose(scale, [2, 3, 4])
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3))
        np.testing.assert_allclose(pose @ np.diag([*scale, 1]), matrix)
        matrix[0, 0] = 0.5
        with self.assertRaisesRegex(ScenePreparationError, "shear"):
            split_scale_transform(matrix)


@unittest.skipIf(Usd is None, "Standalone USD bindings unavailable")
class UsdSceneTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.object = wrapper("charger", scale=(2, 3, 4))
        root = UsdGeom.Xform.Define(self.stage, self.object._prim_path)
        root.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.2, 0.8))
        root.AddOrientOp().Set(Gf.Quatf(0.70710678, Gf.Vec3f(0, 0, 0.70710678)))
        root.AddScaleOp().Set(Gf.Vec3f(2, 3, 4))
        collider = UsdGeom.Cube.Define(self.stage, self.object._prim_path + "/collision")
        collider.CreateSizeAttr(0.01)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        # A visual-only far-away mesh must not participate in collision conversion.
        visual = UsdGeom.Cube.Define(self.stage, self.object._prim_path + "/visual")
        visual.CreateSizeAttr(5)
        self.scene = manager([self.object])
        self.scene.stage = self.stage
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def test_empty_source_options_keep_native_colliders_and_existing_cache(self):
        with patch("env.tactile.scene._source_tetgen_mesh", side_effect=AssertionError("unexpected source override")):
            original = prepare_scene(self.scene, cache_dir=self.temporary.name)
            self.stage.RemovePrim(original.root_path)
            with patch("env.tactile.scene.tetrahedralize_surface", side_effect=AssertionError("cache miss")):
                repeated = prepare_scene(self.scene, cache_dir=self.temporary.name, source_meshes={})
        first = original.bindings["env0:charger"]
        second = repeated.bindings["env0:charger"]
        self.assertEqual(first.cache_path, second.cache_path)
        self.assertAlmostEqual(second.volume, .02 * .03 * .04)
        self.assertEqual(second.provenance["colliders"][0]["prim"], "collision")
        self.assertNotIn("source_mesh_policy", second.provenance)
        self.assertEqual({key: value for key, value in first.provenance.items() if key != "cache_status"},
                         {key: value for key, value in second.provenance.items() if key != "cache_status"})

    def test_invalid_source_options_fail_before_usd_authoring(self):
        source = "Assets/Object/RoboDojo/Rigid/coin/00000/object.usdz"
        policy = {"prim_path": "visual", "method": "tetgen_plc"}
        invalid = [False, [], "", {source: None}, {False: policy},
                   {"/tmp/object.usd": policy}, {"../object.usd": policy},
                   *({source: {**policy, "prim_path": value}}
                     for value in (None, True, float("nan"), "/visual", "../visual")),
                   {source: {**policy, "method": "wildmeshing"}},
                   {source: {**policy, "seam_tolerance_m": 1e-8}}]
        original = self.stage.GetRootLayer().ExportToString()
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ScenePreparationError):
                prepare_scene(self.scene, cache_dir=self.temporary.name, source_meshes=options)
            self.assertEqual(self.stage.GetRootLayer().ExportToString(), original)
            self.assertFalse(self.stage.GetPrimAtPath("/World/TactileObjects").IsValid())

    def test_prepare_preserves_native_usd_and_bakes_scale_once(self):
        original = self.stage.GetRootLayer().ExportToString()
        prepared = prepare_scene(self.scene, cache_dir=self.temporary.name)
        self.assertEqual(len(prepared.specs), 1)
        spec = prepared.specs[0]
        binding = prepared.bindings[spec["name"]]
        self.assertIs(binding.scene_object.wrapper, self.object)
        self.assertFalse(spec["kinematic"])
        self.assertAlmostEqual(binding.volume, 0.02 * 0.03 * 0.04, places=12)
        self.assertAlmostEqual(spec["mass_density"] * binding.volume, 0.03)
        collision = UsdGeom.Mesh.Get(self.stage, spec["prim_path"] + "/mesh")
        points = np.asarray(collision.GetPointsAttr().Get())
        np.testing.assert_allclose(np.ptp(points, axis=0), [0.02, 0.03, 0.04], atol=1e-7)
        matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(spec["prim_path"]))).T
        np.testing.assert_allclose(np.linalg.norm(matrix[:3, :3], axis=0), [1, 1, 1], atol=1e-7)
        np.testing.assert_allclose(matrix[:3, 3], [0.4, 0.2, 0.8])
        self.assertEqual(collision.GetVisibilityAttr().Get(), "invisible")
        self.stage.RemovePrim(prepared.root_path)
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), original)

    def test_cache_reuses_geometry_but_does_not_freeze_layout_pose(self):
        first = prepare_scene(self.scene, cache_dir=self.temporary.name)
        self.stage.RemovePrim(first.root_path)
        root = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.object._prim_path))
        root.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.8, 0.1, 0.9))
        with patch("env.tactile.scene.tetrahedralize_surface", side_effect=AssertionError("cache not reused")):
            second = prepare_scene(self.scene, cache_dir=self.temporary.name)
        self.assertEqual(next(iter(first.bindings.values())).cache_path, next(iter(second.bindings.values())).cache_path)
        np.testing.assert_allclose(second.specs[0]["initial_pose"][:3], [0.8, 0.1, 0.9])

    def test_scale_change_invalidates_cache(self):
        first = prepare_scene(self.scene, cache_dir=self.temporary.name)
        self.stage.RemovePrim(first.root_path)
        root = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.object._prim_path))
        root.GetOrderedXformOps()[2].Set(Gf.Vec3f(1, 1, 1))
        second = prepare_scene(self.scene, cache_dir=self.temporary.name)
        self.assertNotEqual(next(iter(first.bindings.values())).cache_path, next(iter(second.bindings.values())).cache_path)
        self.assertAlmostEqual(next(iter(second.bindings.values())).volume, 1e-6, places=12)

    def test_dense_identical_instances_share_cache_across_world_poses(self):
        from scipy.spatial.transform import Rotation

        collision_path = self.object._prim_path + "/collision"
        self.stage.RemovePrim(collision_path)
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=.01)
        mesh = UsdGeom.Mesh.Define(self.stage, collision_path)
        mesh.CreatePointsAttr(sphere.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(sphere.faces))
        mesh.CreateFaceVertexIndicesAttr(sphere.faces.reshape(-1).tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        second = wrapper("same_asset_elsewhere", scale=(2, 3, 4))
        prim = self.stage.DefinePrim(second._prim_path, "Xform")
        prim.GetReferences().AddInternalReference(self.object._prim_path)
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        ops[0].Set(Gf.Vec3d(-.3, -.45, .765))
        q = Rotation.from_euler("xyz", [37, -21, 81], degrees=True).as_quat()
        ops[1].Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
        scene = manager([self.object, second])
        scene.stage = self.stage
        with patch("env.tactile.scene.tetrahedralize_surface", wraps=tetrahedralize_surface) as convert:
            prepared = prepare_scene(scene, cache_dir=self.temporary.name)
        self.assertEqual(convert.call_count, 1)
        self.assertEqual(len({binding.cache_path for binding in prepared.bindings.values()}), 1)
        self.assertEqual([binding.provenance["cache_status"] for binding in prepared.bindings.values()], ["built", "hit"])

    def test_no_visual_fallback_and_no_partial_usd_on_failure(self):
        self.stage.GetPrimAtPath(self.object._prim_path + "/collision").RemoveAPI(UsdPhysics.CollisionAPI)
        with self.assertRaisesRegex(ScenePreparationError, "No enabled explicit collision"):
            prepare_scene(self.scene, cache_dir=self.temporary.name)
        self.assertFalse(self.stage.GetPrimAtPath("/World/TactileObjects").IsValid())

    def test_explicit_collision_container_uses_only_its_imported_mesh(self):
        collider_path = self.object._prim_path + "/collision"
        self.stage.RemovePrim(collider_path)
        owner = UsdGeom.Xform.Define(self.stage, collider_path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(owner)
        UsdPhysics.MeshCollisionAPI.Apply(owner).CreateApproximationAttr("convexHull")
        cube = trimesh.creation.box(extents=[.01] * 3)
        mesh = UsdGeom.Mesh.Define(self.stage, collider_path + "/mesh")
        mesh.CreatePointsAttr(cube.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(cube.faces))
        mesh.CreateFaceVertexIndicesAttr(cube.faces.reshape(-1).tolist())
        prepared = prepare_scene(self.scene, cache_dir=self.temporary.name)
        binding = next(iter(prepared.bindings.values()))
        self.assertAlmostEqual(binding.volume, .02 * .03 * .04)
        self.assertEqual(binding.provenance["colliders"][0]["collision_owner"], "collision")
        self.assertEqual(binding.provenance["colliders"][0]["prim"], "collision/mesh")

    def test_existing_output_requires_world_teardown(self):
        prepare_scene(self.scene, cache_dir=self.temporary.name)
        with self.assertRaisesRegex(ScenePreparationError, "destroy old world"):
            prepare_scene(self.scene, cache_dir=self.temporary.name)

    def make_robot(self, index=0, envs=(0,), with_housing=False, housing_listed=False):
        for env in envs:
            path = f"/World/envs/env_{env}/robot{index}"
            UsdGeom.Xform.Define(self.stage, path)
            for name in ("base", "finger"):
                body = UsdGeom.Xform.Define(self.stage, path + "/" + name)
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                body.AddTranslateOp().Set(Gf.Vec3d(env * 2.0, 0.1 if name == "finger" else 0, 0.8))
                collider = UsdGeom.Cube.Define(self.stage, path + "/" + name + "/collision")
                collider.CreateSizeAttr(0.01)
                UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            if with_housing:
                housing_path = path + "/finger/housing"
                body = UsdGeom.Xform.Define(self.stage, housing_path)
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                body.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.02))
                collider = UsdGeom.Cube.Define(self.stage, housing_path + "/collision")
                collider.CreateSizeAttr(0.005)
                UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
                joint = UsdPhysics.FixedJoint.Define(self.stage, path + "/housing_attachment")
                joint.CreateBody0Rel().SetTargets([Sdf.Path(path + "/finger")])
                joint.CreateBody1Rel().SetTargets([Sdf.Path(housing_path)])
        names = ["base", "finger"] + (["housing"] if housing_listed else [])
        key = SimpleNamespace(cfg=SimpleNamespace(prim_path=f"{{ENV_REGEX_NS}}/robot{index}", spawn=None),
                              body_names=names)
        return SimpleNamespace(robot_list=[SimpleNamespace()], robot_key=[key], num_envs=len(envs))

    def test_robot_binding_names_persist_across_two_environments(self):
        robots = self.make_robot(envs=(0, 1))
        objects = collect_robot_colliders(robots, self.stage)
        self.assertEqual({(o.env_id, o.body_name) for o in objects},
                         {(0, "base"), (0, "finger"), (1, "base"), (1, "finger")})
        prepared = prepare_scene(self.scene, cache_dir=self.temporary.name, extra_objects=objects)
        self.assertEqual(len(prepared.specs), 5)
        for spec in prepared.specs[1:]:
            binding = prepared.bindings[spec["name"]]
            self.assertTrue(spec["kinematic"])
            self.assertIs(binding.robot_key, robots.robot_key[0])
            self.assertEqual(binding.body_name, binding.scene_object.body_name)
            self.assertEqual(binding.initial_pose[0], binding.scene_object.env_id * 2.0)

    def test_fixed_nested_housing_has_exactly_one_collision_owner(self):
        for listed in (False, True):
            with self.subTest(housing_listed=listed):
                robots = self.make_robot(index=int(listed), with_housing=True, housing_listed=listed)
                objects = collect_robot_colliders(robots, self.stage)
                paths = [path for obj in objects for path in obj.collider_prim_paths]
                self.assertEqual(len(paths), 3)
                self.assertEqual(len(set(paths)), 3)
                housing_owner = next(obj for obj in objects if any("housing/collision" in p for p in obj.collider_prim_paths))
                self.assertEqual(housing_owner.body_name, "housing" if listed else "finger")


    def test_unlisted_unfixed_nested_rigid_body_fails(self):
        robots = self.make_robot(with_housing=True)
        self.stage.RemovePrim("/World/envs/env_0/robot0/housing_attachment")
        with self.assertRaisesRegex(ScenePreparationError, "not fixed"):
            collect_robot_colliders(robots, self.stage)

    def test_gel_exclusion_is_explicit_and_removes_all_descendants(self):
        robots = self.make_robot()
        gel_path = "/World/envs/env_0/robot0/finger/gelpad"
        gel = UsdGeom.Cube.Define(self.stage, gel_path + "/mesh")
        gel.GetPrim().CreateAttribute("tet_points", Sdf.ValueTypeNames.Point3fArray).Set([Gf.Vec3f(0, 0, 0)])
        UsdPhysics.CollisionAPI.Apply(gel.GetPrim())
        with self.assertRaisesRegex(ScenePreparationError, "Exclude FEM"):
            collect_robot_colliders(robots, self.stage)
        objects = collect_robot_colliders(robots, self.stage, exclude_prim_paths=[gel_path])
        self.assertEqual(sum(len(obj.collider_prim_paths) for obj in objects), 2)
        self.assertTrue(all("gelpad" not in p for obj in objects for p in obj.collider_prim_paths))

    def test_shared_articulation_key_does_not_duplicate_shells(self):
        robots = self.make_robot()
        robots.robot_list.append(SimpleNamespace())
        robots.robot_key.append(robots.robot_key[0])
        self.assertEqual(len(collect_robot_colliders(robots, self.stage)), 2)

    def test_robot_cache_ignores_path_alias_and_reset_pose_roundoff(self):
        source_path = Path(self.temporary.name) / "robot.usda"
        source = Usd.Stage.CreateNew(str(source_path))
        root = UsdGeom.Xform.Define(source, "/Robot").GetPrim()
        source.SetDefaultPrim(root)
        finger = UsdGeom.Xform.Define(source, "/Robot/finger").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(finger)
        cube = UsdGeom.Cube.Define(source, "/Robot/finger/collision")
        cube.CreateSizeAttr(.01)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        source.GetRootLayer().Save()
        root_path = "/World/envs/env_0/robot0"
        prim = self.stage.DefinePrim(root_path, "Xform")
        prim.GetReferences().AddReference(str(source_path))
        key = SimpleNamespace(cfg=SimpleNamespace(prim_path=root_path,
                              spawn=SimpleNamespace(usd_path=str(source_path), scale=(1, 1, 1))), body_names=["finger"])
        robots = SimpleNamespace(robot_list=[SimpleNamespace()], robot_key=[key], num_envs=1)
        first = prepare_scene(self.scene, cache_dir=self.temporary.name,
                              extra_objects=collect_robot_colliders(robots, self.stage))
        self.stage.RemovePrim(first.root_path)
        (Path(self.temporary.name) / "alias").mkdir()
        key.cfg.spawn.usd_path = str(Path(self.temporary.name) / "alias/../robot.usda")
        body = UsdGeom.Xformable(self.stage.GetPrimAtPath(root_path + "/finger"))
        matrix = np.diag([1 + 8e-8, 1 - 9e-8, 1 + 4e-8, 1.0])
        matrix[:3, 3] = [.12, -.3, .7]
        body.AddTransformOp(opSuffix="tactileReset").Set(Gf.Matrix4d(matrix.T.tolist()))
        extras = collect_robot_colliders(robots, self.stage)
        self.assertEqual(extras[0].scale, (1., 1., 1.))
        self.assertEqual(extras[0].source_path, str(source_path))
        with patch("env.tactile.scene.tetrahedralize_surface", side_effect=AssertionError("reset geometry cache miss")):
            second = prepare_scene(self.scene, cache_dir=self.temporary.name, extra_objects=extras)
        self.assertEqual(first.bindings["env0:robot0:finger"].cache_path,
                         second.bindings["env0:robot0:finger"].cache_path)
        np.testing.assert_allclose(second.bindings["env0:robot0:finger"].initial_pose[:3], [.12, -.3, .7])

    def test_duplicate_extra_objects_fail_before_authoring(self):
        objects = collect_scene_objects(self.scene)
        with self.assertRaisesRegex(ScenePreparationError, "Duplicate native"):
            prepare_scene(self.scene, cache_dir=self.temporary.name, extra_objects=objects)
        self.assertFalse(self.stage.GetPrimAtPath("/World/TactileObjects").IsValid())


@unittest.skipUnless(Usd is not None and os.environ.get("ROBODOJO_TACTILE_NATIVE_PREFLIGHT") == "1",
                     "Set ROBODOJO_TACTILE_NATIVE_PREFLIGHT=1 for real-asset CPU prewarm")
class NativeLayoutPreflightTests(unittest.TestCase):
    """Real USD/layout fixtures using SceneManager's read-only wrapper protocol.

    This does not start Isaac or claim to replace the live SceneManager reset
    test. The table has the native unit-cube dimensions without Kit tessellation;
    all saved task instances and both ARX assets reference their original USDs.
    """

    def test_native_layouts_prepare_with_two_robots_and_cad_pads(self):
        from scipy.spatial.transform import Rotation
        import yaml

        repo = Path(__file__).resolve().parents[2]
        cache_dir = Path(os.environ.get("ROBODOJO_TACTILE_PREFLIGHT_CACHE", repo / ".cache/tactile/native_collision"))
        sensor_config = yaml.safe_load((repo / "env_cfg/arx_x5_pad_tactile.yml").read_text())["tactile"]
        from env.tactile.config import load_tactile_config

        tactile_config = load_tactile_config(sensor_config)
        robot_config = yaml.safe_load((repo / "env_cfg/robot/dual_x5_pad.yml").read_text())["robots"]
        reports = []
        tasks = os.environ.get("ROBODOJO_TACTILE_PREFLIGHT_TASKS",
                               "insert_tubes,plug_in_charger,deposit_coin").split(",")
        for task in tasks:
            with self.subTest(task=task):
                started = time.monotonic()
                stage = Usd.Stage.CreateInMemory()
                UsdGeom.SetStageMetersPerUnit(stage, 1.0)

                def transform(prim, position, quaternion, scale=(1, 1, 1), reset=False):
                    q = np.asarray(quaternion, dtype=float)
                    q /= np.linalg.norm(q)
                    xform = UsdGeom.Xformable(prim)
                    xform.ClearXformOpOrder()
                    translation = (UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))
                                   if prim.HasAttribute("xformOp:translate") else xform.AddTranslateOp())
                    orientation = (UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
                                   if prim.HasAttribute("xformOp:orient") else xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble))
                    scaling = (UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))
                               if prim.HasAttribute("xformOp:scale") else xform.AddScaleOp())
                    translation.Set(Gf.Vec3d(*position))
                    quaternion_value = (Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:]))
                                        if orientation.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
                                        else Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
                    orientation.Set(quaternion_value)
                    scaling.Set(Gf.Vec3d(*scale))
                    xform.SetXformOpOrder([translation, orientation, scaling], resetXformStack=reset)

                layout = json.loads((repo / "Assets/Eval_Layout/RoboDojo/arx_x5/0" / f"{task}_0.json").read_text())
                objects = {"Rigid": [], "Geometry": []}
                for kind in objects:
                    for category, entries in layout.get(kind, {}).items():
                        for index, entry in enumerate(entries):
                            model = entry["category_idx"]
                            name = f"{category}_{model}_{index}"
                            source = repo / "Assets/Object/RoboDojo" / kind / category / f"{model:05d}/object.usdz"
                            if not source.exists():
                                source = source.with_suffix(".usd")
                            path = f"/World/envs/env_0/{kind.lower()}/{category}/{name}"
                            prim = stage.DefinePrim(path, "Xform")
                            prim.GetReferences().AddReference(str(source))
                            transform(prim, entry["default_pos"], entry["default_ori"], entry["scale"])
                            effective_mass = min(float(entry.get("physics", {}).get("mass", .5)), .5)
                            obj = wrapper(name, kind.lower(), effective_mass if effective_mass > 0 else .05, entry["scale"])
                            obj._prim_path, obj.usd_path = path, str(source)
                            objects[kind].append(obj)
                table_info = layout["Table"]
                table = wrapper("material_0122_0_0", "table", scale=table_info["scale"])
                table._prim_path = "/World/envs/env_0/Table/material_0122/material_0122_0_0"
                mesh = UsdGeom.Mesh.Define(stage, table._prim_path)
                cube = trimesh.creation.box(extents=[1, 1, 1])
                mesh.CreatePointsAttr(cube.vertices.tolist())
                mesh.CreateFaceVertexCountsAttr([3] * len(cube.faces))
                mesh.CreateFaceVertexIndicesAttr(cube.faces.reshape(-1).tolist())
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexDecomposition")
                transform(mesh.GetPrim(), table_info["default_pos"], table_info["default_ori"], table_info["scale"])
                scene = manager(objects["Rigid"], objects["Geometry"], table)
                scene.stage = stage
                robots = SimpleNamespace(robot_list=[], robot_key=[], num_envs=1)
                for index, configuration in enumerate(robot_config):
                    path = f"/World/envs/env_0/robot{index}"
                    source = repo / "Assets/Robots" / configuration["robot_name"] / ("ARX.usda" if index == 0 else "ARX.usd")
                    prim = stage.DefinePrim(path, "Xform")
                    prim.GetReferences().AddReference(str(source))
                    transform(prim, configuration["default_root_pos"], configuration["default_root_rot"])
                    robots.robot_list.append(SimpleNamespace())
                    robots.robot_key.append(SimpleNamespace(cfg=SimpleNamespace(
                        prim_path=path, spawn=SimpleNamespace(usd_path=str(source)))))
                exclusions = [sensor_config[side]["gelpad_prim_path"].replace("env_.*", "env_0")
                              for side in ("left", "right")]
                for path in exclusions:
                    self.assertTrue(stage.GetPrimAtPath(path).IsValid())
                for key in robots.robot_key:
                    key.body_names = [str(prim.GetName()) for prim in Usd.PrimRange(stage.GetPrimAtPath(key.cfg.prim_path))
                                      if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                                      and not any(prim.GetPath().HasPrefix(Sdf.Path(path)) for path in exclusions)]
                extras = collect_robot_colliders(robots, stage, exclude_prim_paths=exclusions)
                self.assertEqual(len(extras), 22)
                self.assertEqual(sum(obj.body_name in ("gelsight_left", "gelsight_right") for obj in extras), 0)
                original = stage.GetRootLayer().ExportToString()
                prepared = prepare_scene(scene, cache_dir=cache_dir, extra_objects=extras, allow_surface_repair=True)
                expected = len(objects["Rigid"]) + len(objects["Geometry"]) + 1 + 22
                self.assertEqual(len(prepared.specs), expected)
                self.assertEqual(sum(not spec["kinematic"] for spec in prepared.specs), len(objects["Rigid"]))
                for spec in prepared.specs:
                    binding = prepared.bindings[spec["name"]]
                    if binding.scene_object.dynamic:
                        self.assertAlmostEqual(spec["mass_density"] * binding.volume, binding.scene_object.mass)
                stage.RemovePrim(prepared.root_path)
                self.assertEqual(stage.GetRootLayer().ExportToString(), original)
                with patch("env.tactile.scene.tetrahedralize_surface", side_effect=AssertionError("cache miss")):
                    repeated = prepare_scene(scene, cache_dir=cache_dir, extra_objects=extras, allow_surface_repair=True)
                self.assertEqual({binding.cache_path for binding in prepared.bindings.values()},
                                 {binding.cache_path for binding in repeated.bindings.values()})
                report = {"task": task, "status": "passed", "objects": expected, "robot_bodies": len(extras),
                          "dynamic_objects": len(objects["Rigid"]), "elapsed_seconds": time.monotonic() - started,
                          "tetrahedra": sum(binding.provenance["tetrahedra"] for binding in prepared.bindings.values()),
                          "cache_dir": str(cache_dir), "second_prepare_cache_only": True}
                reports.append(report)
                print("NATIVE_PREFLIGHT", json.dumps(report), flush=True)
        destination = Path(os.environ.get("ROBODOJO_TACTILE_PREFLIGHT_REPORT", "/tmp/robodojo_native_scene_preflight.json"))
        destination.write_text(json.dumps(reports, indent=2))


class NearConvexFanConversionTests(unittest.TestCase):
    def test_near_convex_source_gets_unreduced_fan_label_and_unchanged_geometry(self):
        source = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
        points, faces = (
            np.asarray(source.vertices).copy(),
            np.asarray(source.faces).copy(),
        )
        top = np.all(points[faces, 2] == 0.01, axis=1)
        perimeter = sorted(
            [index for index, point in enumerate(points) if point[2] == 0.01],
            key=lambda index: np.arctan2(points[index, 1], points[index, 0]),
        )
        points = np.vstack([points, [0.0, 0.0, 0.01 - 2e-8]])
        faces = np.vstack(
            [
                faces[~top],
                [
                    [perimeter[index], perimeter[(index + 1) % 4], 8]
                    for index in range(4)
                ],
            ]
        )
        source = trimesh.Trimesh(points, faces, process=False)
        self.assertTrue(source.is_watertight)
        self.assertTrue(source.is_convex)  # The approximate classifier is insufficient.
        original_vertices, original_faces = source.vertices.copy(), source.faces.copy()
        expected_points = np.vstack([original_vertices, original_vertices.mean(axis=0)])
        expected_tets = np.column_stack(
            [original_faces, np.full(len(original_faces), len(original_vertices))]
        )
        expected = validate_tetrahedra(expected_points, expected_tets)
        result = tetrahedralize_surface(source)
        self.assertEqual(result[3]["method"], "source_centre_fan")
        for index in range(3):
            np.testing.assert_array_equal(result[index], expected[index])
        self.assertEqual(result[3]["volume_m3"], expected[3])
        np.testing.assert_array_equal(source.vertices, original_vertices)
        np.testing.assert_array_equal(source.faces, original_faces)


@unittest.skipIf(Usd is None, "Standalone USD bindings unavailable")
class SourceFanGeometryTests(unittest.TestCase):
    def test_actual_slot_source_fan_keeps_original_points_tets_and_boundary(self):
        from env.tactile.scene import SceneObject, _read_collision_surface

        source = (
            Path(__file__).resolve().parents[2]
            / "Assets/Object/RoboDojo/Geometry/test_tube_slot/00000/object.usdz"
        )
        if not source.is_file():
            self.skipTest("Actual tube-slot source unavailable")
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/Asset", "Xform")
        root.GetReferences().AddReference(str(source))
        descriptor = SceneObject(
            0,
            "slot",
            "/Asset",
            "geometry",
            False,
            None,
            (1.0, 1.0, 1.0),
            str(source),
            None,
        )
        surface, _, _ = _read_collision_surface(stage, descriptor)
        part = next(
            piece
            for piece in surface.split(only_watertight=False, repair=False)
            if len(piece.vertices) == 40 and len(piece.faces) == 76
        )
        original_vertices, original_faces = part.vertices.copy(), part.faces.copy()
        # The valid original fan, before its optional hull-reduction label.
        expected_points = np.vstack([original_vertices, original_vertices.mean(axis=0)])
        expected_tets = np.column_stack(
            [original_faces, np.full(len(original_faces), len(original_vertices))]
        )
        expected = validate_tetrahedra(expected_points, expected_tets)
        result = tetrahedralize_surface(part, allow_surface_repair=True)
        self.assertEqual(result[3]["method"], "source_centre_fan")
        for index in range(3):
            np.testing.assert_array_equal(result[index], expected[index])
        np.testing.assert_array_equal(part.vertices, original_vertices)
        np.testing.assert_array_equal(part.faces, original_faces)
        self.assertEqual(result[3]["volume_m3"], expected[3])


@unittest.skipIf(Usd is None, "Standalone USD bindings unavailable")
class SourceMeshGeometryTests(unittest.TestCase):
    def test_actual_coin_preserves_all_source_facets_and_native_mass_for_each_instance(self):
        from omegaconf import DictConfig, OmegaConf
        from scipy.spatial import cKDTree

        from env.tactile.config import load_tactile_config

        if importlib.util.find_spec("tetgen") is None:
            self.skipTest("Optional TetGen dependency unavailable")
        repo = Path(__file__).resolve().parents[2]
        relative = "Assets/Object/RoboDojo/Rigid/coin/00000/object.usdz"
        source = repo / relative
        layout = repo / "Assets/Eval_Layout/RoboDojo/arx_x5/0/deposit_coin_0.json"
        if not source.is_file() or not layout.is_file():
            self.skipTest("Actual coin USD/layout unavailable")
        entry = json.loads(layout.read_text())["Rigid"]["coin"][0]
        self.assertEqual(entry["physics"]["mass"], .005)
        source_meshes = OmegaConf.create({relative: {"prim_path": "visual", "method": "tetgen_plc"}})
        self.assertIsInstance(source_meshes, DictConfig)
        self.assertIsInstance(source_meshes[relative], DictConfig)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        objects = []
        for index in range(2):
            obj = wrapper(f"coin{index}", mass=entry["physics"]["mass"], scale=entry["scale"])
            obj.usd_path = str(source)
            root = UsdGeom.Xform.Define(stage, obj._prim_path)
            root.GetPrim().GetReferences().AddReference(str(source))
            root.ClearXformOpOrder()
            position = np.asarray(entry["default_pos"]) + [index * .1, 0, 0]
            root.AddTranslateOp(opSuffix="testPlacement").Set(Gf.Vec3d(*position))
            quaternion = entry["default_ori"]
            root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, opSuffix="testPlacement").Set(
                Gf.Quatd(quaternion[0], Gf.Vec3d(*quaternion[1:])))
            objects.append(obj)
        visual = UsdGeom.Mesh(stage.GetPrimAtPath(objects[0]._prim_path + "/visual"))
        counts = np.asarray(visual.GetFaceVertexCountsAttr().Get())
        self.assertEqual(len(counts), 2684)
        self.assertTrue(np.all(counts == 3))
        raw_faces = np.asarray(visual.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        relative_transform, reset = UsdGeom.XformCache().ComputeRelativeTransform(
            visual.GetPrim(), stage.GetPrimAtPath(objects[0]._prim_path))
        self.assertFalse(reset)
        transform = np.asarray(relative_transform).T
        raw_points = np.asarray(visual.GetPointsAttr().Get(), dtype=np.float64)
        raw_points = (raw_points @ transform[:3, :3].T + transform[:3, 3]) * entry["scale"]
        unique, raw_to_unique = np.unique(raw_points, axis=0, return_inverse=True)
        triangles = raw_points[raw_faces]
        self.assertTrue(np.all(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                                     triangles[:, 2] - triangles[:, 0]), axis=1) > 0))

        def oriented_faces(faces):
            canonical = np.take_along_axis(faces, (faces.argmin(axis=1)[:, None] + np.arange(3)) % 3, axis=1)
            return canonical[np.lexsort(canonical.T[::-1])]

        scene = manager(objects)
        scene.stage = stage
        original = stage.GetRootLayer().ExportToString()
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_scene(scene, cache_dir=directory,
                                     source_meshes=source_meshes)
            self.assertEqual(len(prepared.specs), 2)
            self.assertEqual(len({binding.cache_path for binding in prepared.bindings.values()}), 1)
            for spec in prepared.specs:
                binding = prepared.bindings[spec["name"]]
                self.assertFalse(spec["kinematic"])
                self.assertAlmostEqual(spec["mass_density"] * binding.volume, .005, places=15)
                operations = binding.provenance["source_mesh_policy"]["canonicalization"]
                self.assertEqual(operations["removed_faces"], [])
                self.assertEqual(operations["maximum_vertex_shift_m"], 0)
                self.assertEqual(operations["raw_to_canonical_face"], list(range(2684)))
                with np.load(binding.cache_path, allow_pickle=False) as stored:
                    points, _, boundary, actual_volume = validate_tetrahedra(stored["points"], stored["tets"])
                np.testing.assert_allclose(actual_volume, binding.volume, rtol=1e-10, atol=1e-18)
                self.assertAlmostEqual(spec["mass_density"] * actual_volume, .005, places=15)
                boundary_nodes = np.unique(boundary)
                errors, ids = cKDTree(unique).query(points[boundary_nodes])
                self.assertEqual(len(boundary_nodes), len(unique))
                self.assertEqual(len(np.unique(ids)), len(unique))
                self.assertLessEqual(errors.max(), 1e-12)
                mapping = np.full(len(points), -1, dtype=np.int64)
                mapping[boundary_nodes] = ids
                np.testing.assert_array_equal(oriented_faces(mapping[boundary]),
                                              oriented_faces(raw_to_unique[raw_faces]))
            stage.RemovePrim(prepared.root_path)
        self.assertEqual(stage.GetRootLayer().ExportToString(), original)


if __name__ == "__main__":
    unittest.main()
