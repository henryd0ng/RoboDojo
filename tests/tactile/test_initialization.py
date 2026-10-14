"""CPU geometry checks for bounded initialization branch selection."""

from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull

from env.tactile import initialization as init


def fixture(*, obstacle=False, moving_name="moving", compound=False):
    # Millimetre boxes: closest L/R faces require opposite x translations.
    def body(name, dynamic, bounds):
        vertices = np.asarray(list(product(*bounds))) * 1e-3
        piece = init.ConvexPiece.from_vertices(vertices)
        return init.Body(name, dynamic, [piece], np.eye(4).reshape(-1).tolist(), "fixture")

    moving = body(moving_name, True, [(-.1, .1)] * 3)
    if compound:
        moving.pieces.append(moving.pieces[0].translated([0, 0, 1e-6]))
    bodies = [moving,
              body("left", False, [(-.25, -.05), (-.15, .05), (-1, 1)]),
              body("right", False, [(.05, .25), (-1, 1), (-.15, .05)])]
    if obstacle:
        # Initially clear; the first +yz proposal intersects it. Static boxes
        # remain mutually clear, so checking all six pairs is meaningful.
        bodies.append(body("obstacle", False, [(-.04, .04), (-.3, .07), (.2, .4)]))
    return bodies, list(combinations([body.name for body in bodies], 2))


class InitializationBranchTests(unittest.TestCase):
    def assert_fully_checked(self, report, bodies, pairs, bound):
        self.assertEqual(report["status"], "feasible_proposal")
        by_name = {body.name: body for body in bodies}
        for body in bodies:
            delta = np.asarray(report["bodies"][body.name]["delta_xyz_m"])
            self.assertLessEqual(np.linalg.norm(delta), bound)
            if not body.dynamic:
                np.testing.assert_array_equal(delta, np.zeros(3))
            effective = np.asarray(report["bodies"][body.name]["effective_matrix"])
            np.testing.assert_array_equal(effective[:3, :3], np.eye(3))
            np.testing.assert_array_equal(effective[:3, 3], delta)
        for left, right in pairs:
            violations, _ = init.pair_witnesses(
                by_name[left], by_name[right],
                np.asarray(report["bodies"][left]["delta_xyz_m"]),
                np.asarray(report["bodies"][right]["delta_xyz_m"]),
                clearance=report["clearance_m"], tolerance=report["tolerance_m"])
            self.assertEqual(violations, [], (left, right))

    def test_opposite_closest_faces_fallback_and_generator_inputs(self):
        for generators in (False, True):
            with self.subTest(generators=generators):
                bodies, pairs = fixture()
                before = [piece.vertices.copy() for body in bodies for piece in body.pieces]
                strict = init.project(bodies, pairs)
                self.assertEqual(strict["status"], "strict_initial_intersection")
                self.assertEqual(len(strict["branch_attempts"]), 1)
                inputs = ((body for body in bodies), (iter(pair) for pair in pairs)) if generators else (bodies, pairs)
                report = init.project(*inputs, enabled=True, bound=.0003)
                self.assertEqual([attempt["status"] for attempt in report["branch_attempts"]],
                                 ["bounded_branch_infeasible", "feasible_proposal"])
                np.testing.assert_allclose(report["branch_attempts"][1]["preferred_direction"],
                                           [0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
                np.testing.assert_allclose(report["bodies"]["moving"]["delta_xyz_m"],
                                           [0, .000152, .000152], atol=1e-12)
                self.assert_fully_checked(report, bodies, pairs, .0003)
                for old, piece in zip(before, [piece for body in bodies for piece in body.pieces]):
                    np.testing.assert_array_equal(old, piece.vertices)

    def test_dynamic_first_and_piece_swap_preserve_ray(self):
        bodies, pairs = fixture(moving_name="a_moving", compound=True)
        # The dynamic owner sorts first and has more pieces: pair_witnesses
        # recurses with exchanged owners, which must reverse the preferred ray.
        report = init.project(bodies, pairs, enabled=True, bound=.0003)
        self.assertEqual(len(report["branch_attempts"]), 2)
        np.testing.assert_allclose(report["bodies"]["a_moving"]["delta_xyz_m"],
                                   [0, .000152, .000152], atol=1e-12)
        self.assert_fully_checked(report, bodies, pairs, .0003)

    def test_new_pair_is_checked_after_translation_and_iteration_limit_falls_back(self):
        bodies, pairs = fixture(obstacle=True)
        report = init.project(bodies, pairs, enabled=True, bound=.0003)
        self.assertEqual(len(report["branch_attempts"]), 2)
        selected = report["branch_attempts"][1]["iterations"]
        obstacle_records = [record for record in selected if set(record["pair"]) == {"moving", "obstacle"}]
        self.assertEqual([record["sat_violations"] for record in obstacle_records], [0, 1, 0])
        np.testing.assert_allclose(report["bodies"]["moving"]["delta_xyz_m"],
                                   [0, .000172, .000152], atol=1e-12)
        self.assert_fully_checked(report, bodies, pairs, .0003)
        limited = init.project(bodies, pairs, enabled=True, bound=.0003, max_iterations=1)
        self.assertEqual(limited["branch_attempts"][1]["status"], "iteration_limit")
        self.assertGreater(len(limited["branch_attempts"]), 2)
        self.assert_fully_checked(limited, bodies, pairs, .0003)

    def test_all_branches_fail_without_pose_or_budget_accumulation(self):
        bodies, pairs = fixture()
        original_minimize = init.minimize
        starts = []

        def capture_start(function, x0, **kwargs):
            starts.append(np.array(x0, copy=True))
            return original_minimize(function, x0, **kwargs)

        with patch.object(init, "minimize", side_effect=capture_start):
            rejected = init.project(bodies, pairs, enabled=True, bound=.00001)
        self.assertEqual(rejected["status"], "bounded_branch_infeasible")
        self.assertEqual(len(rejected["branch_attempts"]), 15)
        self.assertEqual(len(starts), 15)
        for start in starts:
            np.testing.assert_array_equal(start, np.zeros(3))
        for attempt in rejected["branch_attempts"]:
            self.assertEqual(attempt["bound_m"], .00001)
            self.assertEqual(attempt["start_state"], "original_native_zero_delta")
            self.assertEqual({tuple(record["pair"]) for record in attempt["iterations"]},
                             {tuple(sorted(pair)) for pair in pairs})
        for body in rejected["bodies"].values():
            self.assertNotIn("effective_matrix", body)
            self.assertNotIn("delta_xyz_m", body)


def _tet_volume(points, tets):
    return float(
        np.abs(np.linalg.det(points[tets[:, 1:]] - points[tets[:, 0, None]])).sum() / 6
    )


def _cache_leaf(method, points, tets):
    return {
        "method": method,
        "vertices": len(points),
        "tetrahedra": len(tets),
        "volume_m3": _tet_volume(points, tets),
        "source_to_tet_sampled_max_m": 0.0,
        "tet_to_source_sampled_max_m": 0.0,
        "max_surface_error_m": 0.0003,
    }


def _nested_tets():
    points = np.asarray(
        [
            [0, 0, 0],
            [0.01, 0, 0],
            [0, 0.01, 0],
            [0, 0, 0.01],
            [0.05, 0, 0],
            [0.06, 0, 0],
            [0.05, 0.01, 0],
            [0.05, 0, 0.01],
        ],
        dtype=float,
    )
    tets = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    first = _cache_leaf(
        "wildmeshing", points[:4], np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    )
    second = _cache_leaf(
        "wildmeshing", points[4:], np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    )
    nested = _cache_leaf("oriented_source_components", points, tets)
    nested["parts"] = [first, second]
    metrics = _cache_leaf("author_collider_compound", points, tets)
    metrics["parts"] = [nested]
    return (points, tets, metrics)


def _near_convex_fan():
    corners = np.asarray(
        [[x, y, z] for x in [-0.01, 0.01] for y in [-0.01, 0.01] for z in [-0.01, 0.01]]
    )
    hull = ConvexHull(corners)
    faces = hull.simplices.copy()
    triangles = corners[faces]
    winding = np.einsum(
        "ij,ij->i",
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        hull.equations[:, :3],
    )
    faces[winding < 0] = faces[winding < 0][:, [0, 2, 1]]
    top = np.all(corners[faces, 2] == 0.01, axis=1)
    perimeter = sorted(
        [index for index, point in enumerate(corners) if point[2] == 0.01],
        key=lambda index: np.arctan2(corners[index, 1], corners[index, 0]),
    )
    source = np.vstack([corners, [0.0, 0.0, 0.01 - 2e-08]])
    faces = np.vstack(
        [
            faces[~top],
            [[perimeter[index], perimeter[(index + 1) % 4], 8] for index in range(4)],
        ]
    )
    points = np.vstack([source, source.mean(axis=0)])
    tets = np.column_stack([faces, np.full(len(faces), 9)]).astype(np.int64)
    negative = np.linalg.det(points[tets[:, 1:]] - points[tets[:, 0, None]]) < 0
    tets[negative, 0], tets[negative, 1] = (
        tets[negative, 1].copy(),
        tets[negative, 0].copy(),
    )
    assert not np.isclose(
        ConvexHull(points).volume, _tet_volume(points, tets), rtol=1e-08, atol=1e-14
    )
    return (points, tets)


class CacheFanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity = "{}"
        cls.temp = tempfile.TemporaryDirectory(prefix="cache-fan-test-")
        cls.scratch = Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_source_fan_declared_unreferenced_vertex_is_rejected(self):
        points = np.asarray(
            [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01], [0.05, 0.05, 0.05]]
        )
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        metrics = _cache_leaf("source_centre_fan", points, tets)
        path = self._write_cache("unreferenced-fan-vertex", points, tets, metrics)
        with self.assertRaisesRegex(ValueError, "cover its declared vertices"):
            self._read_cache(init, path)

    def _write_cache(self, label, points, tets, metrics):
        path = self.scratch / (label + ".npz")
        np.savez_compressed(
            path,
            points=np.asarray(points),
            tets=np.asarray(tets),
            metrics=json.dumps(metrics),
            identity=self.identity,
        )
        return path

    def _read_cache(self, module, path):
        return module.body_from_cache("cpu_fixture", path, np.eye(4), dynamic=False)

    def _assert_corruption_rejected(self, path):
        with self.assertRaises(ValueError):
            self._read_cache(init, path)

    def test_legacy_exact_label_keeps_original_near_convex_fan(self):
        points, tets = _near_convex_fan()
        path = self._write_cache(
            "legacy-exact-near-convex",
            points,
            tets,
            _cache_leaf("exact_convex_centre_fan", points, tets),
        )
        original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        body = self._read_cache(init, path)
        self.assertEqual(body.pieces.blocks[0][:3], (0, len(tets), False))
        np.testing.assert_array_equal(body.pieces.vertices, points)
        np.testing.assert_array_equal(body.pieces.tets, tets)
        for index in range(len(tets)):
            np.testing.assert_array_equal(
                body.pieces[index].vertices, points[tets[index]]
            )
        self.assertEqual(body.cache_sha256, original_sha)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_sha)

    def test_source_fan_label_keeps_original_tetrahedra(self):
        points, tets = _near_convex_fan()
        path = self._write_cache(
            "source-near-convex",
            points,
            tets,
            _cache_leaf("source_centre_fan", points, tets),
        )
        body = self._read_cache(init, path)
        self.assertEqual(body.pieces.blocks[0][:3], (0, len(tets), False))
        np.testing.assert_array_equal(body.pieces.vertices, points)
        np.testing.assert_array_equal(body.pieces.tets, tets)

    def test_genuinely_exact_box_retains_convex_hull_optimization(self):
        corners = np.asarray(
            [
                [x, y, z]
                for x in [-0.001, 0.001]
                for y in [-0.001, 0.001]
                for z in [-0.001, 0.001]
            ]
        )
        points = np.vstack([corners, np.zeros((1, 3))])
        faces = ConvexHull(corners).simplices
        tets = np.column_stack([np.full(len(faces), 8, dtype=np.int64), faces]).astype(
            np.int64
        )
        negative = np.linalg.det(points[tets[:, 1:]] - points[tets[:, 0, None]]) < 0
        tets[negative, 1], tets[negative, 2] = (
            tets[negative, 2].copy(),
            tets[negative, 1].copy(),
        )
        metrics = _cache_leaf("exact_convex_centre_fan", points, tets)
        path = self._write_cache("exact-box", points, tets, metrics)
        body = self._read_cache(init, path)
        self.assertEqual(len(body.pieces), 1)
        self.assertEqual(body.pieces.blocks[0][:3], (0, 1, True))
        self.assertEqual(len(body.pieces[0].vertices), 8)
        np.testing.assert_array_equal(body.pieces.tets, tets)
        np.testing.assert_array_equal(body.pieces.vertices, points)

    def test_valid_nested_partition_is_supported(self):
        points, tets, metrics = _nested_tets()
        path = self._write_cache("valid-nested-partition", points, tets, metrics)
        body = self._read_cache(init, path)
        self.assertEqual(len(body.pieces), 2)
        np.testing.assert_array_equal(body.pieces.tets, tets)

    def test_nested_leaf_volume_changed_global_volume_preserved_is_rejected(self):
        points, tets, metrics = _nested_tets()
        global_volume = metrics["volume_m3"]
        metrics["parts"][0]["parts"][0]["volume_m3"] *= 1.01
        self.assertEqual(metrics["volume_m3"], global_volume)
        path = self._write_cache("corrupt-nested-leaf-volume", points, tets, metrics)
        self._assert_corruption_rejected(path)

    def test_cross_leaf_tet_indices_are_rejected(self):
        points, tets, metrics = _nested_tets()
        tets = tets[::-1].copy()
        self.assertAlmostEqual(
            _tet_volume(points, tets), metrics["volume_m3"], places=18
        )
        path = self._write_cache("corrupt-cross-leaf-indices", points, tets, metrics)
        self._assert_corruption_rejected(path)

    def test_duplicate_tets_with_consistent_global_volume_are_rejected(self):
        points = np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
        tets = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
        metrics = _cache_leaf("wildmeshing", points, tets)
        path = self._write_cache("corrupt-duplicate-tets", points, tets, metrics)
        self._assert_corruption_rejected(path)

    def test_nonmanifold_three_tets_on_one_face_are_rejected(self):
        points = np.asarray(
            [
                [0, 0, 0],
                [0.01, 0, 0],
                [0, 0.01, 0],
                [0, 0, 0.01],
                [0, 0, -0.01],
                [0.003, 0.003, 0.02],
            ]
        )
        tets = np.asarray([[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]], dtype=np.int64)
        metrics = _cache_leaf("wildmeshing", points, tets)
        path = self._write_cache("corrupt-nonmanifold-face", points, tets, metrics)
        self._assert_corruption_rejected(path)


if __name__ == "__main__":
    unittest.main()
