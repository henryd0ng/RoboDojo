"""Validate and commit bounded rigid initialization before the first world.init.

Tetrahedra are treated as a union of closed convex pieces. SAT therefore catches
surface crossings and full containment, including edge/edge-only intersections.
An authored exact-convex centre fan may be reduced to its equivalent convex hull;
other parts remain individual tetrahedra. No convex approximation of a concave
part is allowed. The numerical projector is pure data; the adapter commits only
task rigid transforms while physics is paused, before ownership transfer.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class ConvexPiece:
    vertices: np.ndarray
    normals: np.ndarray
    edges: np.ndarray

    @classmethod
    def from_vertices(cls, vertices):
        vertices = np.asarray(vertices, dtype=np.float64)
        if len(vertices) == 4:
            if abs(np.linalg.det(vertices[1:] - vertices[0])) <= 1e-20:
                raise ValueError("Degenerate tetrahedron cannot form a convex solid")
            normals = canonical_axes(np.asarray([np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
                                                  for a, b, c in combinations(range(4), 3)]))
            edges = canonical_axes(np.asarray([vertices[b] - vertices[a] for a, b in combinations(range(4), 2)]))
            return cls(vertices, normals, edges)
        hull = ConvexHull(vertices)
        vertices = vertices[hull.vertices]
        hull = ConvexHull(vertices)
        normals = canonical_axes(hull.equations[:, :3])
        indices = sorted({tuple(sorted(edge)) for face in hull.simplices
                          for edge in combinations(face, 2)})
        edges = canonical_axes(np.asarray([vertices[b] - vertices[a] for a, b in indices]))
        return cls(vertices, normals, edges)

    def translated(self, delta):
        return ConvexPiece(self.vertices + delta, self.normals, self.edges)


def canonical_axes(axes):
    """Normalize and deduplicate unoriented axes without discarding real axes."""
    axes = np.asarray(axes, dtype=np.float64).reshape(-1, 3)
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-30] / lengths[lengths > 1e-30, None]
    for axis in axes:
        index = int(np.flatnonzero(np.abs(axis) > 1e-13)[0])
        if axis[index] < 0:
            axis *= -1
    # Rounding only deduplicates axes; the first unrounded axis is retained.
    _, indices = np.unique(np.round(axes, 12), axis=0, return_index=True)
    return axes[np.sort(indices)]


def sat_witness(left, right, *, clearance=0.0, tolerance=1e-10, preferred_direction=None):
    """Return a sufficient separating translation, or None if already clear.

    For convex polyhedra SAT uses every face normal and every edge cross product.
    The selected direction is a local branch choice, not a global optimum for a
    union of nonconvex pieces. Numerical tolerance is always reported by caller.
    """
    cross = np.cross(left.edges[:, None, :], right.edges[None, :, :]).reshape(-1, 3)
    axes = canonical_axes(np.vstack([left.normals, right.normals, cross]))
    left_intervals = left.vertices @ axes.T
    right_intervals = right.vertices @ axes.T
    left_min, left_max = left_intervals.min(0), left_intervals.max(0)
    right_min, right_max = right_intervals.min(0), right_intervals.max(0)
    positive = right_max - left_min + clearance
    negative = left_max - right_min + clearance
    if np.any(positive <= tolerance) or np.any(negative <= tolerance):
        return None
    if preferred_direction is not None:
        # Choose a sufficient SAT halfspace encountered along one fixed ray,
        # instead of independently selecting possibly opposite closest faces.
        direction = np.asarray(preferred_direction, dtype=float)
        norm = np.linalg.norm(direction)
        if direction.shape != (3,) or not np.isfinite(direction).all() or norm == 0:
            raise ValueError("Finite nonzero three-axis direction required")
        planes, depths = np.vstack([axes, -axes]), np.r_[positive, negative]
        alignment = planes @ (direction / norm)
        choices = np.flatnonzero(alignment > 1e-9)
        if not len(choices):
            raise ValueError("No SAT halfspace aligns with the preferred direction")
        selected = choices[np.argmin(depths[choices] / alignment[choices])]
        return planes[selected], float(max(0.0, depths[selected]))
    positive_index, negative_index = int(positive.argmin()), int(negative.argmin())
    if positive[positive_index] <= negative[negative_index]:
        return axes[positive_index], float(max(0.0, positive[positive_index]))
    return -axes[negative_index], float(max(0.0, negative[negative_index]))


@dataclass
class Body:
    name: str
    dynamic: bool
    pieces: list[ConvexPiece]
    nominal_pose: list[float]
    cache_sha256: str
    world_bounds: tuple | None = None
    world_vertices: np.ndarray | None = None


class LazyPieces:
    """Construct convex pieces only for bodies surviving the pair broad phase."""

    def __init__(self, vertices, tets):
        self.vertices, self.tets = vertices, tets
        self.blocks, self.cache, self.count = [], {}, 0
        self._bounds = None

    def add_tetrahedra(self, indices):
        self.blocks.append((self.count, len(indices), False, indices))
        self.count += len(indices)

    def add_convex(self, vertices):
        self.blocks.append((self.count, 1, True, vertices))
        self.count += 1

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if index < 0 or index >= self.count:
            raise IndexError(index)
        if index not in self.cache:
            start, _, convex, value = next(block for block in self.blocks if block[0] <= index < block[0] + block[1])
            self.cache[index] = ConvexPiece.from_vertices(value if convex else self.vertices[self.tets[value[index - start]]])
        return self.cache[index]

    def certified_convex(self):
        return [self[start] for start, _, convex, _ in self.blocks if convex]

    def bounds(self):
        """Compute original tet bounds in bounded chunks, without normals."""
        if self._bounds is None:
            low, high = np.empty((self.count, 3)), np.empty((self.count, 3))
            for start, count, convex, value in self.blocks:
                if convex:
                    low[start], high[start] = value.min(0), value.max(0)
                    continue
                for offset in range(0, count, 4096):
                    stop = min(offset + 4096, count)
                    coordinates = self.vertices[self.tets[value[offset:stop]]]
                    low[start + offset:start + stop] = coordinates.min(1)
                    high[start + offset:start + stop] = coordinates.max(1)
            self._bounds = low, high
        return self._bounds


@dataclass(frozen=True)
class Ground:
    """Existing implicit halfspace; allowed space is normal dot x >= offset."""

    name: str
    normal: np.ndarray
    offset: float
    dynamic: bool = False
    nominal_pose: tuple = tuple(np.eye(4).reshape(-1))
    cache_sha256: str = "implicit_ground_no_mesh"

    def __post_init__(self):
        if (np.asarray(self.normal).shape != (3,) or not np.isfinite(self.normal).all()
                or not np.isclose(np.linalg.norm(self.normal), 1.0, atol=1e-12)
                or not np.isfinite(self.offset) or self.dynamic):
            raise ValueError("Finite unit-normal fixed halfspace required")


def body_from_cache(name, path, matrix, *, dynamic):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        vertices, tets = archive["points"], archive["tets"]
        metrics = json.loads(str(archive["metrics"]))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Finite three-axis cache vertices required")
    if (tets.ndim != 2 or tets.shape[1] != 4 or not np.issubdtype(tets.dtype, np.integer)
            or len(tets) == 0 or tets.min() < 0 or tets.max() >= len(vertices)):
        raise ValueError("Valid nonempty integer tetrahedral topology required")
    determinant = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, 0, None]])
    # Same nonzero-volume gate already used by RoboDojo scene conversion.
    if not np.isfinite(determinant).all() or np.any(np.abs(determinant) <= 1e-20):
        raise ValueError("Degenerate cached tetrahedron; cannot define a closed solid")
    if not np.isclose(np.abs(determinant).sum() / 6, metrics["volume_m3"], rtol=1e-8, atol=1e-18):
        raise ValueError("Cache aggregate volume does not match provenance")
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or np.linalg.det(matrix[:3, :3]) <= 0):
        raise ValueError("Rigid proper finite world transform required; cache already bakes scale")
    world_vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    pieces = LazyPieces(world_vertices, tets)

    def append_part(part, start_vertex, start_tet):
        vertex_count, tet_count = part.get("vertices"), part.get("tetrahedra")
        volume = part.get("volume_m3")
        if (type(vertex_count) is not int or type(tet_count) is not int
                or vertex_count <= 0 or tet_count <= 0
                or start_vertex + vertex_count > len(vertices)
                or start_tet + tet_count > len(tets)
                or isinstance(volume, bool) or not isinstance(volume, (int, float))
                or not np.isfinite(volume) or volume <= 0):
            raise ValueError("Invalid cache part counts or volume")
        actual_volume = np.abs(determinant[start_tet:start_tet + tet_count]).sum() / 6
        if not np.isclose(actual_volume, volume, rtol=1e-8, atol=1e-18):
            raise ValueError("Cache part tetrahedral volume does not match provenance")
        if "parts" in part:
            vertex_cursor, tet_cursor = start_vertex, start_tet
            for child in part["parts"]:
                append_part(child, vertex_cursor, tet_cursor)
                vertex_cursor += child["vertices"]
                tet_cursor += child["tetrahedra"]
            if vertex_cursor != start_vertex + vertex_count or tet_cursor != start_tet + tet_count:
                raise ValueError("Nested compound provenance does not cover all cache geometry")
            return
        cells = tets[start_tet:start_tet + tet_count]
        if cells.min() < start_vertex or cells.max() >= start_vertex + vertex_count:
            raise ValueError("Cache part tetrahedral indices cross its vertex range")
        if len(np.unique(np.sort(cells, axis=1), axis=0)) != tet_count:
            raise ValueError("Duplicate cache part tetrahedron")
        faces = np.concatenate([cells[:, [0, 2, 1]], cells[:, [0, 1, 3]],
                                cells[:, [0, 3, 2]], cells[:, [1, 2, 3]]])
        _, face_indices, face_counts = np.unique(np.sort(faces, axis=1), axis=0,
                                               return_index=True, return_counts=True)
        if np.any(face_counts > 2):
            raise ValueError("Non-manifold cache part tetrahedral face")
        boundary = faces[face_indices[face_counts == 1]]
        edges = np.concatenate([boundary[:, [0, 1]], boundary[:, [1, 2]], boundary[:, [2, 0]]])
        _, edge_counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        if not len(boundary) or np.any(edge_counts != 2):
            raise ValueError("Cache part tetrahedral boundary is not closed manifold")
        if part["method"] in ("exact_convex_centre_fan", "source_centre_fan"):
            # A source fan is valid geometry even when it is not hull-equivalent.
            if (len(np.unique(cells)) != vertex_count
                    or not set.intersection(*(set(cell) for cell in cells))):
                raise ValueError("Cache centre fan does not cover its declared vertices")
        if part["method"] == "exact_convex_centre_fan":
            # Keep the strict certificate for the optional hull reduction.
            # Legacy is_convex labels can be broader; never change their tets.
            local = vertices[start_vertex:start_vertex + vertex_count]
            hull = ConvexHull(local)
            if np.isclose(hull.volume, actual_volume, rtol=1e-8, atol=1e-14):
                pieces.add_convex(world_vertices[start_vertex:start_vertex + vertex_count])
                return
        pieces.add_tetrahedra(range(start_tet, start_tet + tet_count))

    if "parts" in metrics:
        vertex_start = tet_start = 0
        for part in metrics["parts"]:
            append_part(part, vertex_start, tet_start)
            vertex_start += part["vertices"]
            tet_start += part["tetrahedra"]
        if vertex_start != len(vertices) or tet_start != len(tets):
            raise ValueError("Compound provenance does not cover all cache geometry")
    else:
        append_part(metrics, 0, 0)
    return Body(name, dynamic, pieces, matrix.reshape(-1).tolist(), hashlib.sha256(path.read_bytes()).hexdigest(),
                (world_vertices.min(0), world_vertices.max(0)), world_vertices[np.unique(tets)])


def _body_bounds(body):
    if body.world_bounds is not None:
        return body.world_bounds
    return (np.min([piece.vertices.min(0) for piece in body.pieces], axis=0),
            np.max([piece.vertices.max(0) for piece in body.pieces], axis=0))


def _piece_bounds(body):
    if isinstance(body.pieces, LazyPieces):
        return body.pieces.bounds()
    return (np.asarray([piece.vertices.min(0) for piece in body.pieces]),
            np.asarray([piece.vertices.max(0) for piece in body.pieces]))


def pair_witnesses(left, right, delta_left, delta_right, *, clearance, tolerance, preferred_direction=None):
    """AABB is broad phase only; every remaining candidate uses full SAT."""
    if isinstance(left, Ground) or isinstance(right, Ground):
        if isinstance(left, Ground) and isinstance(right, Ground):
            raise ValueError("Ground/ground must not be an enabled contact pair")
        ground, body = (left, right) if isinstance(left, Ground) else (right, left)
        delta = delta_right if isinstance(left, Ground) else delta_left
        normal = -ground.normal if isinstance(left, Ground) else ground.normal
        low, high = _body_bounds(body)
        lower_bound = np.where(ground.normal >= 0, low, high) @ ground.normal
        if lower_bound + delta @ ground.normal - ground.offset >= clearance - tolerance:
            return [], 0
        vertices = (body.world_vertices if body.world_vertices is not None else
                    np.concatenate([piece.vertices for piece in body.pieces]))
        gaps = vertices @ ground.normal + delta @ ground.normal - ground.offset
        index = int(gaps.argmin())
        if gaps[index] >= clearance - tolerance:
            return [], len(vertices)
        # One uniform translation applies to the whole owner; its smallest
        # vertex gap is the exact strongest ground constraint for the union.
        return [(index, 0, normal, float(clearance - gaps[index]))], len(vertices)
    left_bounds, right_bounds = _body_bounds(left), _body_bounds(right)
    if (np.any(left_bounds[0] + delta_left > right_bounds[1] + delta_right + clearance + tolerance)
            or np.any(right_bounds[0] + delta_right > left_bounds[1] + delta_left + clearance + tolerance)):
        return [], 0
    if len(left.pieces) > len(right.pieces):
        witnesses, count = pair_witnesses(right, left, delta_right, delta_left,
            clearance=clearance, tolerance=tolerance,
            preferred_direction=None if preferred_direction is None else -np.asarray(preferred_direction))
        return [(j, i, -normal, depth) for i, j, normal, depth in witnesses], count
    right_min, right_max = (value + delta_right for value in _piece_bounds(right))
    left_min, left_max = (value + delta_left for value in _piece_bounds(left))
    witnesses, candidate_count = [], 0
    for left_index, (lo, hi) in enumerate(zip(left_min, left_max)):
        candidates = np.flatnonzero(np.all(right_max >= lo - clearance - tolerance, axis=1)
                                    & np.all(right_min <= hi + clearance + tolerance, axis=1))
        candidate_count += len(candidates)
        if not len(candidates):
            continue
        piece = left.pieces[left_index].translated(delta_left)
        for right_index in candidates:
            witness = sat_witness(piece, right.pieces[right_index].translated(delta_right),
                                  clearance=clearance, tolerance=tolerance, preferred_direction=preferred_direction)
            if witness is not None:
                normal, depth = witness
                witnesses.append((left_index, int(right_index), normal, depth))
    return witnesses, candidate_count


def contained_vertex_lower_bound(dynamic, fixed):
    """Certified translation lower bound from fixed vertices inside convex parts.

    A fixed point at depth d remains inside that same moving convex solid under
    any translation with norm < d. This detects impossibility within the budget;
    it is not an estimate of the full minimum separation displacement.
    """
    moving_bounds, fixed_bounds = _body_bounds(dynamic), _body_bounds(fixed)
    if np.any(moving_bounds[0] > fixed_bounds[1]) or np.any(fixed_bounds[0] > moving_bounds[1]):
        return 0.0
    points = (fixed.world_vertices if fixed.world_vertices is not None else
              np.unique(np.concatenate([part.vertices for part in fixed.pieces]), axis=0))
    maximum = 0.0
    parts = dynamic.pieces.certified_convex() if isinstance(dynamic.pieces, LazyPieces) else dynamic.pieces
    for part in parts:
        # Large convex parts only: avoid constructing one hull per FEM tetra.
        if len(part.vertices) <= 4:
            continue
        hull = ConvexHull(part.vertices)
        lo, hi = part.vertices.min(0), part.vertices.max(0)
        candidates = points[np.all(points >= lo, axis=1) & np.all(points <= hi, axis=1)]
        if len(candidates):
            depth = -np.max(candidates @ hull.equations[:, :3].T + hull.equations[:, 3], axis=1)
            maximum = max(maximum, float(depth.max(initial=0.0)))
    return maximum


def project(bodies, enabled_pairs, *, enabled=False, bound=0.0003,
            clearance=2e-6, tolerance=1e-10, max_iterations=12):
    """Try closest SAT branches first, then a finite deterministic set of rays.

    Each attempt starts at the original native poses with the same total budget.
    Return the first fully checked feasible proposal, not a global optimum.
    Strict rejection and certified/fixed failures do not trigger fallback.
    """
    bodies = tuple(bodies)
    enabled_pairs = tuple(tuple(pair) for pair in enabled_pairs)
    diagonals = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    axes = np.eye(3)
    directions = [None] + [np.asarray(sign * np.asarray(ray), dtype=float) / np.linalg.norm(ray)
                           for ray in diagonals + list(axes) for sign in (1, -1)]
    attempts = []
    for direction in directions:
        report = _project_branch(bodies, enabled_pairs, enabled=enabled, bound=bound,
            clearance=clearance, tolerance=tolerance, max_iterations=max_iterations,
            preferred_direction=direction)
        attempts.append({"preferred_direction": None if direction is None else direction.tolist(),
                         "status": report["status"], "iterations": report["iterations"],
                         "solver_message": report.get("solver_message"),
                         "start_state": "original_native_zero_delta", "bound_m": bound})
        report["branch_attempts"] = attempts
        report["branch_selection"] = "closest first; first feasible fixed-ray branch; no global optimum claim"
        if not enabled or report["status"] not in {"bounded_branch_infeasible", "iteration_limit"}:
            return report
    return report


def _project_branch(bodies, enabled_pairs, *, enabled=False, bound=0.0003,
                    clearance=2e-6, tolerance=1e-10, max_iterations=12, preferred_direction=None):
    """Return a proposal; caller must independently validate before world.init.

    Default is strict rejection. Static bodies cannot move. A failed solve never
    returns a pose to apply. No task labels, names, gravity lifts, collision masks,
    collider editing, simulator calls, or rollout retries exist in this module.
    """
    if (not np.isfinite([bound, clearance, tolerance]).all() or bound < 0
            or (enabled and bound == 0) or clearance <= 0 or tolerance < 0):
        raise ValueError("Finite nonnegative budget; enabled correction requires a positive budget")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("Positive integer iteration limit required")
    by_name = {body.name: body for body in bodies}
    if len(by_name) != len(bodies):
        raise ValueError("Body names must be unique")
    enabled_pairs = sorted({tuple(sorted(pair)) for pair in enabled_pairs})
    if any(len(pair) != 2 or pair[0] == pair[1] or any(name not in by_name for name in pair) for pair in enabled_pairs):
        raise ValueError("Enabled pairs must identify distinct known bodies")
    moving = sorted(body.name for body in bodies if body.dynamic)
    indices = {name: i for i, name in enumerate(moving)}
    deltas = {name: np.zeros(3) for name in by_name}
    report = {"protocol": "bounded_translation_sat_candidate_v1", "enabled": enabled,
              "bound_m": bound, "clearance_m": clearance, "tolerance_m": tolerance,
              "enabled_pairs": list(enabled_pairs), "iterations": [], "status": "not_run",
              "bodies": {body.name: {"dynamic": body.dynamic, "native_prep_matrix": body.nominal_pose,
                                     "cache_sha256": body.cache_sha256} for body in bodies}}
    for left_name, right_name in enabled_pairs:
        left, right = by_name[left_name], by_name[right_name]
        if enabled and left.dynamic != right.dynamic:
            dynamic, fixed = (left, right) if left.dynamic else (right, left)
            if isinstance(fixed, Ground):
                continue
            lower_bound = contained_vertex_lower_bound(dynamic, fixed)
            if lower_bound > bound + tolerance:
                report.update(status="certified_over_budget", translation_lower_bound_m=lower_bound,
                              offending_pair=[left_name, right_name])
                return report
    constraint_rows, constraint_rhs = [], []
    for iteration in range(max_iterations + 1):
        violations = []
        for left_name, right_name in enabled_pairs:
            left, right = by_name[left_name], by_name[right_name]
            direction = (preferred_direction if left.dynamic or preferred_direction is None
                         else -preferred_direction)
            witnesses, candidate_count = pair_witnesses(left, right, deltas[left_name], deltas[right_name],
                clearance=clearance, tolerance=tolerance, preferred_direction=direction)
            violations.extend((left_name, right_name, normal, depth) for _, _, normal, depth in witnesses)
            report["iterations"].append({"iteration": iteration, "pair": [left_name, right_name],
                                         "broadphase_candidates": candidate_count, "sat_violations": len(witnesses)})
        if not violations:
            report["status"] = "clear_initial_state" if iteration == 0 else "feasible_proposal"
            for name, delta in deltas.items():
                effective = np.asarray(by_name[name].nominal_pose).reshape(4, 4).copy()
                effective[:3, 3] += delta
                report["bodies"][name].update(delta_xyz_m=delta.tolist(), delta_norm_m=float(np.linalg.norm(delta)),
                                              effective_matrix=effective.tolist())
            return report
        if not enabled:
            report["status"] = "strict_initial_intersection"
            return report
        if iteration == max_iterations:
            report["status"] = "iteration_limit"
            return report
        for left_name, right_name, normal, depth in violations:
            row = np.zeros(3 * len(moving))
            for name, sign in ((left_name, 1.0), (right_name, -1.0)):
                if name in indices:
                    index = indices[name]
                    row[index * 3:index * 3 + 3] += sign * normal
            if not np.any(row):
                report["status"] = "fixed_pair_intersection"
                return report
            rhs = depth + float(normal @ (deltas[left_name] - deltas[right_name]))
            constraint_rows.append(row)
            constraint_rhs.append(rhs)
        rows, rhs = np.asarray(constraint_rows), np.asarray(constraint_rhs)
        # Scale optimization to units of the declared correction budget.
        constraints = [{"type": "ineq", "fun": lambda x: rows @ x - rhs / bound,
                        "jac": lambda x: rows}]
        for index in range(len(moving)):
            constraints.append({"type": "ineq", "fun": lambda x, i=index: 1.0 - np.sum(x[i * 3:i * 3 + 3] ** 2)})
        result = minimize(lambda x: .5 * x @ x, np.concatenate([deltas[name] for name in moving]) / bound,
                          jac=lambda x: x, constraints=constraints, method="SLSQP",
                          options={"ftol": 1e-12, "maxiter": 100})
        if (not result.success or np.min(rows @ result.x - rhs / bound) < -1e-8
                or any(np.linalg.norm(result.x[i * 3:i * 3 + 3]) > 1.0 + 1e-8 for i in range(len(moving)))):
            report.update(status="bounded_branch_infeasible", solver_message=str(result.message))
            return report
        for name, index in indices.items():
            deltas[name] = result.x[index * 3:index * 3 + 3] * bound
            # SLSQP permits a normalized 1e-8 feasibility residue. Remove only
            # that numerical overshoot, then rerun every pair on the next pass.
            norm = np.linalg.norm(deltas[name])
            if norm > bound:
                deltas[name] *= np.nextafter(bound, 0.0) / norm
    raise AssertionError("Unreachable")


def _contact_element(mesh):
    attribute = mesh.meta().find("contact_element_id")
    return 0 if attribute is None else int(np.asarray(attribute.view()).reshape(-1)[0])


def _ground_state(scene):
    grounds = scene.objects().find("ground")
    if len(grounds) != 1 or len(grounds[0].geometries().ids()) != 1:
        raise RuntimeError("Expected exactly one original implicit ground")
    slot, _ = scene.geometries().find(int(grounds[0].geometries().ids()[0]))
    mesh = slot.geometry()
    normal = np.asarray(mesh.instances().find("N").view(), dtype=float).reshape(-1, 3)[0].copy()
    point = np.asarray(mesh.instances().find("P").view(), dtype=float).reshape(-1, 3)[0].copy()
    if not np.isfinite(normal).all() or not np.isfinite(point).all() or np.linalg.norm(normal) == 0:
        raise RuntimeError("Invalid implicit ground normal or point")
    fingerprint = hashlib.sha256(normal.tobytes() + point.tobytes()
                                 + str(_contact_element(mesh)).encode()).hexdigest()
    normal /= np.linalg.norm(normal)
    return mesh, normal, float(normal @ point), fingerprint


def _same_owner_state(mesh, table):
    attribute = mesh.meta().find("self_collision")
    enabled = False if attribute is None else bool(np.asarray(attribute.view()).reshape(-1)[0])
    element = _contact_element(mesh)
    model = table.at(element, element)
    return {"self_collision": enabled, "element_id": element, "contact_enabled": bool(model.is_enabled()),
            "friction": float(model.friction_rate()), "resistance": float(model.resistance())}


def _live_scene(manager):
    """Snapshot original scene geometry and table rows before world.init()."""
    from .scene import validate_tetrahedra

    bodies, meshes, elements = [], {}, {}
    for name, binding in manager._prepared_scene.bindings.items():
        rigid = manager.uipc_rigid_objects[name]
        mesh = rigid.geo_slot_list[0].geometry()
        matrix = np.asarray(mesh.transforms().view()).reshape(-1, 4, 4)[0].copy()
        body = body_from_cache(name, binding.cache_path, matrix,
                               dynamic=binding.scene_object.dynamic and binding.robot_key is None)
        with np.load(binding.cache_path, allow_pickle=False) as archive:
            actual_points = np.asarray(mesh.positions().view()).reshape(-1, 3)
            actual_tets = np.asarray(mesh.tetrahedra().topo().view()).reshape(-1, 4)
            if (actual_points.shape != archive["points"].shape
                    or not np.allclose(actual_points, archive["points"], rtol=0, atol=1e-12)
                    or not np.array_equal(actual_tets, archive["tets"])):
                raise RuntimeError(f"Initial UIPC mesh differs from validated cache: {name}")
        bodies.append(body)
        meshes[name] = mesh
        elements[name] = _contact_element(mesh)
    for name, sensor in manager.sensors.items():
        name = f"gel:{name}"
        mesh = sensor.gelpad.geo_slot_list[0].geometry()
        points, tets, _, _ = validate_tetrahedra(
            np.asarray(mesh.positions().view()).reshape(-1, 3),
            np.asarray(mesh.tetrahedra().topo().view()).reshape(-1, 4),
        )
        matrix = np.asarray(mesh.transforms().view()).reshape(-1, 4, 4)[0].copy()
        world = points @ matrix[:3, :3].T + matrix[:3, 3]
        geometry_sha = hashlib.sha256(points.tobytes() + tets.tobytes()).hexdigest()
        pieces = LazyPieces(world, tets)
        pieces.add_tetrahedra(range(len(tets)))
        bodies.append(Body(name, False, pieces, matrix.reshape(-1).tolist(), geometry_sha,
                           (world.min(0), world.max(0)), world[np.unique(tets)]))
        meshes[name] = mesh
        elements[name] = _contact_element(mesh)
    # The native adapter owns exactly task/robot ABDs, two gel FEMs and ground.
    # Unknown added geometries must be represented explicitly before projection.
    if manager.uipc_sim.scene.objects().size() != len(meshes) + 1:
        raise RuntimeError("Unexpected UIPC object inventory during initialization")
    ground_mesh, normal, offset, _ = _ground_state(manager.uipc_sim.scene)
    bodies.append(Ground("ground", normal, offset))
    elements["ground"] = _contact_element(ground_mesh)
    for name, mesh in meshes.items():
        # Preserve original FEM self-contact. Adjacent tetrahedra share faces;
        # the distinct-owner clearance projector must not constrain them.
        thickness = mesh.vertices().find("thickness")
        if thickness is not None and np.any(np.asarray(thickness.view()) != 0):
            raise RuntimeError(f"Initialization projection requires zero primitive thickness: {name}")
    table = manager.uipc_sim.scene.contact_tabular()
    pairs, pair_models = [], []
    for left, right in combinations(sorted(elements), 2):
        model = table.at(elements[left], elements[right])
        enabled = bool(model.is_enabled())
        pair_models.append({"pair": [left, right], "element_ids": [elements[left], elements[right]],
                            "enabled": enabled, "friction": float(model.friction_rate()),
                            "resistance": float(model.resistance())})
        if enabled:
            pairs.append((left, right))
    fingerprint = hashlib.sha256(json.dumps(pair_models, sort_keys=True).encode()).hexdigest()
    return bodies, meshes, pairs, pair_models, fingerprint


def prepare_initialization(manager, scene_manager):
    """Plan every correction before committing anything to the fresh world."""
    from .dynamics import _native_actor_view
    import resource
    import time

    started = time.perf_counter()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bodies, meshes, pairs, models, fingerprint = _live_scene(manager)
    maximum = manager.config.max_initial_pose_correction_m
    report = project(bodies, pairs, enabled=maximum > 0, bound=maximum)
    report.update(contact_table=models, contact_table_sha256=fingerprint,
                  distinct_owner_pair_inventory_complete=True,
                  complete_distinct_owner_pairs_checked=False,
                  native_world_sanity="pending", velocity_protocol="reset_to_rest",
                  geometry_detection="convex union SAT; implicit ground halfspace",
                  geometry_detection_scope="distinct owners with zero primitive thickness; original UIPC sanity checks same-owner contact")
    report["cpu_plan_elapsed_s"] = time.perf_counter() - started
    report["process_ru_maxrss_before_kib"] = peak_before
    report["process_ru_maxrss_after_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["memory_measurement_scope"] = "Process lifetime high-water RSS; change is not an isolated scratch peak"
    for name, mesh in meshes.items():
        report["bodies"][name]["live_mesh_sha256"] = hashlib.sha256(
            np.asarray(mesh.positions().view()).tobytes()
            + np.asarray(mesh.tetrahedra().topo().view()).tobytes()
            + np.asarray(mesh.transforms().view()).tobytes()
        ).hexdigest()
    ground = next(body for body in bodies if isinstance(body, Ground))
    _, _, _, ground_fingerprint = _ground_state(manager.uipc_sim.scene)
    report["ground"] = {"normal": ground.normal.tolist(), "offset": ground.offset,
                        "live_geometry_sha256": ground_fingerprint}
    table = manager.uipc_sim.scene.contact_tabular()
    report["same_owner_contact"] = {name: _same_owner_state(mesh, table) for name, mesh in meshes.items()}
    report["same_owner_validation"] = "pending original UIPC world.init sanity; no internal compound or FEM pair clearance projection"
    for name, binding in manager._prepared_scene.bindings.items():
        descriptor = binding.scene_object
        info = report["bodies"][name]
        info.update(source_sha256=binding.provenance.get("source_sha256"),
                    native_preparation_pose=list(binding.initial_pose),
                    source_asset=binding.provenance.get("source_path"), cache_path=binding.cache_path)
        if descriptor.dynamic and binding.robot_key is None:
            _, native_view = _native_actor_view(descriptor.prim_path, scene_manager.stage)
            velocities = native_view.get_velocities()
            if hasattr(velocities, "detach"):
                velocities = velocities.detach().cpu().numpy()
            elif not isinstance(velocities, np.ndarray):
                velocities = velocities.numpy()
            info["native_preparation_velocity_by_actor"] = dict(zip(native_view.prim_paths,
                                                                   np.asarray(velocities).tolist()))
            info["effective_initial_velocity"] = [0.0] * 6
            records = scene_manager.layout_manager.get_layout_records(descriptor.env_id, "Rigid")
            record = next(
                (record for record in records if record["inst_name"] == descriptor.instance_name), None
            )
            info["nominal_saved_pose"] = None if record is None else {
                "frame": "native rigid root local frame, as apply_saved_pose uses set_local_pose",
                "position": None if record.get("default_pos") is None else np.asarray(record["default_pos"]).tolist(),
                "quaternion_wxyz": None if record.get("default_ori") is None else np.asarray(record["default_ori"]).tolist(),
            }
            if (record is not None and record.get("default_pos") is not None
                    and record.get("default_ori") is not None):
                from pxr import Usd, UsdGeom
                from transforms3d.quaternions import quat2mat
                from .dynamics import rigid_pose

                prim = scene_manager.stage.GetPrimAtPath(descriptor.prim_path)
                parent = (np.eye(4) if UsdGeom.Xformable(prim).GetResetXformStack() else
                          np.asarray(UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(
                              Usd.TimeCode.Default())).T)
                nominal = np.eye(4)
                nominal[:3, 3] = record["default_pos"]
                nominal[:3, :3] = quat2mat(record["default_ori"])
                nominal_position, nominal_quaternion = rigid_pose(parent @ nominal)
                info["nominal_saved_world_pose"] = np.r_[nominal_position, nominal_quaternion].tolist()
                states = [("native_preparation", np.asarray(binding.initial_pose[:3]))]
                if "effective_matrix" in info:
                    states.append(("effective_initial", np.asarray(info["effective_matrix"])[:3, 3]))
                for key, position in states:
                    delta = position - nominal_position
                    info[key + "_displacement_from_layout_world_xyz_m"] = delta.tolist()
                    info[key + "_displacement_from_layout_norm_m"] = float(np.linalg.norm(delta))
    manager._initialization_report = report
    path = Path(manager.uipc_sim.cfg.workspace) / f"initialization-{manager._reset_count + 1:04d}.json"
    manager._initialization_report_path = str(path)
    _save_report(manager)
    if report["status"] not in {"clear_initial_state", "feasible_proposal"}:
        raise RuntimeError(f"Invalid tactile initial state: {report['status']}; details: {path}")
    # Independent final scan: no intermediate object or pair is committed early.
    by_name = {body.name: body for body in bodies}
    recheck = []
    for left, right in pairs:
        violations, candidates = pair_witnesses(by_name[left], by_name[right],
            np.asarray(report["bodies"][left]["delta_xyz_m"]),
            np.asarray(report["bodies"][right]["delta_xyz_m"]),
            clearance=report["clearance_m"], tolerance=report["tolerance_m"])
        recheck.append({"pair": [left, right], "candidates": candidates, "violations": len(violations)})
        if violations:
            report.update(status="independent_recheck_failed", independent_full_scene_recheck=recheck)
            _save_report(manager)
            raise RuntimeError(f"Initial-state proposal failed full scene recheck: {left}, {right}")
    report["independent_full_scene_recheck"] = recheck
    report["complete_distinct_owner_pairs_checked"] = True
    report["cpu_prepare_elapsed_s"] = time.perf_counter() - started
    report["process_ru_maxrss_after_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _save_report(manager)
    return bodies, meshes, fingerprint


def commit_initialization(manager, proposal):
    """Commit validated task rigid transforms to current/rest geometry together."""
    from copy import deepcopy
    from pxr import Gf
    from uipc import view

    bodies, meshes, fingerprint = proposal
    report = manager._initialization_report
    # No physics step or observation may occur between prepare and this commit.
    elements = {name: _contact_element(mesh) for name, mesh in meshes.items()}
    for name, mesh in meshes.items():
        digest = hashlib.sha256(np.asarray(mesh.positions().view()).tobytes()
                                + np.asarray(mesh.tetrahedra().topo().view()).tobytes()
                                + np.asarray(mesh.transforms().view()).tobytes()).hexdigest()
        if digest != report["bodies"][name]["live_mesh_sha256"]:
            raise RuntimeError(f"Scene geometry changed before initial-state commit: {name}")
    table = manager.uipc_sim.scene.contact_tabular()
    if "ground" in report:
        ground_mesh, _, _, ground_fingerprint = _ground_state(manager.uipc_sim.scene)
        elements["ground"] = _contact_element(ground_mesh)
        if ground_fingerprint != report["ground"]["live_geometry_sha256"]:
            raise RuntimeError("Implicit ground changed before initial-state commit")
    for name, original in report.get("same_owner_contact", {}).items():
        if _same_owner_state(meshes[name], table) != original:
            raise RuntimeError(f"Same-owner contact changed before initial-state commit: {name}")
    for row in report["contact_table"]:
        left, right = row["pair"]
        ids = row["element_ids"]
        if any(elements[name] != index for name, index in zip((left, right), ids)):
            raise RuntimeError("Contact elements changed before initial-state commit")
        model = table.at(*ids)
        if (bool(model.is_enabled()) != row["enabled"] or float(model.friction_rate()) != row["friction"]
                or float(model.resistance()) != row["resistance"]):
            raise RuntimeError("Contact model changed before initial-state commit")
    edits, cfg_edits = [], []
    for body in bodies:
        if not body.dynamic:
            continue
        rigid = manager.uipc_rigid_objects[body.name]
        current_slot, rest_slot = manager.uipc_sim.scene.geometries().find(rigid.geo_slot_list[0].id())
        matrix = np.asarray(body.nominal_pose).reshape(4, 4)
        effective = np.asarray(report["bodies"][body.name]["effective_matrix"])
        for mesh in (current_slot.geometry(), rest_slot.geometry()):
            old = np.asarray(mesh.transforms().view()).reshape(-1, 4, 4)[0].copy()
            if not np.array_equal(old, matrix):
                raise RuntimeError(f"Initial transform changed before commit: {body.name}")
            edits.append((mesh.transforms(), old, effective))
        binding = manager._prepared_scene.bindings[body.name]
        attribute = binding.scene_object.wrapper.stage.GetPrimAtPath(binding.prim_path).GetAttribute("xformOp:translate")
        if not attribute:
            raise RuntimeError(f"Hidden initial root has no translate operation: {body.name}")
        cfg_edits.append((rigid, rigid.cfg.init_state, attribute, attribute.Get(), effective))
    applied, applied_cfg = [], []
    try:
        for attribute, old, effective in edits:
            applied.append((attribute, old))
            view(attribute)[0] = effective
        for rigid, old_state, attribute, old_translation, effective in cfg_edits:
            applied_cfg.append((rigid, old_state, attribute, old_translation))
            state = deepcopy(old_state)
            state.pos = tuple(effective[:3, 3])
            # Orientation is preserved; use the native prepared WXYZ quaternion.
            name = next(name for name, value in manager.uipc_rigid_objects.items() if value is rigid)
            state.rot = tuple(manager._prepared_scene.bindings[name].initial_pose[3:])
            rigid.cfg.init_state = state
            if attribute.Set(Gf.Vec3d(*effective[:3, 3])) is False:
                raise RuntimeError("Hidden initial transform commit was rejected")
        report["commit"] = "all_task_rigid_initial_transforms"
        report["contact_table_sha256"] = fingerprint
        _save_report(manager)
    except BaseException:
        report["commit"] = "failed_rollback_requested"
        for rigid, old_state, attribute, old_translation in reversed(applied_cfg):
            rigid.cfg.init_state = old_state
            attribute.Set(old_translation)
        for attribute, old in reversed(applied):
            view(attribute)[0] = old
        raise


def _save_report(manager):
    path = Path(manager._initialization_report_path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manager._initialization_report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record_world_initialization(manager, *, valid):
    manager._initialization_report["native_world_sanity"] = "success" if valid else "failed"
    manager._initialization_report["same_owner_validation"] = "original UIPC world.init sanity " + ("passed" if valid else "failed")
    _save_report(manager)
