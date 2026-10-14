"""Prepare native SceneManager objects for the optional UIPC world.

This module does not import Isaac/TacEx or start a simulator. USD and meshing
dependencies are loaded only by ``prepare_scene``. The caller must finish native
layout creation and restore its poses before calling it, while physics is
suspended. Register every returned spec before UIPC ``world.init``. Afterwards
the caller must disable native dynamic physics and bind native pose/reset queries
to the UIPC state. Preparation alone does not transfer physics ownership.

Independent, hidden meshes are authored below ``root_path``; original USD meshes
and physics schemas are never edited here. A changed layout requires destruction
of the old UIPC world and removal of that subtree before preparing again.

Enabled USD collision surfaces are used by default; explicit source-mesh
overrides may select visible original meshes. PhysX SDF/decomposition
cooking is not reproduced: provenance records this backend change. Open surfaces
require explicit repair permission. Error checks are sampled bidirectional
surface distances, not a Hausdorff proof or a guarantee that small holes survive.
Application-specific opening/probe tests remain necessary before task acceptance.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


class ScenePreparationError(ValueError):
    """A native object cannot be converted without an unsupported assumption."""


@dataclass(frozen=True)
class SceneObject:
    env_id: int
    instance_name: str
    prim_path: str
    object_type: str
    dynamic: bool
    mass: float | None
    scale: tuple[float, float, float]
    source_path: str | None
    wrapper: Any
    robot_key: Any = None
    body_name: str | None = None
    collider_prim_paths: tuple[str, ...] | None = None
    layout_label: str | None = None


@dataclass(frozen=True)
class SceneBinding:
    scene_object: SceneObject
    name: str
    prim_path: str
    initial_pose: tuple[float, ...]
    cache_path: str
    volume: float
    provenance: dict[str, Any]
    robot_key: Any = None
    body_name: str | None = None


@dataclass
class PreparedScene:
    specs: list[dict[str, Any]]
    bindings: dict[str, SceneBinding]
    root_path: str


def collect_scene_objects(scene_manager) -> list[SceneObject]:
    """Collect *active layout* instances, excluding relocated stale wrappers.

    Dynamic/articulation/cloth/fluid layout categories are deliberately rejected:
    RoboDojo's ``Dynamic`` wrapper may contain arbitrary articulated USD assets,
    and must not silently be flattened into one rigid body. Native ``Rigid``
    instances (including a socket with metadata mass zero) retain their effective
    wrapper mass and remain dynamic. Tables retain their explicit static flag.
    """
    result = []
    layout = scene_manager.layout_manager
    for env_id in range(scene_manager.num_envs):
        for kind in ("Dynamic", "Articulation", "Garment", "Fluid"):
            if layout.get_layout_records(env_id, kind):
                raise ScenePreparationError(f"env {env_id}: unsupported native category {kind}")
        seen = set()
        for kind, collection in (
            ("Rigid", scene_manager._rigid_and_dynamic_objects[env_id]),
            ("Geometry", scene_manager._geometry_objects[env_id]),
        ):
            for record in layout.get_layout_records(env_id, kind):
                name = record["inst_name"]
                if name in seen:
                    raise ScenePreparationError(f"Duplicate active instance: env{env_id}:{name}")
                seen.add(name)
                obj = collection.get(name)
                if obj is None:
                    raise ScenePreparationError(f"Active native instance not spawned: env{env_id}:{name}")
                result.append(_describe(obj, env_id, name, kind.lower(), kind == "Rigid", label=record.get("label")))
        table = scene_manager._tables[env_id]
        if table is not None:
            name = str(getattr(table, "instance_name", "table"))
            if name in seen:
                raise ScenePreparationError(f"Table and task instance share name: {name}")
            result.append(_describe(table, env_id, name, "table", not table.is_static))
    if not result:
        raise ScenePreparationError("No native scene objects; prepare after loading a saved layout")
    return result


def _describe(obj, env_id, name, kind, dynamic, *, label=None):
    import math

    path = getattr(obj, "usd_prim_path", None) or getattr(obj, "_prim_path", None)
    if not path:
        raise ScenePreparationError(f"Native instance {name} has no prim path")
    scale = tuple(float(x) for x in obj.scale)
    if len(scale) != 3 or any(not math.isfinite(x) or x <= 0 for x in scale):
        raise ScenePreparationError(f"{name}: positive finite three-axis scale required, got {scale}")
    mass = None
    if dynamic:
        mass = getattr(obj, "mass", None)
        if mass is None or not math.isfinite(float(mass)) or float(mass) <= 0:
            raise ScenePreparationError(f"{name}: missing positive effective native mass")
        mass = float(mass)
    source = getattr(obj, "usd_path", None)
    return SceneObject(env_id, name, str(path), kind, dynamic, mass, scale,
                       str(Path(source).resolve()) if source else None, obj, layout_label=label)


def collect_robot_colliders(robot_manager, stage, *, exclude_prim_paths=()) -> list[SceneObject]:
    """Collect original articulation colliders as one kinematic shell per link.

    Call after PhysX articulation views expose ``body_names`` and after mounting
    sensor housings. Pass concrete gel subtree paths in ``exclude_prim_paths``;
    they remain separate FEM bodies. Each collision prim has exactly one owner,
    even when a fixed housing is nested below a finger. A nested rigid body not
    exposed in ``body_names`` must have an explicit fixed joint to its ancestor
    link. No force feedback into PhysX is implied by these kinematic shells.

    The returned ``robot_key`` and ``body_name`` identify the actual PhysX pose
    that the caller must read before *every* UIPC substep. Do not disable PhysX
    ownership or write UIPC poses back for these robot bindings.
    """
    import re

    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    exclusions = tuple(Sdf.Path(str(path)) for path in exclude_prim_paths)
    if any(not path.IsAbsolutePath() or not path.IsPrimPath() for path in exclusions):
        raise ScenePreparationError("Robot exclusions must be concrete absolute prim paths")

    def excluded(prim):
        return any(prim.GetPath().HasPrefix(path) for path in exclusions)

    def fixed_to(body, target):
        for joint_prim in stage.Traverse():
            if not joint_prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.FixedJoint(joint_prim)
            if joint.GetJointEnabledAttr().Get() is False:
                continue
            ends = set(joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets())
            if ends == {body.GetPath(), target.GetPath()}:
                return True
        return False

    keys = getattr(robot_manager, "robot_key", ())
    robots = getattr(robot_manager, "robot_list", ())
    if len(keys) != len(robots) or not keys:
        raise ScenePreparationError("Initialize robot articulation views before collecting colliders")
    result, visited, source_stages = [], set(), {}
    cache = UsdGeom.XformCache()

    def native_body_scale(body, key, live_scale):
        # A published float32 PhysX pose can contain ~1e-7 apparent stretch.
        # Recover the authored body's scale before runtime matrix overrides,
        # rather than treating quaternion roundoff as a different robot mesh.
        for spec in body.GetPrimStack():
            file = spec.layer.realPath
            if not file or not Path(file).is_file():
                continue
            identity = (str(Path(file).resolve()), str(spec.path))
            if identity not in source_stages:
                source_stages[identity] = Usd.Stage.OpenMasked(identity[0], Usd.StagePopulationMask([spec.path]))
            original = source_stages[identity].GetPrimAtPath(spec.path)
            if not original.IsValid() or not original.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            _, nominal = split_scale_transform(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(original)).T)
            if not UsdGeom.Xformable(body).GetResetXformStack():
                configured = getattr(getattr(getattr(key, "cfg", None), "spawn", None), "scale", None)
                if configured is not None:
                    nominal = nominal * np.asarray(configured, dtype=float)
            return np.round(nominal, 12)
        return live_scale
    for robot_index, (robot, key) in enumerate(zip(robots, keys)):
        if id(key) in visited:  # two arm wrappers can share one articulation
            continue
        visited.add(id(key))
        expression = str(getattr(getattr(key, "cfg", None), "prim_path", ""))
        expression = expression.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        if not expression:
            raise ScenePreparationError(f"robot {robot_index}: articulation has no prim path")
        try:
            roots = [prim for prim in stage.Traverse() if re.fullmatch(expression, str(prim.GetPath()))]
        except re.error as exc:
            raise ScenePreparationError(f"Invalid articulation path expression {expression!r}") from exc
        if not roots:
            raise ScenePreparationError(f"No native articulation matches {expression!r}")
        body_names = tuple(str(name) for name in getattr(key, "body_names", ()))
        if not body_names or len(set(body_names)) != len(body_names):
            raise ScenePreparationError(f"robot {robot_index}: missing or ambiguous PhysX body_names")
        source = getattr(getattr(getattr(key, "cfg", None), "spawn", None), "usd_path", None)
        for root in roots:
            match = re.search(r"/env_(\d+)(?:/|$)", str(root.GetPath()))
            if match:
                env_id = int(match.group(1))
            elif len(roots) == 1 and int(getattr(robot_manager, "num_envs", 1)) == 1:
                env_id = 0
            else:
                raise ScenePreparationError(f"Cannot identify environment for {root.GetPath()}")
            rigid_prims = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                           if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not excluded(prim)]
            bodies = {}
            for name in body_names:
                matches = [prim for prim in rigid_prims if prim.GetName() == name]
                if len(matches) != 1:
                    raise ScenePreparationError(f"{root.GetPath()}: PhysX body {name!r} has {len(matches)} USD matches")
                bodies[str(matches[0].GetPath())] = (name, matches[0])
            assigned = {path: [] for path in bodies}
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if excluded(prim):
                    continue
                if prim.GetAttribute("tet_points").HasAuthoredValueOpinion():
                    raise ScenePreparationError(f"Exclude FEM gel subtree explicitly: {prim.GetPath()}")
                collision = UsdPhysics.CollisionAPI(prim)
                if not collision or collision.GetCollisionEnabledAttr().Get() is False:
                    continue
                ancestor, nearest_rigid = prim, None
                while ancestor.IsValid() and ancestor.GetPath().HasPrefix(root.GetPath()):
                    if nearest_rigid is None and ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                        nearest_rigid = ancestor
                    if str(ancestor.GetPath()) in bodies:
                        break
                    ancestor = ancestor.GetParent()
                if not ancestor.IsValid() or str(ancestor.GetPath()) not in bodies:
                    raise ScenePreparationError(f"Collider has no matching PhysX body owner: {prim.GetPath()}")
                if nearest_rigid is not None and nearest_rigid != ancestor and not fixed_to(nearest_rigid, ancestor):
                    raise ScenePreparationError(f"Unlisted nested rigid body is not fixed to its link: {nearest_rigid.GetPath()}")
                assigned[str(ancestor.GetPath())].append(str(prim.GetPath()))
            for path, (name, body) in bodies.items():
                if not assigned[path]:
                    continue  # source link genuinely has no enabled collision geometry
                _, scale = split_scale_transform(np.asarray(cache.GetLocalToWorldTransform(body)).T)
                scale = native_body_scale(body, key, scale)
                result.append(SceneObject(env_id, f"robot{robot_index}:{name}", path,
                    "RobotCollider", False, None, tuple(float(x) for x in scale),
                    str(Path(source).resolve()) if source else None, robot, key, name, tuple(assigned[path])))
    if not result:
        raise ScenePreparationError("No original robot collision geometry was found")
    return result


def split_scale_transform(matrix):
    """Return a pure SE(3) transform and positive scale; reject shear/reflection."""
    import numpy as np

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ScenePreparationError("Expected a finite 4x4 world transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ScenePreparationError("Projective transforms are unsupported")
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    if np.any(scale <= 0):
        raise ScenePreparationError("Zero scale is unsupported")
    rotation = matrix[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
        raise ScenePreparationError("Native root shear or reflection is unsupported")
    pose = matrix.copy()
    pose[:3, :3] = rotation
    return pose, scale


def _triangulate(indices, counts, points):
    """Triangulate convex planar USD polygons; reject ambiguous concave faces."""
    import numpy as np

    faces, offset = [], 0
    for count in counts:
        count = int(count)
        ids = indices[offset : offset + count]
        offset += count
        if count < 3 or len(ids) != count:
            raise ScenePreparationError("Invalid USD polygon topology")
        if count > 3:
            polygon = points[ids]
            normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
            norm = np.linalg.norm(normal)
            if norm == 0:
                raise ScenePreparationError("Degenerate USD polygon")
            normal /= norm
            if np.max(np.abs((polygon - polygon[0]) @ normal)) > 1e-6:
                raise ScenePreparationError("Non-planar USD polygon is unsupported")
            turns = np.cross(np.roll(polygon, -1, axis=0) - polygon,
                             np.roll(polygon, -2, axis=0) - np.roll(polygon, -1, axis=0)) @ normal
            if np.any(turns < -1e-12):
                raise ScenePreparationError("Concave USD polygon must be triangulated in the source asset")
        faces.extend((int(ids[0]), int(ids[j]), int(ids[j + 1])) for j in range(1, count - 1))
    if offset != len(indices):
        raise ScenePreparationError("USD face counts do not match indices")
    return np.asarray(faces, dtype=np.int64)


def _read_collision_surface(stage, descriptor, *, selected_meshes=False):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    import trimesh

    root = stage.GetPrimAtPath(descriptor.prim_path)
    if not root.IsValid():
        raise ScenePreparationError(f"Missing original USD prim: {descriptor.prim_path}")
    cache = UsdGeom.XformCache()
    pose, scale = split_scale_transform(np.asarray(cache.GetLocalToWorldTransform(root)).T)
    nominal_scale = np.asarray(descriptor.scale, dtype=float)
    scale = np.where(np.isclose(scale, nominal_scale, rtol=1e-6, atol=1e-12), nominal_scale, scale)
    inverse_pose = np.linalg.inv(pose)
    parts, sources = [], []
    collision_prims = (Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if descriptor.collider_prim_paths is None else
                       (stage.GetPrimAtPath(path) for path in descriptor.collider_prim_paths))
    for prim in collision_prims:
        if not prim.IsValid() or not prim.GetPath().HasPrefix(root.GetPath()):
            raise ScenePreparationError(f"Missing/out-of-body collider for {descriptor.prim_path}")
        collision = UsdPhysics.CollisionAPI(prim)
        if selected_meshes:
            if (not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible"
                    or UsdGeom.Imageable(prim).ComputePurpose() not in {"default", "render"}):
                raise ScenePreparationError(f"Selected source mesh must be visible: {prim.GetPath()}")
        elif not collision or collision.GetCollisionEnabledAttr().Get() is False:
            continue
        collision_owner = prim
        if prim.IsA(UsdGeom.Xform):
            # ARX authors CollisionAPI + convexHull on a collision-container
            # Xform, with its single imported STL mesh immediately below it.
            # This is explicit author geometry, never a visual fallback.
            children = [child for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                        if child != prim and child.IsA(UsdGeom.Gprim)
                        and not child.HasAPI(UsdPhysics.CollisionAPI)]
            if len(children) != 1:
                raise ScenePreparationError(f"Collision container {prim.GetPath()} must contain exactly one unambiguous Gprim")
            prim = children[0]
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            if mesh.GetHoleIndicesAttr().Get():
                raise ScenePreparationError(f"USD polygon holes unsupported at {prim.GetPath()}")
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            faces = _triangulate(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64),
                                 mesh.GetFaceVertexCountsAttr().Get(), points)
            approximation = ("none" if selected_meshes else
                             UsdPhysics.MeshCollisionAPI(collision_owner).GetApproximationAttr().Get()
                             if collision_owner.HasAPI(UsdPhysics.MeshCollisionAPI) else "none")
        elif prim.IsA(UsdGeom.Cube):
            cube = trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())] * 3)
            points, faces, approximation = cube.vertices, cube.faces, "cube"
        else:
            raise ScenePreparationError(f"Unsupported collider {prim.GetTypeName()} at {prim.GetPath()}; author a collision Mesh")
        relative, reset_stack = cache.ComputeRelativeTransform(prim, root)
        transform = (inverse_pose @ np.asarray(cache.GetLocalToWorldTransform(prim)).T if reset_stack else
                     np.diag([*scale, 1.0]) @ np.asarray(relative).T)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        if len(points) == 0 or not np.isfinite(points).all() or len(faces) == 0:
            raise ScenePreparationError(f"Empty/non-finite collision surface at {prim.GetPath()}")
        if np.min(faces) < 0 or np.max(faces) >= len(points):
            raise ScenePreparationError(f"Out-of-range collision indices at {prim.GetPath()}")
        part = trimesh.Trimesh(points, faces, process=False)
        if approximation == "convexHull":
            from scipy.spatial import ConvexHull

            hull = ConvexHull(points)
            faces = hull.simplices.copy()
            triangles = points[faces]
            winding = np.einsum("ij,ij->i", np.cross(triangles[:, 1] - triangles[:, 0],
                                triangles[:, 2] - triangles[:, 0]), hull.equations[:, :3])
            faces[winding < 0] = faces[winding < 0][:, [0, 2, 1]]
            part = trimesh.Trimesh(points, faces, process=False)
            part.remove_unreferenced_vertices()
            part.metadata["robodojo_author_convex_hull"] = True
        elif approximation not in ("none", "sdf", "convexDecomposition", "meshSimplification", "cube"):
            raise ScenePreparationError(f"Unsupported collision approximation {approximation!r}")
        elif not selected_meshes:
            part.merge_vertices(digits_vertex=8)
            part.update_faces(part.unique_faces())
            part.update_faces(part.nondegenerate_faces())
            part.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(part, multibody=False)
        static_triangle_skin = (not descriptor.dynamic and descriptor.object_type == "geometry"
                                and approximation == "none")
        if static_triangle_skin:
            part.metadata["robodojo_static_triangle_skin"] = True
        parts.append(part)
        sources.append({"prim": str(prim.GetPath().MakeRelativePath(root.GetPath())), "approximation": approximation,
                        "collision_owner": str(collision_owner.GetPath().MakeRelativePath(root.GetPath())),
                        "vertices": len(part.vertices), "triangles": len(part.faces),
                        **({"conversion_policy": "bounded_static_triangle_skin"} if static_triangle_skin else {})})
    if not parts:
        raise ScenePreparationError(f"No enabled explicit collision geometry below {descriptor.prim_path}")
    surface = parts[0].copy() if len(parts) == 1 else trimesh.util.concatenate(parts)
    # Preserve author collider boundaries. Unioning overlapping author shapes
    # can erase interior faces and corrupt a surface-error audit. Every part
    # will share one ABD transform, never become an additional simulated body.
    if len(parts) > 1:
        surface.metadata["robodojo_collision_parts"] = parts
    source_path = Path(descriptor.source_path).resolve() if descriptor.source_path else None
    if source_path is not None and not source_path.is_file():
        raise ScenePreparationError(f"Original source asset is missing: {source_path}")
    provenance = {"source_path": str(source_path) if source_path else None,
                  "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path else None,
                  "colliders": sources, "baked_scale": np.round(scale, 12).tolist(),
                  "source_watertight": bool(surface.is_watertight),
                  "scale_roundoff_policy": "Snap float32 pose stretch to authored native scale only within one part per million",
                  "backend_note": "Original collision surfaces; PhysX SDF/decomposition cooking is not reproduced"}
    return surface, pose, provenance


def _source_mesh_options(options):
    from pxr import Sdf

    if not isinstance(options, Mapping):
        raise ScenePreparationError("source_meshes must map source asset files to options")
    result = {}
    repo = Path(__file__).resolve().parents[2]
    for source, policy in options.items():
        if not isinstance(source, str) or not isinstance(policy, Mapping):
            raise ScenePreparationError("source_meshes requires source-file keys and mapping values")
        if set(policy) != {"prim_path", "method"} or policy["method"] != "tetgen_plc":
            raise ScenePreparationError("Source meshes require prim_path and method=tetgen_plc")
        if not isinstance(policy["prim_path"], str):
            raise ScenePreparationError("Source mesh prim_path must be a plain string")
        path = Sdf.Path(policy["prim_path"])
        if not path.IsPrimPath() or path.IsAbsolutePath() or ".." in path.pathString.split("/"):
            raise ScenePreparationError("Source mesh prim_path must remain relative inside its native body")
        relative = Path(source)
        if relative.is_absolute() or ".." in relative.parts:
            raise ScenePreparationError("Source assets must be repository-relative without parent traversal")
        asset = (repo / relative).resolve()
        if not asset.is_relative_to(repo):
            raise ScenePreparationError("Source asset resolves outside the repository")
        if not asset.is_file() or str(asset) in result:
            raise ScenePreparationError(f"Missing or duplicate source asset: {asset}")
        result[str(asset)] = {**policy, "prim_path": path.pathString}
    return result


def _canonical_source_surface(surface):
    """Remove only exact geometric redundancy, recording every source mapping."""
    import numpy as np
    import trimesh

    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ScenePreparationError("Selected source requires finite vertices and valid triangles")
    _, first, inverse = np.unique(vertices, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    lookup = np.empty(len(order), dtype=np.int64)
    lookup[order] = np.arange(len(order))
    raw_to_vertex = lookup[inverse]
    points = vertices[first[order]]
    mapped = raw_to_vertex[faces]
    cross = np.cross(points[mapped[:, 1]] - points[mapped[:, 0]],
                     points[mapped[:, 2]] - points[mapped[:, 0]])
    if not np.isfinite(cross).all():
        raise ScenePreparationError("Source facet areas must be finite")
    areas = np.linalg.norm(cross, axis=1) / 2
    retained, raw_to_face, removed, seen = [], np.full(len(faces), -1, dtype=np.int64), [], {}
    for raw_id, (face, area) in enumerate(zip(mapped, areas)):
        if area == 0:
            from fractions import Fraction

            triangle = [[Fraction(float(x)) for x in vertex] for vertex in points[face]]
            a = [triangle[1][i] - triangle[0][i] for i in range(3)]
            b = [triangle[2][i] - triangle[0][i] for i in range(3)]
            if any(a[i] * b[j] != a[j] * b[i] for i, j in ((0, 1), (1, 2), (2, 0))):
                raise ScenePreparationError("Positive source facet area is below floating-point representation")
            removed.append({"source_face": raw_id, "reason": "exact_zero_area", "area_m2": 0.0})
            continue
        at = int(face.argmin())
        oriented = tuple(np.roll(face, -at).tolist())
        key = tuple(sorted(oriented))
        previous = seen.get(key)
        if previous is not None:
            retained_id, winding, retained_source_face = previous
            if oriented != winding:
                raise ScenePreparationError("Oppositely wound coincident source facets are unsupported")
            raw_to_face[raw_id] = retained_id
            removed.append({"source_face": raw_id, "reason": "exact_oriented_duplicate",
                            "retained_face": retained_id, "retained_source_face": retained_source_face,
                            "area_m2": float(area)})
            continue
        raw_to_face[raw_id] = len(retained)
        seen[key] = (len(retained), oriented, raw_id)
        retained.append(face)
    if not retained:
        raise ScenePreparationError("Source contains no nonzero unique triangles")
    retained = np.asarray(retained, dtype=np.int64)
    used = np.unique(retained)
    compact = np.full(len(points), -1, dtype=np.int64)
    compact[used] = np.arange(len(used))
    raw_to_vertex = compact[raw_to_vertex]
    canonical = trimesh.Trimesh(points[used], compact[retained], process=False)
    counts = np.bincount(canonical.edges_unique_inverse)
    if (np.any(counts > 2) or not canonical.is_watertight
            or not canonical.is_winding_consistent or not np.isfinite(canonical.volume) or canonical.volume <= 0):
        raise ScenePreparationError("Exact canonical source must be a positive closed consistently wound manifold")
    operations = {"raw_vertices": len(vertices), "raw_faces": len(faces),
                  "canonical_vertices": len(canonical.vertices), "canonical_faces": len(canonical.faces),
                  "raw_to_canonical_vertex": raw_to_vertex.tolist(),
                  "canonical_vertex_source_representatives": first[order][used].tolist(),
                  "raw_to_canonical_face": raw_to_face.tolist(), "removed_faces": removed,
                  "maximum_vertex_shift_m": 0.0}
    return canonical, operations


def _source_mesh_hash(surface):
    import numpy as np

    return hashlib.sha256(np.asarray(surface.vertices, dtype="<f8").tobytes()
                          + np.asarray(surface.faces, dtype="<i8").tobytes()).hexdigest()


def _check_source_facets(source, points, boundary):
    """Require unchanged original boundary faces, allowing only float roundoff."""
    import numpy as np
    from scipy.spatial import cKDTree

    boundary_nodes = np.unique(boundary)
    error, ids = cKDTree(source.vertices).query(points[boundary_nodes])
    if (len(boundary_nodes) != len(source.vertices) or error.max(initial=0.0) > 1e-12
            or len(np.unique(ids)) != len(source.vertices)):
        raise ScenePreparationError("TetGen must preserve every original source boundary vertex")
    mapping = np.full(len(points), -1, dtype=np.int64)
    mapping[boundary_nodes] = ids
    def canonical(faces):
        first = faces.argmin(axis=1)
        return np.take_along_axis(faces, (first[:, None] + np.arange(3)) % 3, axis=1)

    expected = canonical(source.faces)
    actual = canonical(mapping[boundary])
    expected = expected[np.lexsort(expected.T[::-1])]
    actual = actual[np.lexsort(actual.T[::-1])]
    if not np.array_equal(actual, expected):
        raise ScenePreparationError("TetGen boundary face multiset differs from the original source")
    return float(error.max(initial=0.0))


def _source_tetgen_mesh(surface):
    """Constrained PLC volume: no boundary refinement, merging or coarsening."""
    import numpy as np
    import trimesh

    try:
        import tetgen
    except ImportError as exc:
        raise ScenePreparationError("Install tactile dependencies (tetgen==0.8.4) for source-mesh conversion") from exc

    origin = surface.bounds.mean(axis=0)
    scale = float(np.linalg.norm(surface.extents))
    mesher = tetgen.TetGen((surface.vertices - origin) / scale, surface.faces)
    points, tets, _, _ = mesher.tetrahedralize(
        plc=True, quality=False, nobisect=True, nomergefacet=True, nomergevertex=True,
        coarsen=False, cdt=False, noexact=False, zeroindex=True, order=1,
        steinerleft=1000, epsilon=1e-10, docheck=True, quiet=True,
    )
    if len(points) > 20000 or len(tets) > 100000:
        raise ScenePreparationError("Source-conforming TetGen mesh exceeds its bounded size")
    points, tets, boundary, volume = validate_tetrahedra(points * scale + origin, tets)
    boundary_error = _check_source_facets(surface, points, boundary)
    if not np.isclose(volume, surface.volume, rtol=1e-10, atol=1e-18):
        raise ScenePreparationError("Source-conforming tetrahedra must preserve enclosed volume")
    output = trimesh.Trimesh(points, boundary, process=False)
    output.remove_unreferenced_vertices()
    if not output.is_watertight or not output.is_winding_consistent:
        raise ScenePreparationError("Source-conforming tetrahedra require a closed consistent boundary")
    normalized_source = surface.copy()
    normalized_output = output.copy()
    normalized_source.vertices = (surface.vertices - origin) / scale
    normalized_output.vertices = (output.vertices - origin) / scale
    forward, backward = _sampled_surface_errors(normalized_source, normalized_output)
    forward, backward = forward * scale, backward * scale
    metrics = {"method": "source_conforming_tetgen_plc", "volume_m3": volume,
               "vertices": len(points), "tetrahedra": len(tets),
               "source_to_tet_sampled_max_m": forward, "tet_to_source_sampled_max_m": backward,
               "max_surface_error_m": 1e-12, "boundary_faces_preserved": True,
               "surface_error_gate": "oriented_facet_multiset_and_bijective_boundary_vertices",
               "boundary_vertex_max_error_m": boundary_error, "sampled_errors_are_diagnostic": True,
               "mass_distribution_note": "Native total mass retained; uniform source solid, not PhysX cooked COM/inertia"}
    return points, tets, boundary, metrics



def tetrahedralize_surface(surface, *, allow_surface_repair=False, max_surface_error=0.0003):
    """Convert collision surface; return validated vertices/tets/boundary/metrics.

    Closed convex surfaces use an exact centre fan. Other surfaces require
    wildmeshing. Repair may close gaps; numerical proximity alone cannot certify
    insertion holes, so callers must additionally run task-specific probes.
    """
    import numpy as np
    import trimesh

    if not np.isfinite(max_surface_error) or max_surface_error <= 0:
        raise ScenePreparationError("max_surface_error must be positive finite metres")
    if surface.metadata.get("source_conforming"):
        return _source_tetgen_mesh(surface)
    author_parts = surface.metadata.get("robodojo_collision_parts")
    if author_parts:
        results = []
        for index, part in enumerate(author_parts):
            try:
                results.append(tetrahedralize_surface(part, allow_surface_repair=allow_surface_repair,
                                                      max_surface_error=max_surface_error))
            except ScenePreparationError as exc:
                raise ScenePreparationError(f"author collider {index}: {exc}") from exc
        return _combine_tetrahedra(results, "author_collider_compound", max_surface_error)
    if not surface.is_watertight and not allow_surface_repair:
        raise ScenePreparationError("Open collision surface requires explicit allow_surface_repair=True")
    static_triangle_skin = bool(surface.metadata.get("robodojo_static_triangle_skin"))
    if static_triangle_skin and not allow_surface_repair:
        raise ScenePreparationError("Static triangle contact skin requires explicit allow_surface_repair=True")
    working = surface.copy()
    repair = {}
    if static_triangle_skin:
        working, repair = _repair_static_micro_edges(working, max_surface_error)
    centred = np.asarray(working.vertices) - np.mean(working.vertices, axis=0)
    _, singular, basis = np.linalg.svd(centred, full_matrices=False)
    planar = singular[-1] <= max(singular[0] * 1e-10, 1e-12)
    components = working.split(only_watertight=False, repair=False)
    # Split independent, positively oriented solids and open contact sheets.
    # Keep any inward-oriented nested shell with its outer component:
    # its negative winding can define a cavity rather than another solid.
    def independent_component(part):
        return (part.is_watertight and part.is_winding_consistent and part.volume > 1e-20
                or not part.is_watertight and part.is_winding_consistent and allow_surface_repair)
    if (len(components) > 1 and not repair and not planar and not static_triangle_skin
            and all(independent_component(c) for c in components)):
        results = [tetrahedralize_surface(part, allow_surface_repair=allow_surface_repair,
                                         max_surface_error=max_surface_error) for part in components]
        return _combine_tetrahedra(results, "oriented_source_components", max_surface_error)
    if (not working.is_watertight or static_triangle_skin) and allow_surface_repair:
        # Keep original open contact surfaces and openings. Filling their edge
        # loops can cap a socket; instead the explicitly permitted approximation
        # offsets the source wall symmetrically, never more than +/-0.1 mm.
        thickness = min(0.0002, max_surface_error)
        if static_triangle_skin and np.any(np.bincount(working.edges_unique_inverse) > 2):
            raise ScenePreparationError("Static triangle contact skin has a non-manifold source edge")
        normal = basis[-1] if planar else np.asarray(working.vertex_normals)
        if not planar and (not working.is_winding_consistent or np.any(np.linalg.norm(normal, axis=1) < .5)):
            raise ScenePreparationError("Open surface lacks consistently oriented nonzero normals for bounded thickening")
        offset = normal * (thickness / 2)
        points = np.vstack([working.vertices - offset, working.vertices + offset])
        lower = np.sort(working.faces, axis=1)
        a, b, c = lower.T
        n = len(working.vertices)
        tets = np.concatenate([np.column_stack([a, b, c, c + n]),
                               np.column_stack([a, b, b + n, c + n]),
                               np.column_stack([a, a + n, b + n, c + n])])
        method = ("bounded_static_triangle_skin" if static_triangle_skin else
                  "bounded_planar_thickening" if planar else "bounded_open_surface_thickening")
        repair["symmetric_thickness_m"] = thickness
        repair["mass_distribution_note"] = "Original open contact surface represented by a finite-thickness shell; total mass can be preserved, but interior mass distribution and inertia differ from PhysX SDF cooking"
        if static_triangle_skin:
            repair["mass_distribution_note"] = "Original static triangle contact surface represented by a finite-thickness kinematic shell; surface offset is bounded, and no dynamic mass/inertia equivalence is claimed"
    elif working.is_watertight and (working.metadata.get("robodojo_author_convex_hull") or working.is_convex):
        points = np.vstack([working.vertices, working.vertices.mean(axis=0)])
        tets = np.column_stack([working.faces, np.full(len(working.faces), len(points) - 1)])
        method = "exact_convex_centre_fan"
    else:
        points, tets = _wild_tetrahedra(working, max_surface_error)
        method = "wildmeshing_surface_repair" if not surface.is_watertight else "wildmeshing"
    try:
        points, tets, boundary, volume = validate_tetrahedra(points, tets)
        output = trimesh.Trimesh(points, boundary, process=False)
        output.remove_unreferenced_vertices()
        if not output.is_watertight:
            raise ScenePreparationError("Converted tetrahedra have a non-manifold/open boundary")
    except ScenePreparationError as exc:
        if method != "bounded_open_surface_thickening":
            raise
        # Non-manifold source walls cannot always be offset into valid prisms.
        # The explicitly permitted volumetric repair is still audited against
        # every original vertex/face centre below, and may fail that audit.
        points, tets = _wild_tetrahedra(surface, max_surface_error)
        points, tets, boundary, volume = validate_tetrahedra(points, tets)
        output = trimesh.Trimesh(points, boundary, process=False)
        output.remove_unreferenced_vertices()
        if not output.is_watertight:
            raise ScenePreparationError("Volumetric repair has a non-manifold/open boundary") from exc
        method = "wildmeshing_surface_repair"
        repair = {"thin_shell_rejected": str(exc)}
    if method == "exact_convex_centre_fan":
        from scipy.spatial import ConvexHull

        # Trimesh's is_convex is approximate. The label certifies only an exact
        # optional reduction; a valid source fan keeps its original points/tets.
        if not np.isclose(ConvexHull(points).volume, volume, rtol=1e-8, atol=1e-14):
            method = "source_centre_fan"
    forward, backward = _sampled_surface_errors(surface, output)
    if max(forward, backward) > max_surface_error:
        raise ScenePreparationError(f"Surface deviation {max(forward, backward):.9g} m exceeds {max_surface_error:.9g} m "
                                    f"(source→tet={forward:.9g}, tet→source={backward:.9g})")
    metrics = {"method": method, "source_to_tet_sampled_max_m": forward,
               "tet_to_source_sampled_max_m": backward, "max_surface_error_m": max_surface_error,
               "volume_m3": volume, "vertices": len(points), "tetrahedra": len(tets)}
    if repair:
        metrics["repair"] = repair
    return points, tets, boundary, metrics


def _sampled_surface_errors(surface, output):
    """Bidirectional all-vertex/face-centre audit, not an exact Hausdorff bound."""
    import numpy as np
    import trimesh

    def maximum_distance(source, target):
        samples = np.vstack([source.vertices, source.triangles_center])
        # Trimesh's fixed geometric tolerances are dimensionless. Normalize
        # metre-scale miniatures or thin convex facets are falsely treated as
        # degenerate and can report sub-millimetre errors for identical faces.
        origin = np.minimum(source.bounds[0], target.bounds[0])
        scale = float(np.linalg.norm(np.maximum(source.bounds[1], target.bounds[1]) - origin))
        target = target.copy()
        target.vertices = (target.vertices - origin) / scale
        samples = (samples - origin) / scale
        maximum = 0.0
        for start in range(0, len(samples), 4096):
            _, distances, _ = trimesh.proximity.closest_point(target, samples[start:start + 4096])
            maximum = max(maximum, float(np.max(distances, initial=0.0)) * scale)
        return maximum
    return maximum_distance(surface, output), maximum_distance(output, surface)


def _compatible_cached_geometries(cache_dir, identity):
    """Find old source-identical caches; callers must still audit their mesh."""
    import numpy as np

    current = dict(identity["provenance"])
    if not current.get("source_sha256"):
        return
    current.pop("scale_roundoff_policy", None)
    candidates = sorted(Path(cache_dir).glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            with np.load(path, allow_pickle=False) as stored:
                previous = json.loads(str(stored["identity"]))
            if (previous.get("format_version") not in (5, 6)
                    or previous.get("allow_surface_repair") != identity["allow_surface_repair"]
                    or previous.get("max_surface_error") != identity["max_surface_error"]):
                continue
            original = dict(previous["provenance"])
            original.pop("scale_roundoff_policy", None)
            if original.get("source_path"):
                original["source_path"] = str(Path(original["source_path"]).resolve())
            if not np.allclose(original["baked_scale"], current["baked_scale"], rtol=1e-6, atol=1e-12):
                continue
            original["baked_scale"] = current["baked_scale"]
            if original == current:
                yield path, previous
        except (OSError, KeyError, ValueError):
            continue  # an unrelated incomplete/corrupt cache is not a candidate


def _wild_tetrahedra(surface, max_surface_error):
    import numpy as np
    try:
        import wildmeshing
    except ImportError as exc:
        raise ScenePreparationError("Install wildmeshing to tetrahedralize non-convex collision surfaces") from exc
    diagonal = float(np.linalg.norm(surface.extents))
    if diagonal <= 0:
        raise ScenePreparationError("Zero-size collision surface")
    mesher = wildmeshing.Tetrahedralizer(stop_quality=15, max_its=30,
                epsilon=min(max_surface_error / diagonal / 4, 0.0025), edge_length_r=0.06,
                skip_simplify=False, coarsen=True)
    mesher.set_log_level(6)
    mesher.set_mesh(np.asarray(surface.vertices), np.asarray(surface.faces, dtype=np.int32))
    mesher.tetrahedralize()
    points, tets, _ = mesher.get_tet_mesh(correct_surface_orientation=True,
                                       manifold_surface=True, use_input_for_wn=False)
    return np.asarray(points), np.asarray(tets, dtype=np.int64)


def _repair_static_micro_edges(surface, max_surface_error):
    """Collapse only sub-tolerance non-manifold CAD edges, never large defects.

    Caller audits the result against the untouched source surface. This handles
    tiny CAD slivers, not arbitrary triangle soups or missing task geometry.
    """
    import numpy as np
    import trimesh

    repaired = surface.copy()
    moved, collapsed, removed = 0.0, 0, 0
    for _ in range(3):
        bad = repaired.edges_unique[np.bincount(repaired.edges_unique_inverse) > 2]
        if not len(bad):
            break
        points, faces = np.asarray(repaired.vertices).copy(), np.asarray(repaired.faces).copy()
        lengths = np.linalg.norm(points[bad[:, 0]] - points[bad[:, 1]], axis=1)
        limit = min(0.0001, max_surface_error / 3)
        if np.any(lengths > limit):
            raise ScenePreparationError(f"Static triangle contact skin has a non-manifold source edge of {float(max(lengths)):.9g} m; micro-edge repair limit is {limit:.9g} m")
        parent = np.arange(len(points))
        def root(index):
            while parent[index] != index:
                index = int(parent[index])
            return index
        for a, b in bad:
            parent[root(int(b))] = root(int(a))
        groups = {}
        for index in np.unique(bad):
            groups.setdefault(root(int(index)), []).append(int(index))
        for indices in groups.values():
            centre = points[indices].mean(axis=0)
            displacement = float(np.max(np.linalg.norm(points[indices] - centre, axis=1)))
            if displacement > limit:
                raise ScenePreparationError("Non-manifold CAD edge cluster exceeds bounded local repair")
            moved = max(moved, displacement)
            points[indices] = centre
        mapping = np.asarray([root(index) for index in range(len(points))])
        faces = mapping[faces]
        keep = np.all(np.diff(np.sort(faces, axis=1), axis=1) != 0, axis=1)
        removed += int((~keep).sum())
        repaired = trimesh.Trimesh(points, faces[keep], process=False)
        repaired.update_faces(repaired.unique_faces())
        repaired.update_faces(repaired.nondegenerate_faces(height=1e-12))
        repaired.remove_unreferenced_vertices()
        collapsed += len(bad)
        if not len(repaired.faces):
            raise ScenePreparationError("Non-manifold CAD repair would remove the entire collision surface")
    if np.any(np.bincount(repaired.edges_unique_inverse) > 2):
        raise ScenePreparationError("Static triangle contact skin remains non-manifold after bounded micro-edge repair")
    if collapsed:
        trimesh.repair.fix_normals(repaired, multibody=False)
        return repaired, {"micro_edges_collapsed": int(collapsed), "collapsed_faces": removed,
                          "maximum_local_vertex_shift_m": moved}
    return repaired, {}


def _combine_tetrahedra(results, method, max_surface_error):
    """Retain author collider parts in one ABD, including original overlap."""
    import numpy as np

    points, tets, offset = [], [], 0
    for result in results:
        vertices, cells = result[:2]
        points.append(vertices)
        tets.append(cells + offset)
        offset += len(vertices)
    points, tets, boundary, volume = validate_tetrahedra(np.concatenate(points), np.concatenate(tets))
    metrics = {"method": method,
               "source_to_tet_sampled_max_m": max(r[3]["source_to_tet_sampled_max_m"] for r in results),
               "tet_to_source_sampled_max_m": max(r[3]["tet_to_source_sampled_max_m"] for r in results),
               "max_surface_error_m": max_surface_error, "volume_m3": volume,
               "vertices": len(points), "tetrahedra": len(tets),
               "parts": [r[3] for r in results],
               "mass_distribution_note": "Total mass preserved using summed tetrahedral volume; overlapping author collider volumes are counted per part, so inertia is not equivalent to PhysX cooking"}
    return points, tets, boundary, metrics


def validate_tetrahedra(points, tets):
    """Orient tetrahedra and return their consistently wound exterior boundary."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64).copy()
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ScenePreparationError("Invalid tetrahedral vertices")
    if tets.ndim != 2 or tets.shape[1] != 4 or not len(tets):
        raise ScenePreparationError("No valid tetrahedral cells")
    if np.min(tets) < 0 or np.max(tets) >= len(points):
        raise ScenePreparationError("Tetrahedron index out of range")
    determinant = np.linalg.det(points[tets[:, 1:]] - points[tets[:, 0, None]])
    if not np.isfinite(determinant).all() or np.any(np.abs(determinant) <= 1e-20):
        raise ScenePreparationError("Degenerate tetrahedral cell")
    negative = determinant < 0
    tets[negative, 0], tets[negative, 1] = tets[negative, 1].copy(), tets[negative, 0].copy()
    if len(np.unique(np.sort(tets, axis=1), axis=0)) != len(tets):
        raise ScenePreparationError("Duplicate tetrahedral cell")
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]],
                            tets[:, [0, 3, 2]], tets[:, [1, 2, 3]]])
    _, indices, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ScenePreparationError("Non-manifold tetrahedral face")
    return points, tets, faces[indices[counts == 1]], float(np.abs(determinant).sum() / 6)


def _cached_mesh(surface, provenance, cache_dir, allow_surface_repair, max_surface_error):
    import math

    import numpy as np

    if not math.isfinite(max_surface_error) or max_surface_error <= 0:
        raise ScenePreparationError("max_surface_error must be positive finite metres")
    decimals = max(9, math.ceil(-math.log10(max_surface_error)) + 3)
    identity = {"format_version": 6, "provenance": provenance,
                "allow_surface_repair": bool(allow_surface_repair), "max_surface_error": max_surface_error,
                "geometry_hash_quantization_m": 10.0 ** -decimals}
    def cache_path(metadata, digits, canonical_zeros=True):
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        vertices = np.round(np.asarray(surface.vertices, dtype="<f8"), digits)
        if canonical_zeros:
            vertices[vertices == 0] = 0.0  # -0 and +0 are identical geometry
        digest.update(vertices.tobytes())
        digest.update(np.asarray(surface.faces, dtype="<i8").tobytes())
        return Path(cache_dir) / f"{digest.hexdigest()}.npz"
    path = cache_path(identity, decimals)
    path.parent.mkdir(parents=True, exist_ok=True)
    def read_cached(candidate, metadata):
        try:
            with np.load(candidate, allow_pickle=False) as cached:
                points, tets, boundary, volume = validate_tetrahedra(cached["points"], cached["tets"])
                metrics = json.loads(str(cached["metrics"]))
                if str(cached["identity"]) != json.dumps(metadata, sort_keys=True):
                    raise ScenePreparationError(f"Cache identity mismatch: {candidate}")
                if not np.isclose(volume, metrics["volume_m3"], rtol=1e-8):
                    raise ScenePreparationError(f"Cache volume mismatch: {candidate}")
                if provenance.get("source_mesh_policy"):
                    if (not np.isclose(volume, metrics["volume_m3"], rtol=1e-10, atol=1e-18)
                            or not np.isclose(volume, surface.volume, rtol=1e-10, atol=1e-18)):
                        raise ScenePreparationError(f"Source cache volume mismatch: {candidate}")
                    _check_source_facets(surface, points, boundary)
                return points, tets, boundary, metrics
        except (OSError, KeyError, ValueError) as exc:
            raise ScenePreparationError(f"Invalid collision cache {candidate}: {exc}") from exc
    if path.exists():
        points, tets, boundary, metrics = read_cached(path, identity)
        return points, tets, boundary, {**metrics, "cache_status": "hit"}, path
    legacy_identity = {"format_version": 5, "provenance": provenance,
                       "allow_surface_repair": bool(allow_surface_repair), "max_surface_error": max_surface_error}
    legacy_path = cache_path(legacy_identity, 12, canonical_zeros=False)
    parts = surface.metadata.get("robodojo_collision_parts")
    migration = None
    if not legacy_path.exists():
        import trimesh

        for candidate, metadata in _compatible_cached_geometries(cache_dir, identity):
            candidate_points, candidate_tets, candidate_boundary, candidate_metrics = read_cached(candidate, metadata)
            output = trimesh.Trimesh(candidate_points, candidate_boundary, process=False)
            output.remove_unreferenced_vertices()
            forward, backward = _sampled_surface_errors(surface, output)
            if max(forward, backward) <= max_surface_error and output.is_watertight:
                candidate_metrics.update(source_to_tet_sampled_max_m=forward, tet_to_source_sampled_max_m=backward,
                                         cache_status="migrated_revalidated", migrated_from_cache=str(candidate))
                migration = candidate_points, candidate_tets, candidate_boundary, candidate_metrics
                break
    if legacy_path.exists():
        points, tets, boundary, metrics = read_cached(legacy_path, legacy_identity)
        metrics["cache_status"] = "migrated"
    elif migration is not None:
        points, tets, boundary, metrics = migration
    elif parts:
        results = []
        for index, part in enumerate(parts):
            part_provenance = {**provenance, "compound_part": index,
                               "colliders": [provenance["colliders"][index]],
                               "source_watertight": bool(part.is_watertight)}
            try:
                results.append(_cached_mesh(part, part_provenance, cache_dir,
                                            allow_surface_repair, max_surface_error))
            except ScenePreparationError as exc:
                raise ScenePreparationError(f"author collider {index}: {exc}") from exc
        points, tets, boundary, metrics = _combine_tetrahedra(results, "author_collider_compound", max_surface_error)
        metrics["part_cache_paths"] = [str(result[4]) for result in results]
        metrics["cache_status"] = ("assembled_cached_parts" if all(result[3].get("cache_status") in ("hit", "migrated")
                                                                   for result in results) else "built")
    else:
        points, tets, boundary, metrics = tetrahedralize_surface(surface,
                allow_surface_repair=allow_surface_repair, max_surface_error=max_surface_error)
        metrics["cache_status"] = "built"
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, points=points, tets=tets,
                            metrics=json.dumps(metrics), identity=json.dumps(identity, sort_keys=True))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return points, tets, boundary, metrics, path


def prepare_scene(scene_manager, *, cache_dir, stage=None, root_path="/World/TactileObjects",
                  allow_surface_repair=False, max_surface_error=0.0003,
                  extra_objects=(), source_meshes=None) -> PreparedScene:
    """Prepare hidden UIPC collision assets from current original layout poses.

    No changes are made to native assets. All conversions complete before any
    USD authoring, and partial authoring is removed on failure. Remove the previous
    returned ``root_path`` only after destroying its UIPC world before reuse.
    ``bindings`` uses exactly the ``spec['name']`` keys consumed by the manager.
    ``extra_objects`` accepts descriptors from ``collect_robot_colliders``.
    """
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, Vt
    from scipy.spatial.transform import Rotation

    stage = stage if stage is not None else scene_manager.stage
    if not Sdf.Path(root_path).IsAbsolutePath() or root_path in ("/", "/World"):
        raise ScenePreparationError("A dedicated absolute UIPC output subtree is required")
    if stage.GetPrimAtPath(root_path).IsValid():
        raise ScenePreparationError(f"UIPC output already exists; destroy old world first: {root_path}")
    options = _source_mesh_options({} if source_meshes is None else source_meshes)
    descriptors = [*collect_scene_objects(scene_manager), *extra_objects]
    names = [(obj.env_id, obj.instance_name) for obj in descriptors]
    paths = [obj.prim_path for obj in descriptors]
    if len(set(names)) != len(names) or len(set(paths)) != len(paths):
        raise ScenePreparationError("Duplicate native instance or source root in scene preparation")
    prepared = []
    for descriptor in descriptors:
        if (descriptor.prim_path == root_path or descriptor.prim_path.startswith(root_path + "/")
                or root_path.startswith(descriptor.prim_path + "/")):
            raise ScenePreparationError("UIPC output subtree overlaps a native object")
        policy = options.get(str(Path(descriptor.source_path).resolve())) if descriptor.source_path else None
        if policy:
            if descriptor.robot_key is not None or descriptor.object_type not in {"rigid", "geometry"}:
                raise ScenePreparationError("Source-mesh overrides only apply to native task rigid/geometry actors")
            selected = str(Sdf.Path(descriptor.prim_path).AppendPath(Sdf.Path(policy["prim_path"])))
            descriptor = replace(descriptor, collider_prim_paths=(selected,))
        try:
            surface, matrix, provenance = _read_collision_surface(stage, descriptor, selected_meshes=bool(policy))
            if policy:
                from importlib.metadata import PackageNotFoundError, version

                try:
                    tetgen_version = version("tetgen")
                except PackageNotFoundError as exc:
                    raise ScenePreparationError("Install tactile dependencies (tetgen==0.8.4) for source-mesh conversion") from exc
                raw_hash = _source_mesh_hash(surface)
                surface, canonicalization = _canonical_source_surface(surface)
                canonical_hash = _source_mesh_hash(surface)
                surface.metadata["source_conforming"] = True
                provenance["source_mesh_policy"] = {**policy, "version": 2, "tetgen_version": tetgen_version,
                    "raw_sha256": raw_hash, "canonical_sha256": canonical_hash,
                    "canonicalization": canonicalization}
            points, tets, boundary, metrics, cache_path = _cached_mesh(surface, provenance, cache_dir,
                    allow_surface_repair, max_surface_error)
            _LOGGER.info("Tactile scene env%s:%s cache=%s tetrahedra=%d", descriptor.env_id,
                         descriptor.instance_name, metrics.get("cache_status", "unknown"), len(tets))
        except ScenePreparationError as exc:
            raise ScenePreparationError(f"env{descriptor.env_id}:{descriptor.instance_name}: {exc}") from exc
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        pose = tuple(float(x) for x in [*matrix[:3, 3], q[3], *q[:3]])
        provenance = {**provenance, **metrics}
        prepared.append((descriptor, points, tets, boundary, pose, provenance, cache_path))
    specs, bindings = [], {}
    try:
        stage.DefinePrim(root_path, "Xform")
        for index, (descriptor, points, tets, boundary, pose, provenance, cache_path) in enumerate(prepared):
            # Numeric suffixes avoid invalid USD tokens in user asset names.
            path = f"{root_path}/env_{descriptor.env_id}/object_{index}"
            root = UsdGeom.Xform.Define(stage, path)
            root.SetResetXformStack(True)
            root.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
            root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(pose[3], Gf.Vec3d(*pose[4:])))
            mesh = UsdGeom.Mesh.Define(stage, path + "/mesh")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(boundary)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(boundary.reshape(-1).tolist()))
            mesh.CreateVisibilityAttr("invisible")
            mesh.CreateSubdivisionSchemeAttr("none")
            for key, kind, value in (
                ("tet_points", Sdf.ValueTypeNames.Point3dArray, Vt.Vec3dArray.FromNumpy(points)),
                ("tet_indices", Sdf.ValueTypeNames.IntArray, Vt.IntArray(tets.reshape(-1).tolist())),
                ("tet_surf_indices", Sdf.ValueTypeNames.IntArray, Vt.IntArray(boundary.reshape(-1).tolist())),
            ):
                mesh.GetPrim().CreateAttribute(key, kind, custom=True).Set(value)
            mesh.GetPrim().SetCustomDataByKey("robodojo:source", json.dumps(provenance, sort_keys=True))
            name = f"env{descriptor.env_id}:{descriptor.instance_name}"
            volume = provenance["volume_m3"]
            specs.append({"name": name, "prim_path": path, "usd_mesh_prim_name": "mesh",
                          "kinematic": not descriptor.dynamic,
                          "mass_density": descriptor.mass / volume if descriptor.dynamic else 1000.0,
                          "initial_pose": list(pose)})
            bindings[name] = SceneBinding(descriptor, name, path, pose, str(cache_path), volume, provenance,
                                         descriptor.robot_key, descriptor.body_name)
    except Exception:
        stage.RemovePrim(root_path)
        raise
    return PreparedScene(specs, bindings, root_path)
