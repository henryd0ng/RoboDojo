#!/usr/bin/env python3
"""Import the supplied X5A fingertip assembly without starting Isaac Sim.

Run with the robodojo-tactile Python environment. The visual skin preserves
the source STL geometry; a separately audited volume mesh supplies UIPC FEM.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import xml.etree.ElementTree as ET

try:
    import numpy as np
    from scipy.spatial import ConvexHull, Delaunay
    from shapely.geometry import Polygon
    import trimesh
    import yaml
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing build dependency {error.name}. "
        "Install numpy scipy shapely trimesh rtree tetgen PyYAML in this Python environment."
    ) from None


REPOSITORY = Path(__file__).resolve().parents[2]
STOCK_ASSET = REPOSITORY / "Assets/Robots/x5"
PAD_NAMES = {"left": "tactile_pad_left", "right": "tactile_pad_right"}
MINIMUM_OPENING = 0.001
# Sampling parameters for the supplied left/right CAD pads, not sensor defaults.
SECTION_RADIUS_MM = {"left": 0.8, "right": 0.9}


def load_usd():
    """Find the bundled USD libraries without importing the Isaac application."""
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    except ImportError:
        cache = Path(sysconfig.get_paths()["purelib"]) / "isaacsim/extscache"
        libraries = sorted(cache.glob("omni.usd.libs-*"))
        if not libraries or os.environ.get("ROBODOJO_PAD_USD_BOOTSTRAPPED"):
            raise RuntimeError("Use the robodojo-tactile Python environment with bundled USD libraries")
        library = libraries[-1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(library) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [
                str(library / "bin"),
                str(cache.parent / "kit"),
                str(Path(sys.prefix) / "lib"),
                environment.get("LD_LIBRARY_PATH", ""),
            ]
        )
        environment["ROBODOJO_PAD_USD_BOOTSTRAPPED"] = "1"
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(value: str) -> np.ndarray:
    return np.asarray([float(number) for number in value.split()])


def transform(origin) -> np.ndarray:
    result = trimesh.transformations.euler_matrix(*numbers(origin.get("rpy", "0 0 0")))
    result[:3, 3] = numbers(origin.get("xyz", "0 0 0"))
    return result


def read_pad(robot, source: Path, side: str):
    name = PAD_NAMES[side]
    link = robot.find(f"link[@name='{name}']")
    if link is None:
        raise ValueError(f"Missing source pad link: {name}")
    joint = next(joint for joint in robot.findall("joint") if joint.find("child").get("link") == name)
    visual = link.find("visual")
    geometry = visual.find("geometry/mesh")
    mesh_path = source / geometry.get("filename")
    skin = trimesh.load(mesh_path, force="mesh")
    if not skin.is_watertight or not skin.is_winding_consistent:
        raise ValueError(f"Source pad must be a closed, consistently oriented solid: {mesh_path}")
    scale = numbers(geometry.get("scale", "1 1 1"))
    skin.apply_scale(scale)
    source_pose = transform(joint.find("origin")) @ transform(visual.find("origin"))
    skin.apply_transform(source_pose)

    # Image X runs across the finger; image Y runs toward its tip. Local +Z
    # points from the backing toward the contacted object on both fingers.
    normal = source_pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    image_y = np.array([1.0, 0.0, 0.0])
    image_y -= normal * np.dot(image_y, normal)
    image_y /= np.linalg.norm(image_y)
    rotation = np.column_stack((np.cross(image_y, normal), image_y, normal))
    rotated = np.asarray(skin.vertices) @ rotation
    bounds = np.array([rotated.min(axis=0), rotated.max(axis=0)])
    origin = rotation @ np.array([*bounds.mean(axis=0)[:2], bounds[0, 2]])
    canonical = skin.copy()
    canonical.vertices = (np.asarray(skin.vertices) - origin) @ rotation
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, origin
    reconstructed = np.asarray(canonical.vertices) @ rotation.T + origin
    error = float(np.max(np.linalg.norm(reconstructed - skin.vertices, axis=1)))
    color = numbers(visual.find("material/color").get("rgba"))
    return canonical, pose, mesh_path, joint.find("parent").get("link"), color, error


def boundary_faces(tets: np.ndarray) -> np.ndarray:
    faces = tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, first, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(count > 2):
        raise ValueError("Volume mesh has a non-manifold internal face")
    return faces[first[count == 1]]


def surface_probes(mesh) -> np.ndarray:
    return np.vstack((mesh.vertices[np.unique(mesh.faces)], mesh.triangles_center))


def volume_mesh(skin, *, section_radius_mm: float):
    """Build a quality-gated FEM approximation from measured CAD sections.

    Six depth sections resolve the backing and rounded front, avoiding the
    source STL's nearly coincident edge vertices. Only FEM is approximated;
    the visual mesh and mounting transform retain their source geometry.
    """
    source_mm = skin.copy()
    source_mm.apply_scale(1000)
    low, high = source_mm.bounds
    if not np.isclose(high[2] - low[2], 1.0, atol=1e-6):
        raise ValueError("This source-specific FEM sampler requires the supplied 1 mm-thick pads")
    if not 0 < section_radius_mm <= high[2] - low[2]:
        raise ValueError("Section sampling radius must fit within the source pad thickness")
    depths = np.unique(
        np.r_[low[2], high[2] - section_radius_mm + section_radius_mm * np.sin(np.linspace(0, np.pi / 2, 5))]
    )
    rings = []
    for depth in depths:
        slicing_depth = min(high[2] - 1e-7, max(low[2] + 1e-7, depth))
        section = source_mm.section(plane_origin=[0, 0, slicing_depth], plane_normal=[0, 0, 1])
        if section is None or len(section.discrete) != 1:
            raise ValueError("Source pad must have one closed contour at each sampled depth")
        polygon = Polygon(section.discrete[0][:, :2]).convex_hull.simplify(0.005, preserve_topology=True)
        xy = np.asarray(polygon.exterior.coords)[:-1]
        rings.append(np.column_stack((xy, np.full(len(xy), depth))))
    hull = trimesh.convex.convex_hull(np.vstack(rings))
    low, high = hull.bounds
    axes = [
        np.linspace(low[index] + 0.3, high[index] - 0.3, max(2, int(np.ceil((high[index] - low[index]) / 1.25))))
        for index in (0, 1)
    ]
    # Sample both planar caps so the backing attachment covers the whole pad.
    xy = np.array(np.meshgrid(*axes, indexing="ij")).reshape(2, -1).T
    caps = np.vstack(
        (np.column_stack((xy, np.full(len(xy), low[2]))), np.column_stack((xy, np.full(len(xy), high[2]))))
    )
    equations = ConvexHull(hull.vertices).equations
    caps = caps[np.max(caps @ equations[:, :3].T + equations[:, 3], axis=1) <= 1e-8]
    try:
        import tetgen
    except ModuleNotFoundError:
        raise SystemExit("Missing build dependency tetgen. Install tetgen in this Python environment.") from None

    shell_points = np.vstack((hull.vertices, caps))
    shell_faces = np.asarray(hull.faces)
    planar = np.any(
        [np.all(np.abs(shell_points[shell_faces, 2] - depth) < 1e-8, axis=1) for depth in (low[2], high[2])], axis=0
    )
    face_groups = [shell_faces[~planar]]
    for depth, sign in ((low[2], -1), (high[2], 1)):
        indices = np.flatnonzero(np.abs(shell_points[:, 2] - depth) < 1e-8)
        triangles = indices[Delaunay(shell_points[indices, :2]).simplices]
        cross = np.cross(
            shell_points[triangles[:, 1]] - shell_points[triangles[:, 0]],
            shell_points[triangles[:, 2]] - shell_points[triangles[:, 0]],
        )[:, 2]
        triangles[cross * sign < 0] = triangles[cross * sign < 0][:, [0, 2, 1]]
        face_groups.append(triangles[np.abs(cross) > 1e-12])
    shell_faces = np.vstack(face_groups)
    shell = trimesh.Trimesh(vertices=shell_points, faces=shell_faces, process=False)
    if not shell.is_watertight:
        raise ValueError("Planar FEM sampling did not preserve a closed shell")
    interior_depths = low[2] + np.array([0.25, 0.5, 0.75])
    interior = np.array(np.meshgrid(*axes, interior_depths, indexing="ij")).reshape(3, -1).T
    # Staggering avoids zero-volume cells within a perfectly regular plane.
    interior += np.random.default_rng(7).uniform(-0.03, 0.03, interior.shape)
    interior = interior[hull.contains(interior)]
    seeds = np.vstack((shell_points, interior))
    mesher = tetgen.TetGen(seeds, shell_faces)
    points, tets, _, _ = mesher.tetrahedralize(
        plc=True,
        quality=True,
        minratio=1.5,
        mindihedral=5,
        steinerleft=1000,
        quiet=True,
    )
    points, tets = np.asarray(points), np.asarray(tets, dtype=np.int64)
    volumes = np.linalg.det(points[tets[:, 1:]] - points[tets[:, 0, None]]) / 6
    tets[volumes < 0] = tets[volumes < 0][:, [0, 2, 1, 3]]
    volumes = np.abs(volumes)
    if not np.all(volumes > 0):
        raise ValueError("FEM mesh contains zero-volume tetrahedra")
    boundary = boundary_faces(tets)
    boundary_skin = trimesh.Trimesh(vertices=points, faces=boundary, process=False)
    if not boundary_skin.is_watertight or not boundary_skin.is_winding_consistent:
        raise ValueError("FEM boundary is not a closed, consistently oriented surface")
    errors = []
    for probes, target in ((surface_probes(source_mm), boundary_skin), (surface_probes(boundary_skin), source_mm)):
        _, distances, _ = trimesh.proximity.closest_point(target, probes)
        errors.append(float(distances.max()))
    error = max(errors)
    relative_volume_error = abs(float(volumes.sum() / source_mm.volume - 1))
    if error > 0.05 or relative_volume_error > 0.01:
        raise ValueError(f"FEM boundary differs from source: {error:.6g} mm, volume error {relative_volume_error:.6g}")
    backing = np.flatnonzero(points[:, 2] <= points[:, 2].min() + 0.01)
    boundary_nodes = np.unique(boundary)
    cell_edges = np.stack(
        [points[tets[:, b]] - points[tets[:, a]] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], axis=1
    )
    mean_ratio = 12 * (3 * volumes) ** (2 / 3) / np.sum(cell_edges**2, axis=(1, 2))
    edge_matrix = points[tets[:, 1:]] - points[tets[:, 0, None]]
    circumcenters = np.linalg.solve(edge_matrix, np.sum(edge_matrix**2, axis=2)[..., None] / 2)[..., 0]
    circumradius = np.linalg.norm(circumcenters, axis=1)
    face_points = points[tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]]]
    face_area = (
        np.linalg.norm(
            np.cross(face_points[:, :, 1] - face_points[:, :, 0], face_points[:, :, 2] - face_points[:, :, 0]), axis=2
        )
        / 2
    )
    radius_ratio = 9 * volumes / (face_area.sum(axis=1) * circumradius)
    if not np.all(mean_ratio >= 0.001) or not np.all(radius_ratio >= 0.001):
        raise ValueError(
            f"FEM quality gate failed: mean ratio {mean_ratio.min():.6g}, radius ratio {radius_ratio.min():.6g}"
        )
    report = {
        "method": "sampled_source_cross_sections_tetgen_plc",
        "section_sampling_radius_mm": section_radius_mm,
        "section_sampling_parameter_origin": "Chosen for the supplied CAD rounded front; contours sampled from STL",
        "section_depths_mm": depths.tolist(),
        "section_contour_simplification_mm": 0.005,
        "interior_depth_bands": 3,
        "interior_depths_mm": interior_depths.tolist(),
        "interior_stagger_mm": 0.03,
        "lateral_grid_spacing_limit_mm": 1.25,
        "planar_skin_sample_spacing_limit_mm": 1.25,
        "nodes": len(points),
        "tetrahedra": len(tets),
        "boundary_triangles": len(boundary),
        "interior_nodes": len(points) - len(boundary_nodes),
        "backing_nodes": len(backing),
        "source_volume_mm3": float(source_mm.volume),
        "fem_volume_mm3": float(volumes.sum()),
        "relative_volume_error": relative_volume_error,
        "relative_volume_error_limit": 0.01,
        "minimum_tetrahedron_volume_mm3": float(volumes.min()),
        "minimum_mean_ratio": float(mean_ratio.min()),
        "mean_ratio_percentiles": np.percentile(mean_ratio, [1, 10, 50]).tolist(),
        "cells_below_mean_ratio_0_001": int(np.count_nonzero(mean_ratio < 0.001)),
        "minimum_normalized_radius_ratio": float(radius_ratio.min()),
        "cells_below_radius_ratio_0_001": int(np.count_nonzero(radius_ratio < 0.001)),
        "quality_gate_0_001_passed": True,
        "quality_warning": "Static mesh quality passed; runtime stability and mesh convergence remain unverified",
        "quality_steiner_node_limit": 1000,
        "sampled_bidirectional_surface_error_mm": error,
        "sampled_surface_error_limit_mm": 0.05,
        "fem_geometry_note": "Measured surface approximation; source visual geometry is preserved separately",
        "boundary_watertight": True,
        "all_tetrahedron_volumes_positive": True,
        "mesh_convergence_verified": False,
    }
    return points / 1000, tets, boundary, report


def write_gel(path, skin, points, tets, boundary, color, usd):
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt = usd
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/gelpad")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/gelpad/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(skin.vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(skin.faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(skin.faces.reshape(-1).tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([Gf.Vec3f(*skin.bounds[0]), Gf.Vec3f(*skin.bounds[1])])
    for name, kind, value in (
        ("tet_points", Sdf.ValueTypeNames.Point3dArray, Vt.Vec3dArray.FromNumpy(points)),
        ("tet_indices", Sdf.ValueTypeNames.IntArray, Vt.IntArray(tets.reshape(-1).tolist())),
        ("tet_surf_indices", Sdf.ValueTypeNames.IntArray, Vt.IntArray(boundary.reshape(-1).tolist())),
    ):
        mesh.GetPrim().CreateAttribute(name, kind, custom=True).Set(value)
    material = UsdShade.Material.Define(stage, "/gelpad/material")
    shader = UsdShade.Shader.Define(stage, "/gelpad/material/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def write_assembly(output: Path, pads, usd):
    Gf, _, Usd, UsdGeom, UsdPhysics, _, _ = usd
    original = Usd.Stage.Open(str(STOCK_ASSET / "ARX.usd"))
    root_path = str(original.GetDefaultPrim().GetPath())
    stage = Usd.Stage.CreateNew(str(output / "ARX.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().GetReferences().AddReference(os.path.relpath(STOCK_ASSET / "ARX.usd", output), root_path)
    stage.SetDefaultPrim(root.GetPrim())
    for side, pad in pads.items():
        path = f"{root_path}/{pad['parent']}/gelpad_{side}"
        gel = UsdGeom.Xform.Define(stage, path)
        gel.GetPrim().GetReferences().AddReference(f"gelpad_{side}.usda", "/gelpad")
        gel.AddTransformOp().Set(Gf.Matrix4d(pad["pose"].T.tolist()))
    for name in ("joint7", "joint8"):
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(f"{root_path}/joints/{name}"))
        if not joint:
            raise ValueError(f"Stock USD is missing {name}")
        joint.GetLowerLimitAttr().Set(MINIMUM_OPENING)
    stage.GetRootLayer().Save()
    return root_path


def write_urdf(robot, output: Path):
    for link in robot.findall("link"):
        side = next((side for side, name in PAD_NAMES.items() if link.get("name") == name), None)
        for mesh in link.findall(".//mesh"):
            if side:
                mesh.set("filename", f"meshes/gelpad_{side}.stl")
            else:
                mesh.set("filename", os.path.relpath(STOCK_ASSET / mesh.get("filename"), output))
    for name in ("joint7", "joint8"):
        robot.find(f"joint[@name='{name}']/limit").set("lower", str(MINIMUM_OPENING))
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(output / "X5A_pad.urdf", encoding="utf-8", xml_declaration=True)


def write_runtime_metadata(output: Path):
    """Keep the native action schema and planner model with the new limits."""
    metadata = yaml.safe_load((STOCK_ASSET / "robot_config.yml").read_text())
    metadata["urdf_path"] = "./X5A_pad.urdf"
    metadata["gripper_scale"] = [MINIMUM_OPENING, 0.044]
    (output / "robot_config.yml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    planner = yaml.safe_load((STOCK_ASSET / "curobo.yml").read_text())
    kinematics = planner["robot_cfg"]["kinematics"]
    kinematics["asset_root_path"] = "."
    kinematics["urdf_path"] = "X5A_pad.urdf"
    (output / "curobo.yml").write_text(
        "# Stock planning spheres do not cover the fingers or tactile pads.\n"
        + yaml.safe_dump(planner, sort_keys=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "Assets/Robots/x5_pad")
    arguments = parser.parse_args()
    source, output = arguments.source_dir.resolve(), arguments.output_dir.resolve()
    usd = load_usd()
    robot = ET.parse(source / "X5A_pad.urdf").getroot()
    output.mkdir(parents=True, exist_ok=True)
    (output / "meshes").mkdir(exist_ok=True)
    pads, reports = {}, {}
    for side in PAD_NAMES:
        skin, pose, mesh_path, parent, color, reconstruction_error = read_pad(robot, source, side)
        points, tets, boundary, report = volume_mesh(skin, section_radius_mm=SECTION_RADIUS_MM[side])
        shutil.copyfile(mesh_path, output / f"meshes/gelpad_{side}.stl")
        write_gel(output / f"gelpad_{side}.usda", skin, points, tets, boundary, color, usd)
        pads[side] = {"parent": parent, "pose": pose}
        reports[side] = {
            "source_mesh": str(mesh_path),
            "source_mesh_sha256": digest(mesh_path),
            "source_link": PAD_NAMES[side],
            "parent_link": parent,
            "canonical_to_parent": pose.tolist(),
            "canonical_bounds_m": skin.bounds.tolist(),
            "source_transform_reconstruction_error_m": reconstruction_error,
            "source_visual_vertices": len(skin.vertices),
            "source_visual_triangles": len(skin.faces),
            "backing_gap_to_original_finger_mm": 0.102584,
            "fem": report,
        }
    root_path = write_assembly(output, pads, usd)
    write_urdf(robot, output)
    write_runtime_metadata(output)
    manifest = {
        "source_urdf": str(source / "X5A_pad.urdf"),
        "source_urdf_sha256": digest(source / "X5A_pad.urdf"),
        "stock_usd": str(STOCK_ASSET / "ARX.usd"),
        "stock_usd_sha256": digest(STOCK_ASSET / "ARX.usd"),
        "default_prim": root_path,
        "units": "metres",
        "pads": reports,
        "changes": ["Normalize mesh paths", "Set joint7 and joint8 minimum travel to 0.001 m"],
        "unchanged": [
            "Source visual pad geometry", "Source mounting transforms", "Original robot USD and finger collisions"
        ],
        "preview_behavior": "Pad meshes are visual-only until the tactile manager creates UIPC FEM objects",
        "minimum_finger_travel_m": MINIMUM_OPENING,
        "source_zero_travel_pad_overlap_mm": 1.398002,
        "nominal_front_plane_gap_at_minimum_travel_mm": 0.601998,
        "fem_boundary_conditions": (
            "Constrain backing nodes to measured finger motion with a soft position penalty; no rigid-body API on pads"
        ),
        "optical_camera": "Created by tactile configuration; canonical +Z points toward the contact object",
        "dependencies": {
            name: version(name) for name in ("numpy", "scipy", "shapely", "trimesh", "rtree", "tetgen", "PyYAML")
        },
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({side: report["fem"] for side, report in reports.items()}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"Missing build dependency {error.name}. "
            "Install numpy scipy shapely trimesh rtree tetgen PyYAML in this Python environment."
        ) from None
