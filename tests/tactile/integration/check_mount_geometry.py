"""CPU-only source geometry of X5 tactile mounts; never starts Isaac or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def enable_cpu_usd():
    if importlib.util.find_spec("pxr") is not None:
        return
    site = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    isaac = site / "isaacsim"
    roots = sorted(isaac.glob("extscache/omni.usd.libs-*"))
    if not roots:
        raise RuntimeError("Install USD Python bindings or use the Isaac environment's bundled USD libraries")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(roots[-1]) + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(Path(sys.prefix) / "lib"), str(roots[-1] / "bin"), str(isaac / "kit"), env.get("LD_LIBRARY_PATH", "")]
    )
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def main(args):
    from pxr import Usd, UsdGeom, UsdPhysics

    def load(path):
        # Reference only the default prim; ignore unrelated source-stage props.
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/Asset")
        root.GetReferences().AddReference(str(path.resolve()))
        return stage, root

    def points(stage, root, *, substring=None):
        cache = UsdGeom.XformCache()
        inverse = cache.GetLocalToWorldTransform(root).GetInverse()
        arrays = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or (substring and substring not in str(prim.GetPath())):
                continue
            xyz = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim) * inverse)
            arrays.append(np.c_[xyz, np.ones(len(xyz))] @ matrix)
        if not arrays:
            raise ValueError(f"No mesh vertices below {root.GetPath()}")
        return np.concatenate(arrays)[:, :3]

    def bounds(xyz):
        return {"min": xyz.min(0).tolist(), "max": xyz.max(0).tolist(), "extent": np.ptp(xyz, axis=0).tolist()}

    def section_points(stage, root, y):
        cache = UsdGeom.XformCache()
        inverse = cache.GetLocalToWorldTransform(root).GetInverse()
        intersections = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            xyz = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim) * inverse)
            xyz = (np.c_[xyz, np.ones(len(xyz))] @ matrix)[:, :3]
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            triangles = []
            offset = 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face = indices[offset : offset + count]
                triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
                offset += count
            corners = xyz[np.asarray(triangles)]
            a, b = corners.reshape(-1, 3), np.roll(corners, -1, axis=1).reshape(-1, 3)
            dy = b[:, 1] - a[:, 1]
            mask = ((a[:, 1] - y) * (b[:, 1] - y) <= 0) & (np.abs(dy) > 1e-12)
            t = (y - a[mask, 1]) / dy[mask]
            intersections.append(a[mask] + t[:, None] * (b[mask] - a[mask]))
        return np.concatenate(intersections) if intersections else np.empty((0, 3))

    def quat(q):
        return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()

    def pose_points(xyz, pose):
        return xyz @ quat(pose[3:]).T + pose[:3]

    def gf_quat(value):
        return [value.GetReal(), *value.GetImaginary()]

    repo = args.repo.resolve()
    config = yaml.safe_load(args.config.read_text())
    sensors = config.get("tactile", config)
    if config.get("config", {}).get("robot") != "dual_x5_pad":
        raise ValueError("Geometry measurement requires the CAD dual_x5_pad configuration")
    robot_path = repo / "Assets/Robots/x5_pad/ARX.usda"
    manifest_path = robot_path.with_name("source_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    expected_sources = [
        (Path(manifest["source_urdf"]), manifest["source_urdf_sha256"]),
        (Path(manifest["stock_usd"]), manifest["stock_usd_sha256"]),
        *[(Path(pad["source_mesh"]), pad["source_mesh_sha256"]) for pad in manifest["pads"].values()],
    ]
    for path, digest in expected_sources:
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source-pad asset manifest is stale for {path}")
    source_paths = [robot_path, manifest_path, args.config.resolve(), Path(manifest["source_urdf"])]
    source_paths.extend(Path(pad["source_mesh"]) for pad in manifest["pads"].values())
    robot, robot_root = load(robot_path)
    source_paths.extend(
        Path(layer.identifier) for layer in robot.GetUsedLayers() if Path(layer.identifier).is_file()
    )
    fingers = {}
    for name, joint_name in (("left", "joint7"), ("right", "joint8")):
        body = sensors[name]["body_name"]
        body_prim = robot.GetPrimAtPath(f"/Asset/{body}")
        gel_prim = robot.GetPrimAtPath(f"/Asset/{body}/gelpad_{name}")
        if not gel_prim:
            raise ValueError(f"Source pad is missing below {body}")
        joint = UsdPhysics.PrismaticJoint(robot.GetPrimAtPath(f"/Asset/joints/{joint_name}"))
        r0, r1 = quat(gf_quat(joint.GetLocalRot0Attr().Get())), quat(gf_quat(joint.GetLocalRot1Attr().Get()))
        p0, p1 = np.asarray(joint.GetLocalPos0Attr().Get()), np.asarray(joint.GetLocalPos1Attr().Get())
        rotation_matrix = r0 @ r1.T
        axis = np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
        cache = UsdGeom.XformCache()
        gel_to_body = np.asarray(
            cache.GetLocalToWorldTransform(gel_prim) * cache.GetLocalToWorldTransform(body_prim).GetInverse()
        )
        gel_points = points(robot, gel_prim)
        # The source defines canonical +Z as the outward contact direction.
        front = gel_points[np.isclose(gel_points[:, 2], gel_points[:, 2].max(), atol=1e-7)]
        if not len(front):
            raise ValueError(f"Source contact surface is empty: {name}")
        front_body = (np.c_[front, np.ones(len(front))] @ gel_to_body)[:, :3]
        pad_body = (np.c_[gel_points, np.ones(len(gel_points))] @ gel_to_body)[:, :3]
        geometry = {}
        for mode, q in (("closed", joint.GetLowerLimitAttr().Get()), ("open", joint.GetUpperLimitAttr().Get())):
            offset = p0 + r0 @ (axis * q) - rotation_matrix @ p1
            geometry[mode] = {
                "q": q,
                "gel": bounds(pad_body @ rotation_matrix.T + offset),
                "contact_surface": bounds(front_body @ rotation_matrix.T + offset),
            }
        fingers[name] = {
            "body_name": body,
            "gelpad_source_prim": str(gel_prim.GetPath()),
            "gel_canonical_bounds": bounds(gel_points),
            "finger_local": bounds(points(robot, body_prim, substring="/collisions/")),
            "joint_p0": p0.tolist(),
            "joint_p1": p1.tolist(),
            "joint_axis_link6": (r0 @ axis).tolist(),
            "geometry": {"current": geometry},
        }
    gaps = {"current": {}}
    for mode in ("closed", "open"):
        left = fingers["left"]["geometry"]["current"][mode]["contact_surface"]
        right = fingers["right"]["geometry"]["current"][mode]["contact_surface"]
        gaps["current"][mode] = {"gel": left["min"][1] - right["max"][1]}
    a = fingers["left"]["geometry"]["current"]["closed"]["contact_surface"]
    b = fingers["right"]["geometry"]["current"]["closed"]["contact_surface"]
    gaps["current"]["tcp_link6"] = [
        (a["min"][0] + a["max"][0] + b["min"][0] + b["max"][0]) / 4,
        (a["min"][1] + b["max"][1]) / 2,
        (a["min"][2] + a["max"][2] + b["min"][2] + b["max"][2]) / 4,
    ]
    limits = [
        (fingers[name]["geometry"]["current"][mode]["q"])
        for name in ("left", "right")
        for mode in ("closed", "open")
    ]
    if not np.allclose(limits[:2], limits[2:], atol=1e-9):
        raise ValueError("The source pads require symmetric finger travel")
    gripper_range = limits[:2]
    closed_joint_position = gripper_range[0]
    case_bounds = gel_bounds = None
    objects = {}
    for task, category, index, label in (
        ("plug_in_charger", "charger", 0, "charger"),
        ("insert_tubes", "test_tube", 2, "tube1"),
        ("deposit_coin", "coin", 0, "coin0"),
    ):
        path = repo / f"Assets/Object/RoboDojo/Rigid/{category}/{index:05d}/object.usdz"
        source_paths.append(path)
        stage, root = load(path)
        xyz = points(stage, root)
        layout_path = repo / f"Assets/Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json"
        source_paths.append(layout_path)
        layout = json.loads(layout_path.read_text())
        item = next(item for group in layout["Rigid"].values() for item in group if item["label"] == label)
        xyz *= np.asarray(item["scale"])
        world = pose_points(xyz, item["default_pos"] + item["default_ori"])
        matrix = quat(item["default_ori"])
        closing_axis = matrix[:, 2 if category == "coin" else 0].copy()
        closing_axis[2] = 0
        closing_axis /= np.linalg.norm(closing_axis)
        width = float(np.ptp(world @ closing_axis))
        center = (xyz.min(0) + xyz.max(0)) / 2
        sections = {}
        for distance in (-0.01, 0.0, 0.01):
            cut = section_points(stage, root, (center[1] + distance) / item["scale"][1])
            cut *= np.asarray(item["scale"])
            cut_world = pose_points(cut, item["default_pos"] + item["default_ori"])
            sections[str(distance)] = float(np.ptp(cut_world @ closing_axis)) if len(cut_world) else None
        widths = [value for value in sections.values() if value is not None]
        objects[task] = {
            "target": label,
            "asset_path": str(path),
            "actual_local_bounds": bounds(xyz),
            "actual_center_local": center.tolist(),
            "actual_center_world": (matrix @ center + item["default_pos"]).tolist(),
            "top_down_closing_axis_world": closing_axis.tolist(),
            "actual_width_on_closing_axis": width,
            "central_section_widths": sections,
            "grasp_region_width_on_closing_axis": max(widths) if widths else width,
        }
        gap = gaps["current"]["closed"]["gel"]
        command = (closed_joint_position + (width - gap) / 2 - gripper_range[0]) / np.ptp(gripper_range)
        objects[task][
            "normalized_gripper_at_first_touch"
        ] = float(command)
    report = {
        "scope": "CPU source geometry; no dynamic contact or reachability claim",
        "env_cfg_path": str(args.config.resolve()),
        "robot_source_asset_path": str(source_paths[0].resolve()),
        "robot": config.get("config", {}).get("robot", "dual_x5"),
        "mounting": "source_fingertip_pads",
        "gripper_range_m": gripper_range,
        "closed_gap_joint_position_m": closed_joint_position,
        "case_root_bounds": case_bounds,
        "gel_root_bounds": gel_bounds,
        "fingers": fingers,
        "gap_m": gaps,
        "objects": objects,
        "sources_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gap_m": gaps, "objects": objects}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path, default=Path("env_cfg/arx_x5_pad_tactile.yml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enable_cpu_usd()
    main(args)
