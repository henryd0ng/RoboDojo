"""Virtual optical cameras for deformable pads already mounted on a robot."""

from __future__ import annotations

import math
from typing import Any


def _concrete_path(path: str) -> str:
    return path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")


def surface_optical_height(stage: Any, sensor_cfg: Any) -> float:
    """Use the mounted pad thickness to align object depth with its front face."""
    sensor = stage.GetPrimAtPath(_concrete_path(sensor_cfg.sensor_prim_path))
    spans = sensor.GetAttribute("robodojo:gelSpanMm").Get()
    height = float(spans[2]) / 1000.0
    near, _ = sensor_cfg.camera_clipping_range
    if not math.isclose(near + height, sensor_cfg.camera_focus_distance, abs_tol=1e-7):
        raise ValueError("Surface object optics require near + source pad thickness = focus distance")
    return height


def mount_surface_camera(stage: Any, sensor_cfg: Any) -> tuple[float, ...]:
    """Mount an orthographic camera behind a pad whose outward axis is +Z.

    The camera inherits the finger transform and has no rigid body or housing.
    Its returned pose is relative to that finger. Apertures are also recorded
    so TacEx can retain this physical field of view when it wraps the camera.
    """
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    gel_path = _concrete_path(sensor_cfg.gelpad_prim_path)
    sensor_path = _concrete_path(sensor_cfg.sensor_prim_path)
    body_path = str(Sdf.Path(sensor_path).GetParentPath())
    body = stage.GetPrimAtPath(body_path)
    gel = stage.GetPrimAtPath(gel_path)
    if not body.IsValid() or body.GetName() != sensor_cfg.body_name:
        raise ValueError(f"Surface camera must be mounted directly below {sensor_cfg.body_name!r}: {sensor_path}")
    if not gel.IsValid() or not gel_path.startswith(body_path + "/"):
        raise ValueError(f"Surface gel must be mounted below the same finger: {gel_path}")
    meshes = [prim for prim in Usd.PrimRange(gel) if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError(f"Surface gel requires exactly one physical mesh: {gel_path}")
    mesh = meshes[0]
    nodes = np.asarray(mesh.GetAttribute("tet_points").Get(), dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 4 or not np.isfinite(nodes).all():
        raise ValueError(f"Surface gel requires finite N x 3 tetrahedral points: {gel_path}")
    lower, upper = nodes.min(axis=0), nodes.max(axis=0)
    spans = upper - lower
    if np.any(spans <= 0):
        raise ValueError(f"Surface gel must have positive width, length, and thickness: {gel_path}")

    field = np.asarray(getattr(sensor_cfg, "optical_field_mm", None), dtype=float)
    if field.shape != (2,) or not np.isfinite(field).all() or np.any(field <= 0):
        raise ValueError("Surface mounting requires positive optical_field_mm: [width, height]")
    focus = float(sensor_cfg.camera_focus_distance)
    if not np.isfinite(focus) or focus <= 0:
        raise ValueError("Surface camera focus distance must be positive and finite")
    metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(metres_per_unit) or metres_per_unit <= 0:
        raise ValueError("Surface camera requires a positive stage metres-per-unit value")

    # USD stores row-vector transforms; convert once to column-vector form.
    time = Usd.TimeCode.Default()
    body_world = np.asarray(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(time), dtype=float).T
    mesh_world = np.asarray(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(time), dtype=float).T
    mesh_to_body = np.linalg.solve(body_world, mesh_world)
    scales = np.linalg.norm(mesh_to_body[:3, :3], axis=0)
    if np.any(scales <= 0):
        raise ValueError(f"Surface gel has a singular mounting transform: {gel_path}")
    rotation = mesh_to_body[:3, :3] / scales
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0):
        raise ValueError(f"Surface gel mounting must not contain shear or reflection: {gel_path}")
    front = (lower + upper) / 2
    front[2] = upper[2]
    front_body = (mesh_to_body @ np.r_[front, 1.0])[:3]
    position = front_body - rotation[:, 2] * (focus / metres_per_unit)
    physical_spans = spans * scales * metres_per_unit * 1000.0

    # USD cameras look down -Z. A half-turn about X makes them look out of
    # the gel; a quarter-turn maps its longer in-plane dimension to width.
    camera_rotation = rotation @ np.diag([1.0, -1.0, -1.0])
    width_axis = "x"
    if physical_spans[1] > physical_spans[0]:
        quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        camera_rotation = rotation @ quarter_turn @ np.diag([1.0, -1.0, -1.0])
        width_axis = "y"
    transform = np.eye(4)
    transform[:3, :3] = camera_rotation
    transform[:3, 3] = position
    usd_transform = Gf.Matrix4d(transform.T.tolist())
    quaternion = usd_transform.ExtractRotation().GetQuat()
    pose = tuple(float(value) for value in (*position, quaternion.GetReal(), *quaternion.GetImaginary()))

    existing = stage.GetPrimAtPath(sensor_path)
    if existing.IsValid() and existing.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"A virtual surface camera must not be a rigid body: {sensor_path}")
    sensor = UsdGeom.Xform.Define(stage, sensor_path)
    sensor.ClearXformOpOrder()
    attribute = sensor.GetPrim().GetAttribute("xformOp:transform:surfaceMount")
    op = UsdGeom.XformOp(attribute) if attribute else sensor.AddTransformOp(opSuffix="surfaceMount")
    sensor.SetXformOpOrder([op], resetXformStack=False)
    op.Set(usd_transform)

    # USD apertures are measured in tenths of a stage unit, including for
    # orthographic cameras: 20 mm at metre scale corresponds to 0.2.
    aperture = field * 0.01 / metres_per_unit
    camera = UsdGeom.Camera.Define(stage, f"{sensor_path}/Camera")
    camera.ClearXformOpOrder()
    camera.SetResetXformStack(False)
    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
    camera.CreateHorizontalApertureAttr().Set(float(aperture[0]))
    camera.CreateVerticalApertureAttr().Set(float(aperture[1]))
    camera.CreateHorizontalApertureOffsetAttr().Set(0.0)
    camera.CreateVerticalApertureOffsetAttr().Set(0.0)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*(value / metres_per_unit for value in sensor_cfg.camera_clipping_range)))
    camera.CreateFocalLengthAttr().Set(float(sensor_cfg.camera_focal_length))
    camera.CreateFocusDistanceAttr().Set(focus / metres_per_unit)
    prim = sensor.GetPrim()
    prim.CreateAttribute("robodojo:mountPose", Sdf.ValueTypeNames.DoubleArray, custom=True).Set(pose)
    prim.CreateAttribute("robodojo:opticalFieldMm", Sdf.ValueTypeNames.Double2, custom=True).Set(Gf.Vec2d(*field))
    prim.CreateAttribute("robodojo:cameraAperture", Sdf.ValueTypeNames.Double2, custom=True).Set(Gf.Vec2d(*aperture))
    prim.CreateAttribute("robodojo:surfaceCenterBody", Sdf.ValueTypeNames.Double3, custom=True).Set(Gf.Vec3d(*front_body))
    prim.CreateAttribute("robodojo:gelSpanMm", Sdf.ValueTypeNames.Double3, custom=True).Set(Gf.Vec3d(*physical_spans))
    prim.CreateAttribute("robodojo:gelWidthAxis", Sdf.ValueTypeNames.Token, custom=True).Set(width_axis)
    return pose


def restore_surface_geometry(stage: Any, gel_path: str, source_usd: str, robot_root: str) -> None:
    """Restore source gel geometry and its local frame without touching a finger.

    Restore attributes in place: replacing a prim beneath an active robot
    invalidates PhysX's articulation view even when its collider is disabled.
    Source transforms and FEM data are restored without altering prims, APIs,
    references, materials, sibling cameras, or live articulation transforms.
    """
    from pxr import Sdf, Usd, UsdGeom

    gel_path, robot_root = _concrete_path(gel_path), _concrete_path(robot_root).rstrip("/")
    if not gel_path.startswith(robot_root + "/"):
        raise ValueError(f"Surface gel {gel_path!r} is outside robot {robot_root!r}")
    source = Usd.Stage.Open(str(source_usd))
    if source is None or not source.GetDefaultPrim().IsValid():
        raise ValueError(f"Surface source asset requires a default prim: {source_usd}")
    source_path = str(source.GetDefaultPrim().GetPath()) + gel_path[len(robot_root):]
    source_gel = source.GetPrimAtPath(source_path)
    if not source_gel.IsValid() or not any(prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(source_gel)):
        raise ValueError(f"Surface source asset has no gel mesh at {source_path}")
    geometry_attributes = (
        "points", "faceVertexCounts", "faceVertexIndices", "extent", "normals",
        "orientation", "doubleSided", "subdivisionScheme", "tet_points",
        "tet_indices", "tet_surf_indices", "surf_points", "surf_indices",
    )
    pairs = []
    for original in Usd.PrimRange(source_gel):
        if not original.IsA(UsdGeom.Xformable):
            continue
        path = gel_path + str(original.GetPath())[len(source_path):]
        live = stage.GetPrimAtPath(path)
        if not live.IsValid() or live.GetTypeName() != original.GetTypeName():
            raise ValueError(f"Surface geometry must already exist with its source type: {path}")
        pairs.append((original, live))

    def restore_attribute(original, live, name):
        attribute = original.GetAttribute(name)
        current = live.GetAttribute(name)
        if not attribute.IsValid():
            if current.IsValid():
                current.Clear()
            return
        value = attribute.Get()
        if not current.IsValid():
            if value is None:
                return
            current = live.CreateAttribute(name, attribute.GetTypeName(), custom=attribute.IsCustom())
        if value is None:
            current.Clear()
        else:
            current.Set(value)

    with Sdf.ChangeBlock():
        for original, live in pairs:
            original_xform = UsdGeom.Xformable(original)
            live_xform = UsdGeom.Xformable(live)
            restored_ops = []
            for op in original_xform.GetOrderedXformOps():
                name = op.GetAttr().GetName()
                restore_attribute(original, live, name)
                restored_ops.append(UsdGeom.XformOp(live.GetAttribute(name), op.IsInverseOp()))
            live_xform.SetXformOpOrder(restored_ops, resetXformStack=original_xform.GetResetXformStack())
            restore_attribute(original, live, "visibility")
            if original.IsA(UsdGeom.Mesh):
                for name in geometry_attributes:
                    restore_attribute(original, live, name)
                UsdGeom.Mesh(live).SetNormalsInterpolation(UsdGeom.Mesh(original).GetNormalsInterpolation())
