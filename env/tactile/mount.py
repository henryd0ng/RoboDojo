"""Mount CAD surface cameras before the robot articulation starts playing."""

from __future__ import annotations

from typing import Any

from .config import TactileSensorConfig


def _single_env_path(path: str) -> str:
    """Resolve either Isaac namespace spelling to the supported environment."""
    return path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")


def _set_world_pose(stage: Any, path: str, pose: Any) -> None:
    """Publish a rigid body pose without changing its USD parent frame."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    from transforms3d.quaternions import mat2quat, quat2mat

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    position = np.asarray(pose[:3], dtype=float)
    rotation = quat2mat(pose[3:])
    if not xform.GetResetXformStack():
        parent = np.asarray(xform.ComputeParentToWorldTransform(Usd.TimeCode.Default())).T
        position = parent[:3, :3].T @ (position - parent[:3, 3])
        rotation = parent[:3, :3].T @ rotation
    quaternion = mat2quat(rotation)
    ops = xform.GetOrderedXformOps()
    if len(ops) < 2 or ops[0].GetOpType() != UsdGeom.XformOp.TypeTranslate or ops[1].GetOpType() != UsdGeom.XformOp.TypeOrient:
        reset_stack = xform.GetResetXformStack()
        xform.ClearXformOpOrder()
        translate = stage.GetPrimAtPath(path).GetAttribute("xformOp:translate")
        orient = stage.GetPrimAtPath(path).GetAttribute("xformOp:orient")
        ops = [UsdGeom.XformOp(translate) if translate else xform.AddTranslateOp(),
               UsdGeom.XformOp(orient) if orient else xform.AddOrientOp()]
        xform.SetXformOpOrder(ops, resetXformStack=reset_stack)
    vector_types = {
        UsdGeom.XformOp.PrecisionDouble: Gf.Vec3d,
        UsdGeom.XformOp.PrecisionFloat: Gf.Vec3f,
        UsdGeom.XformOp.PrecisionHalf: Gf.Vec3h,
    }
    quaternion_types = {
        UsdGeom.XformOp.PrecisionDouble: Gf.Quatd,
        UsdGeom.XformOp.PrecisionFloat: Gf.Quatf,
        UsdGeom.XformOp.PrecisionHalf: Gf.Quath,
    }
    ops[0].Set(vector_types[ops[0].GetPrecision()](*position.tolist()))
    orient_vector = vector_types[ops[1].GetPrecision()](*quaternion[1:].tolist())
    ops[1].Set(quaternion_types[ops[1].GetPrecision()](float(quaternion[0]), orient_vector))


def _mount_sensor_assets(sensor_cfg: TactileSensorConfig, robot_key: Any) -> tuple[str, str]:
    """Create a virtual camera behind the CAD pad already on the finger."""
    from isaacsim.core.utils import prims as prims_utils
    from omni.usd import get_context
    from .surface import mount_surface_camera

    root_expr = sensor_cfg.rigid_body_prim_path or robot_key.cfg.prim_path
    if not prims_utils.find_matching_prim_paths(root_expr):
        raise ValueError(f"No spawned articulation matches tactile rigid_body_prim_path={root_expr!r}.")
    mount_surface_camera(get_context().get_stage(), sensor_cfg)
    return _single_env_path(sensor_cfg.sensor_prim_path), _single_env_path(sensor_cfg.gelpad_prim_path)
