"""Lazy runtime validation and shared conversions for tactile simulation.

The bridge is intentionally lazy. RoboDojo can be imported and evaluated in
its stock vision-only environment without TacEx installed. When enabled, a
missing or incompatible TacEx installation fails during scene setup with a
diagnostic that names the optional packages.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import numpy as np

from .config import (
    TactileSensorConfig,
)


class TactileDependencyError(RuntimeError):
    """Raised when enabled tactile mode cannot load its optional backend."""


def _set_manual_runtime_device(obj: Any, device: str) -> None:
    """Populate IsaacLab callback fields before invoking a private init hook.

    TacEx/UIPC normally fills these fields from its PLAY callback. RoboDojo
    can create the adapter after the timeline is already playing, so its
    deterministic fallback calls the same private hooks directly. The hooks
    read ``device`` while allocating buffers; without this small bridge they
    fail before the first tactile frame.
    """

    if getattr(obj, "_device", None) is None:
        setattr(obj, "_device", device)


def _validate_runtime_versions() -> None:
    """Fail early when installed Isaac distributions contradict RoboDojo v1.

    Isaac Sim is commonly delivered as an application rather than a Python
    distribution, so an unavailable metadata entry is reported at startup but
    is not treated as a mismatch. A known, incompatible version is fatal.
    """

    expected = {
        "isaacsim": os.environ.get("ROBODOJO_TACTILE_ISAAC_SIM", "5.1"),
        "isaaclab": os.environ.get("ROBODOJO_TACTILE_ISAAC_LAB", "2.3"),
        "torch": os.environ.get("ROBODOJO_TACTILE_TORCH", "2.7"),
        "torch-scatter": os.environ.get("ROBODOJO_TACTILE_TORCH_SCATTER", "2.1.2+pt27cu128"),
        "tacex": os.environ.get("ROBODOJO_TACTILE_TACEX", "0.1.0"),
        "tacex-assets": os.environ.get("ROBODOJO_TACTILE_TACEX_ASSETS", "0.1.0"),
        "tacex-uipc": os.environ.get("ROBODOJO_TACTILE_TACEX_UIPC", "0.1.0"),
        "pyuipc": os.environ.get("ROBODOJO_TACTILE_PYUIPC", "0.9.0"),
    }
    mismatches = []
    for distribution, required_prefix in expected.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        if not installed.startswith(required_prefix):
            mismatches.append(f"{distribution}={installed} (requires {required_prefix}.*)")
    if mismatches:
        raise TactileDependencyError(
            "RoboDojo tactile v1 requires the Isaac Sim 5.1 / Isaac Lab 2.3 "
            "runtime; detected incompatible packages: " + ", ".join(mismatches)
        )
    try:
        import torch

        expected_cuda = os.environ.get("ROBODOJO_TACTILE_CUDA", "12.8")
        if torch.version.cuda != expected_cuda:
            mismatches.append(f"torch CUDA={torch.version.cuda} (requires {expected_cuda})")
        if not torch.cuda.is_available():
            mismatches.append("torch.cuda.is_available=False (Taxim/UIPC requires a CUDA GPU)")
    except Exception as exc:
        mismatches.append(f"torch import failed: {exc}")
    if mismatches:
        raise TactileDependencyError("RoboDojo tactile v1 runtime compatibility check failed: " + "; ".join(mismatches))


def _require_backend_modules() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    required = {
        "tacex": "tacex",
        "tacex_assets": "tacex_assets",
        "tacex_uipc": "tacex_uipc",
        "libuipc": "uipc",
        # TorchScatter is loaded by the UIPC/TacEx tensor path. Import it
        # during preflight so an ABI/undefined-symbol error is reported before
        # a sensor or USD reference is created.
        "torch_scatter": "torch_scatter",
    }
    errors = []
    for label, module_name in required.items():
        try:
            modules[label] = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised in Isaac env
            errors.append(f"{module_name}: {exc}")
    if errors:
        raise TactileDependencyError(
            "RoboDojo tactile mode requires the RoboDojo-local TacEx stack "
            "(tacex, tacex_assets, tacex_uipc, libuipc, torch_scatter). Install the optional "
            "tactile dependencies before setting tactile.enabled=true.\n- " + "\n- ".join(errors)
        )
    return modules


def _to_numpy_rgb(value: Any, resolution: tuple[int, int]) -> np.ndarray:
    """Normalize TacEx output to the public HWC uint8 RGB contract."""

    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"TacEx tactile RGB must be HWC or BCHW-like, got {arr.shape}.")
    if arr.shape[-1] not in (3, 4) and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] not in (3, 4):
        raise ValueError(f"TacEx tactile RGB must have 3 or 4 channels, got {arr.shape}.")
    arr = arr[..., :3]
    if tuple(arr.shape[:2]) != tuple(resolution):
        raise ValueError(f"TacEx tactile RGB shape {arr.shape[:2]} does not match configured resolution {resolution}.")
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            raise ValueError("TacEx tactile RGB contains non-finite values.")
        if float(np.nanmax(arr, initial=0.0)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
    # Match TacEx's torch-side conversion, which rounds normalized optical
    # output before casting instead of truncating every fractional channel.
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _iter_robot_candidates(robot_manager: Any) -> Iterable[tuple[Any, Any]]:
    yield from zip(getattr(robot_manager, "robot_list", []), getattr(robot_manager, "robot_key", []))


def _prim_expr_tail(value: Any) -> str:
    """Return the final concrete-ish component of a USD prim expression."""

    return str(value or "").rstrip("/").split("/")[-1]


def _robot_matches_root_expr(key: Any, root_expr: str | None) -> bool:
    """Match an explicitly configured articulation root to a RobotManager key.

    Isaac Lab stores the root as ``{ENV_REGEX_NS}/robotN`` while user YAML
    commonly uses the expanded ``/World/envs/env_.*/robotN`` form. Comparing
    the final component keeps both forms equivalent and avoids selecting the
    first of two otherwise identical X5 articulations.
    """

    if not root_expr:
        return False
    configured_tail = _prim_expr_tail(root_expr)
    key_tail = _prim_expr_tail(getattr(getattr(key, "cfg", None), "prim_path", ""))
    if configured_tail and key_tail and configured_tail == key_tail:
        return True
    # A custom expression may contain a regular-expression suffix rather than
    # a literal robot name. In that case compare the normalized expressions.
    configured = str(root_expr).replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    key_expr = str(getattr(getattr(key, "cfg", None), "prim_path", ""))
    key_expr = key_expr.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    try:
        return bool(re.fullmatch(configured, key_expr)) or bool(re.fullmatch(key_expr, configured))
    except re.error:
        return configured == key_expr


def _resolve_robot_for_sensor(robot_manager: Any, sensor_cfg: TactileSensorConfig) -> tuple[Any, Any]:
    """Resolve one robot key, honoring the configured articulation root.

    Body names such as ``link7`` and ``link8`` occur on every X5 arm in the
    default dual-arm embodiment. A body-name-only lookup therefore silently
    attaches a sensor to the first X5. The tactile config already carries a
    validated ``rigid_body_prim_path``; use it to select the v1 gripper and
    fail loudly when a multi-robot config omits that information.
    """

    candidates = [
        (robot, key)
        for robot, key in _iter_robot_candidates(robot_manager)
        if sensor_cfg.body_name in list(getattr(key, "body_names", []))
    ]
    # During DirectRLEnv scene setup IsaacLab has created the Articulation
    # wrappers but has not initialized their PhysX views yet, so
    # ``robot_manager.robot_key`` is still empty.  Resolve the configured root
    # directly to the corresponding scene object; body/joint indices are
    # filled later by RobotManager.reset().
    if not candidates and not getattr(robot_manager, "robot_key", []):
        scene = getattr(robot_manager, "scene", None)
        if scene is not None and sensor_cfg.rigid_body_prim_path:
            robots = list(getattr(robot_manager, "robot_list", []))
            for idx, robot in enumerate(robots):
                try:
                    key = scene[f"robot{idx}"]
                except Exception:
                    continue
                if _robot_matches_root_expr(key, sensor_cfg.rigid_body_prim_path):
                    return robot, key
    if not candidates:
        available = []
        for _, key in _iter_robot_candidates(robot_manager):
            available.extend(str(name) for name in getattr(key, "body_names", []))
        raise ValueError(
            f"Tactile attachment body {sensor_cfg.body_name!r} was not found in the initialized robot. "
            f"Available body names: {available}"
        )
    if sensor_cfg.rigid_body_prim_path:
        rooted = [pair for pair in candidates if _robot_matches_root_expr(pair[1], sensor_cfg.rigid_body_prim_path)]
        if len(rooted) == 1:
            return rooted[0]
        if len(rooted) > 1:
            candidates = rooted
    if len(candidates) == 1:
        return candidates[0]
    roots = [str(getattr(getattr(key, "cfg", None), "prim_path", "<unknown>")) for _, key in candidates]
    raise ValueError(
        f"Tactile body {sensor_cfg.body_name!r} is present on multiple robots ({roots}). "
        "Set tactile.<side>.rigid_body_prim_path to the validated articulation root."
    )


def _validate_sensor_prims(sensor_cfg: TactileSensorConfig) -> None:
    """Fail with a useful message before TacEx constructs a USD view."""

    try:
        from isaacsim.core.utils import prims as prims_utils
        from omni.usd import get_context
        from pxr import Usd, UsdGeom
    except Exception as exc:  # pragma: no cover - Isaac-only path
        raise TactileDependencyError("Isaac Sim prim utilities are unavailable in tactile mode.") from exc
    for label, path in (("gelpad", sensor_cfg.gelpad_prim_path), ("sensor", sensor_cfg.sensor_prim_path)):
        try:
            matches = prims_utils.find_matching_prim_paths(path)
        except Exception as exc:
            raise ValueError(f"Invalid tactile.{sensor_cfg.name}.{label}_prim_path={path!r}: {exc}") from exc
        if not matches:
            raise ValueError(
                f"No USD prim matches tactile.{sensor_cfg.name}.{label}_prim_path={path!r}. "
                "Add the calibrated GelSight asset to the robot USD before enabling tactile mode."
            )

        # ``find_matching_prim_paths`` only confirms that the Xform exists. A
        # malformed or partially loaded reference would otherwise fail later
        # inside UIPC/GelSight with an opaque child lookup error. Validate the
        # concrete first-environment prim and its required payload here.
        concrete = str(matches[0])
        stage = get_context().get_stage()
        root = stage.GetPrimAtPath(concrete)
        if not root.IsValid():
            raise ValueError(f"tactile.{sensor_cfg.name}.{label} prim is invalid: {concrete}")
        descendants = list(Usd.PrimRange(root))
        if label == "sensor":
            camera_path = f"{concrete}/Camera"
            camera = stage.GetPrimAtPath(camera_path)
            if not camera.IsValid() or not camera.IsA(UsdGeom.Camera):
                raise ValueError(
                    f"GelSight asset at {concrete} has no Camera child at {camera_path}. "
                    "Check the surface camera mounting before enabling tactile mode."
                )
        elif not any(child.IsA(UsdGeom.Mesh) for child in descendants):
            raise ValueError(
                f"UIPC gelpad asset at {concrete} contains no UsdGeom.Mesh descendant. "
                "Check the CAD pad mesh before enabling tactile mode."
            )


def _default_calibration_dir() -> str:
    """Return the bundled GSmini Taxim response calibration."""
    try:
        from tacex_assets import TACEX_SENSORS_DATA_DIR
    except Exception as exc:  # pragma: no cover - Isaac-only path
        raise TactileDependencyError("TacEx asset package does not expose TACEX_SENSORS_DATA_DIR.") from exc
    return str(Path(TACEX_SENSORS_DATA_DIR) / "GelSight_Mini" / "calibs" / "640x480")
