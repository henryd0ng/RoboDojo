"""Configuration and validation for the optional RoboDojo tactile plugin.

This module has no Isaac/TacEx imports so it can be used by config validation
and unit tests in a plain Python environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping


class TactileConfigError(ValueError):
    """Raised when a tactile configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class TactileSensorConfig:
    """Configuration for one physical GelSight pad."""

    name: str
    body_name: str
    gelpad_prim_path: str
    sensor_prim_path: str
    # Filled from the CAD surface camera after mounting; never guessed in YAML.
    sensor_pose: tuple[float, float, float, float, float, float, float] = (0, 0, 0, 1, 0, 0, 0)
    camera_clipping_range: tuple[float, float] = (0.024, 0.034)
    camera_focal_length: float = 1.05
    camera_focus_distance: float = 0.028
    calibration_dir: str | None = None
    calibration_id: str | None = None
    # Optional explicit root for the Isaac articulation. When omitted the
    # adapter derives it from the initialized RoboDojo articulation config.
    rigid_body_prim_path: str | None = None
    # Geometry and pose come from the robot's CAD pad.
    mounting: str = "surface"
    # Physical width/height of a custom pad's orthographic optical field.
    # Taxim retains its calibrated spatial response while normals use this
    # explicit millimetre scale.
    optical_field_mm: tuple[float, float] | None = None
    # Backing attachment penalty relative to nodal mass, not a hard clamp.
    attachment_strength_ratio: float = 1e5
    # Optional rest-material stiffness multiplier, replacing the mass-only
    # penalty on backing nodes while preserving the legacy default.
    attachment_stiffness_multiplier: float | None = None
    # Opt-in target-transform arithmetic; raw measured link poses are unchanged.
    attachment_target_dtype: str = "float32"


@dataclass(frozen=True)
class TactileConfig:
    """Public RoboDojo tactile configuration.

    The first implementation intentionally supports exactly two GelSight Mini
    RGB streams.  Additional outputs can be added without changing the
    manager/observation boundary.
    """

    enabled: bool = False
    render_in_vision: bool = False
    sensor_type: str = "gsmini"
    optical_backend: str = "taxim"
    optical_input: str = "deformed_surface"
    collision_detection_method: str = "linear_bvh"
    # Newton displacement tolerance is this velocity (m/s) times physics dt.
    solver_velocity_tolerance: float = 0.05
    # Dimensionless squared residual ratio in the PCG preconditioner metric.
    solver_linear_tolerance: float = 0.001
    resolution: tuple[int, int] = (128, 128)
    update_hz: float = 30.0
    device: str = "cuda"
    uipc_workspace: str = ""
    calibration_dir: str | None = None
    stabilise_steps: int = 10
    max_initial_pose_correction_m: float = 0.0
    left: TactileSensorConfig | None = None
    right: TactileSensorConfig | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    # Work budget only; exceeding it still fails strict convergence.
    solver_max_iterations: int = 1024

    @property
    def period(self) -> float:
        return 1.0 / self.update_hz


def _as_range(value: Any, label: str, *, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        value = default
    values = tuple(float(v) for v in value)
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise TactileConfigError(f"{label} must contain two finite numbers [near, far].")
    if values[0] <= 0 or values[1] <= values[0]:
        raise TactileConfigError(f"{label} must satisfy 0 < near < far.")
    return values


def _as_positive_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TactileConfigError(f"{label} must be positive and finite.") from exc
    if not math.isfinite(number) or number <= 0:
        raise TactileConfigError(f"{label} must be positive and finite.")
    return number


def _as_solver_max_iterations(value: Any) -> int:
    # libuipc stores newton/max_iter in IndexT (int32_t).
    if type(value) is not int or not 1 <= value <= 2**31 - 1:
        raise TactileConfigError("tactile.solver_max_iterations must be an integer in [1, 2147483647].")
    return value


def _as_attachment_target_dtype(value: Any, label: str) -> str:
    if type(value) is not str or value not in {"float32", "float64"}:
        raise TactileConfigError(f"{label} must be 'float32' or 'float64'.")
    return value


def _as_linear_tolerance(value: Any) -> float:
    label = "tactile.solver_linear_tolerance"
    number = _as_positive_float(value, label)
    if isinstance(value, bool) or number > 1:
        raise TactileConfigError(f"{label} must satisfy 0 < tolerance <= 1.")
    return number


def _sensor_from_mapping(name: str, data: Mapping[str, Any], defaults: TactileConfig) -> TactileSensorConfig:
    body_name = str(data.get("body_name", "")).strip()
    if not body_name:
        raise TactileConfigError(f"tactile.{name}.body_name is required when tactile is enabled.")
    gelpad_path = str(data.get("gelpad_prim_path", "")).strip()
    sensor_path = str(data.get("sensor_prim_path", "")).strip()
    if not gelpad_path or not sensor_path:
        raise TactileConfigError(
            f"tactile.{name} requires both gelpad_prim_path and sensor_prim_path."
        )
    mounting = str(data.get("mounting", "surface"))
    if mounting != "surface":
        raise TactileConfigError(f"tactile.{name}.mounting must be 'surface'.")
    if any(key in data for key in ("pose", "gelpad_pose", "sensor_pose", "camera_pose")):
        raise TactileConfigError(f"tactile.{name}: surface mounting uses the source USD pose; omit pose overrides.")
    camera_clipping_range = _as_range(
        data.get("camera_clipping_range", data.get("clipping_range")),
        f"tactile.{name}.camera_clipping_range",
        default=(0.024, 0.034),
    )
    camera_focal_length = float(data.get("camera_focal_length", data.get("focal_length", 1.05)))
    camera_focus_distance = float(data.get("camera_focus_distance", data.get("focus_distance", 0.028)))
    if not math.isfinite(camera_focal_length) or camera_focal_length <= 0:
        raise TactileConfigError(f"tactile.{name}.camera_focal_length must be positive and finite.")
    if not math.isfinite(camera_focus_distance) or camera_focus_distance <= 0:
        raise TactileConfigError(f"tactile.{name}.camera_focus_distance must be positive and finite.")
    try:
        attachment_strength_ratio = float(data.get("attachment_strength_ratio", 1e5))
    except (TypeError, ValueError) as exc:
        raise TactileConfigError(
            f"tactile.{name}.attachment_strength_ratio must be positive and finite."
        ) from exc
    if not math.isfinite(attachment_strength_ratio) or attachment_strength_ratio <= 0:
        raise TactileConfigError(f"tactile.{name}.attachment_strength_ratio must be positive and finite.")
    stiffness_multiplier = data.get("attachment_stiffness_multiplier")
    if stiffness_multiplier is not None:
        try:
            stiffness_multiplier = float(stiffness_multiplier)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TactileConfigError(
                f"tactile.{name}.attachment_stiffness_multiplier must be positive and finite."
            ) from exc
        if not math.isfinite(stiffness_multiplier) or stiffness_multiplier <= 0:
            raise TactileConfigError(
                f"tactile.{name}.attachment_stiffness_multiplier must be positive and finite."
            )
    optical_field = data.get("optical_field_mm")
    if optical_field is not None:
        optical_field = tuple(float(v) for v in optical_field)
        if len(optical_field) != 2 or any(not math.isfinite(v) or v <= 0 for v in optical_field):
            raise TactileConfigError(f"tactile.{name}.optical_field_mm must contain positive [width, height].")
        if not math.isclose(optical_field[0] / optical_field[1], defaults.resolution[1] / defaults.resolution[0], rel_tol=1e-6):
            raise TactileConfigError(f"tactile.{name}.optical_field_mm must match the image aspect ratio.")
    if optical_field is None:
        raise TactileConfigError(f"tactile.{name}.surface mounting requires optical_field_mm.")
    return TactileSensorConfig(
        name=name,
        body_name=body_name,
        gelpad_prim_path=gelpad_path,
        sensor_prim_path=sensor_path,
        camera_clipping_range=camera_clipping_range,
        camera_focal_length=camera_focal_length,
        camera_focus_distance=camera_focus_distance,
        calibration_dir=data.get("calibration_dir") or defaults.calibration_dir,
        calibration_id=str(data.get("calibration_id", name)),
        rigid_body_prim_path=data.get("rigid_body_prim_path"),
        mounting=mounting,
        optical_field_mm=optical_field,
        attachment_strength_ratio=attachment_strength_ratio,
        attachment_stiffness_multiplier=stiffness_multiplier,
        attachment_target_dtype=_as_attachment_target_dtype(
            data.get("attachment_target_dtype", "float32"),
            f"tactile.{name}.attachment_target_dtype",
        ),
    )


def load_tactile_config(raw: Mapping[str, Any] | None) -> TactileConfig:
    """Convert an OmegaConf/dict node into a validated plain dataclass."""

    data = dict(raw or {})
    for name in ("left", "right"):
        sensor_data = data.get(name)
        if isinstance(sensor_data, Mapping) and "attachment_target_dtype" in sensor_data:
            _as_attachment_target_dtype(
                sensor_data["attachment_target_dtype"], f"tactile.{name}.attachment_target_dtype"
            )
    resolution = tuple(int(v) for v in data.get("resolution", (128, 128)))
    if len(resolution) != 2:
        raise TactileConfigError("tactile.resolution must be [height, width].")
    defaults = TactileConfig(
        enabled=bool(data.get("enabled", False)),
        render_in_vision=data.get("render_in_vision", False),
        sensor_type=str(data.get("sensor_type", "gsmini")),
        optical_backend=str(data.get("optical_backend", "taxim")),
        optical_input=str(data.get("optical_input", "deformed_surface")),
        collision_detection_method=str(data.get("collision_detection_method", "linear_bvh")),
        solver_velocity_tolerance=_as_positive_float(
            data.get("solver_velocity_tolerance", 0.05), "tactile.solver_velocity_tolerance"
        ),
        solver_linear_tolerance=_as_linear_tolerance(data.get("solver_linear_tolerance", 0.001)),
        solver_max_iterations=_as_solver_max_iterations(data.get("solver_max_iterations", 1024)),
        resolution=resolution,
        update_hz=float(data.get("update_hz", 30.0)),
        device=str(data.get("device", "cuda")),
        uipc_workspace=str(data.get("uipc_workspace", "")),
        calibration_dir=data.get("calibration_dir"),
        stabilise_steps=int(data.get("stabilise_steps", 10)),
        max_initial_pose_correction_m=float(data.get("max_initial_pose_correction_m", 0.0)),
        extra={k: v for k, v in data.items() if k not in {
            "enabled", "render_in_vision", "sensor_type", "optical_backend", "optical_input", "resolution", "update_hz", "device",
            "uipc_workspace", "calibration_dir",
            "stabilise_steps", "left", "right", "collision_detection_method", "max_initial_pose_correction_m",
            "solver_velocity_tolerance", "solver_linear_tolerance", "solver_max_iterations",
        }},
    )
    if defaults.enabled:
        object.__setattr__(defaults, "left", _sensor_from_mapping("left", data.get("left") or {}, defaults))
        object.__setattr__(defaults, "right", _sensor_from_mapping("right", data.get("right") or {}, defaults))
    validate_tactile_config(defaults)
    return defaults


def validate_tactile_config(config: TactileConfig) -> None:
    """Validate values that can be checked without launching Isaac Sim."""

    if not isinstance(config.render_in_vision, bool):
        raise TactileConfigError("tactile.render_in_vision must be a boolean.")
    if config.sensor_type != "gsmini":
        raise TactileConfigError(f"Only sensor_type='gsmini' is supported in v1, got {config.sensor_type!r}.")
    if config.optical_backend != "taxim":
        raise TactileConfigError(
            f"Only optical_backend='taxim' is supported in v1, got {config.optical_backend!r}."
        )
    if config.optical_input not in {"object", "deformed_surface"}:
        raise TactileConfigError("tactile.optical_input must be 'object' or 'deformed_surface'.")
    if config.collision_detection_method not in {"linear_bvh", "stackless_bvh"}:
        raise TactileConfigError(
            "tactile.collision_detection_method must be 'linear_bvh' or 'stackless_bvh'."
        )
    _as_positive_float(config.solver_velocity_tolerance, "tactile.solver_velocity_tolerance")
    _as_linear_tolerance(config.solver_linear_tolerance)
    _as_solver_max_iterations(config.solver_max_iterations)
    for sensor in (config.left, config.right):
        if sensor is not None:
            _as_attachment_target_dtype(
                sensor.attachment_target_dtype, f"tactile.{sensor.name}.attachment_target_dtype"
            )
    height, width = config.resolution
    if (height, width) not in {(128, 128), (240, 320)}:
        raise TactileConfigError("Supported HWC resolutions are [128, 128] and [240, 320].")
    if not math.isfinite(config.update_hz) or config.update_hz <= 0:
        raise TactileConfigError("tactile.update_hz must be positive and finite.")
    if config.enabled and not config.device.lower().startswith("cuda"):
        raise TactileConfigError(
            "Taxim GelSight simulation requires a CUDA device in v1; "
            f"got tactile.device={config.device!r}."
        )
    if not math.isfinite(config.max_initial_pose_correction_m) or config.max_initial_pose_correction_m < 0:
        raise TactileConfigError("tactile.max_initial_pose_correction_m must be finite and non-negative.")
    if config.stabilise_steps < 0:
        raise TactileConfigError("tactile.stabilise_steps must be non-negative.")
    if config.enabled and (config.left is None or config.right is None):
        raise TactileConfigError("Enabled tactile mode requires both left and right sensors.")
    if config.enabled and config.left is not None and config.right is not None:
        if config.left.sensor_prim_path == config.right.sensor_prim_path:
            raise TactileConfigError("tactile.left.sensor_prim_path and tactile.right.sensor_prim_path must differ.")
        if config.left.gelpad_prim_path == config.right.gelpad_prim_path:
            raise TactileConfigError("tactile.left.gelpad_prim_path and tactile.right.gelpad_prim_path must differ.")
        for sensor in (config.left, config.right):
            if sensor.camera_clipping_range[0] >= sensor.camera_clipping_range[1]:
                raise TactileConfigError(f"tactile.{sensor.name}.camera_clipping_range must be increasing.")
            if sensor.mounting != "surface":
                raise TactileConfigError(f"tactile.{sensor.name}.mounting must be 'surface'.")
            nominal_depth = sensor.camera_focus_distance
            if config.optical_input == "object" and not sensor.camera_clipping_range[0] < nominal_depth < sensor.camera_clipping_range[1]:
                raise TactileConfigError(
                    f"tactile.{sensor.name}.camera_clipping_range must cover the nominal plane of the gel "
                    "in object mode (camera focus distance)."
                )


def validate_calibration_dir(path: str | Path | None, *, resolution: tuple[int, int] = (128, 128)) -> None:
    """Validate a Taxim calibration directory, when one is configured."""

    if path is None:
        return
    root = Path(path).expanduser()
    # Taxim's torch implementation consumes all five files below. Checking
    # only params/gelmap/shadowTable lets a broken calibration fail much later
    # inside the first optical render.
    required = ("params.json", "gelmap.npy", "shadowTable.npz", "polycalib.npz", "dataPack.npz")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise TactileConfigError(f"Taxim calibration directory {root} is missing: {', '.join(missing)}")
    try:
        params = json.loads((root / "params.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise TactileConfigError(f"Taxim calibration params.json is not valid JSON: {root / 'params.json'}") from exc
    if not isinstance(params, dict):
        raise TactileConfigError("Taxim calibration params.json must contain a JSON object.")
    # NumPy is supplied by Isaac Sim/TacEx. Keep the plain config module
    # importable without it, while checking the calibration array shapes and
    # keys whenever it is available.
    try:
        import numpy as np

        gelmap = np.load(root / "gelmap.npy", mmap_mode="r")
        if gelmap.ndim < 2 or 0 in gelmap.shape:
            raise TactileConfigError(f"Taxim gelmap.npy has invalid shape {gelmap.shape}.")
        required_npz_keys = {
            "shadowTable.npz": (),
            "polycalib.npz": ("grad_b", "grad_g", "grad_r"),
            "dataPack.npz": ("f0",),
        }
        for filename, keys in required_npz_keys.items():
            try:
                with np.load(root / filename, allow_pickle=True) as archive:
                    missing_keys = [key for key in keys if key not in archive.files]
                    if missing_keys:
                        raise TactileConfigError(
                            f"Taxim calibration {filename} is missing arrays: {', '.join(missing_keys)}"
                        )
            except TactileConfigError:
                raise
            except Exception as exc:
                raise TactileConfigError(
                    f"Taxim calibration {filename} could not be loaded: {root / filename}"
                ) from exc
    except ModuleNotFoundError:
        pass
    except TactileConfigError:
        raise
    except Exception as exc:
        raise TactileConfigError(f"Taxim gelmap.npy could not be loaded: {root / 'gelmap.npy'}") from exc
    if resolution not in {(128, 128), (240, 320)}:
        raise TactileConfigError("Unsupported tactile calibration output resolution.")
