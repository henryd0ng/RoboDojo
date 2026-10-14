"""TacEx gel physics, optical capture, and render synchronization."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from .config import TactileSensorConfig
from .runtime import TactileDependencyError, _set_manual_runtime_device, _to_numpy_rgb

if TYPE_CHECKING:
    import numpy as np


def _backplate_indices(local_points) -> list[int]:
    """Select the GSmini backing plane in mesh-local metres.

    Node IDs depend on the physical tetrahedral mesh. A refined asset adds
    backing nodes as well as free nodes, so a fixed vendor ID list is unsafe.
    """
    import numpy as np

    points = np.asarray(local_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Gel tetrahedral points must be a finite nonempty N x 3 array")
    minimum = float(points[:, 2].min())
    indices = np.flatnonzero(points[:, 2] <= minimum + 1e-5)
    if len(indices) == len(points):
        raise ValueError("GSmini gel must have free nodes above its local backing plane")
    return indices.tolist()


def _mounted_backplate_indices(prim_path: str) -> list[int]:
    """Read the mounted physical mesh before UIPC initializes attachments."""
    import omni.usd
    from pxr import Usd, UsdGeom

    path = prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"Mounted gel prim is missing: {path}")
    meshes = [child for child in Usd.PrimRange(prim) if child.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError(f"GSmini gel requires one physical mesh: {path}")
    return _backplate_indices(meshes[0].GetAttribute("tet_points").Get())


class _TacExGelSight:
    """Small wrapper containing all TacEx/UIPC implementation details."""

    def __init__(
        self,
        *,
        sensor_cfg: TactileSensorConfig,
        parent_robot: Any,
        rigid_body_prim_path: str,
        uipc_sim: Any,
        resolution: tuple[int, int],
        update_period: float,
        optical_backend: str,
        device: str,
        calibration_dir: str | None,
        optical_input: str = "deformed_surface",
        owner_registry: dict | None = None,
    ):
        # Publish ownership before constructing any backend component. The
        # manager can then release a gel whose optical camera fails to build.
        self.sensor_cfg = sensor_cfg
        self.optical_input = optical_input
        self.gelpad = None
        self.attachment = None
        self.sensor = None
        self._reference_depth = None
        # Nodal state is allocated later by PLAY/manual initialization.
        self._nominal_state = None
        self._resolution = resolution
        self._last_rgb: np.ndarray | None = None
        self._valid = False
        self._last_update_ms = 0.0
        self._last_update_time: float | None = None
        self._render_proxy_mesh = None
        self._render_proxy_path = None
        if owner_registry is not None:
            owner_registry[sensor_cfg.name] = self

        from tacex import GelSightSensor

        from tacex_assets.sensors.gelsight_mini import (
            GELSIGHT_MINI_PIX2PIX_CFG,
            GELSIGHT_MINI_TAXIM_CFG,
        )
        from tacex_uipc import UipcDeformableObject, UipcDeformableObjectCfg, UipcIsaacAttachmentsCfg

        attachment_options = {}
        if sensor_cfg.attachment_target_dtype == "float64":
            if "target_dtype" not in getattr(UipcIsaacAttachmentsCfg, "__dataclass_fields__", {}):
                raise TactileDependencyError(
                    "Float64 attachment targets require the current TacEx target_dtype patch."
                )
            attachment_options["target_dtype"] = "float64"

        if optical_backend == "taxim":
            optical_cfg = GELSIGHT_MINI_TAXIM_CFG.copy()
        elif optical_backend == "pix2pix":
            optical_cfg = GELSIGHT_MINI_PIX2PIX_CFG.copy()
        else:
            raise ValueError(f"Unsupported GelSight optical backend: {optical_backend!r}")

        optical_cfg = optical_cfg.replace(
            prim_path=sensor_cfg.sensor_prim_path,
            sensor_camera_cfg=optical_cfg.SensorCameraCfg(
                prim_name="Camera",
                resolution=(resolution[1], resolution[0]),
                # RoboDojo calls capture() at the observation boundary and
                # passes force_recompute=True.  Keeping the underlying RTX
                # camera period at zero prevents Isaac Lab's 30 Hz sensor
                # scheduler from returning a four-physics-step-old frame
                # when the control loop is faster than the tactile rate.
                update_period=0.0,
                data_types=["depth", "rgb"],
                clipping_range=sensor_cfg.camera_clipping_range,
                focal_length=sensor_cfg.camera_focal_length,
                focus_distance=sensor_cfg.camera_focus_distance,
                update_latest_camera_pose=True,
            ),
            device=device,
            debug_vis=False,
            update_period=0.0,
            marker_motion_sim_cfg=None,
            data_types=["tactile_rgb"],
        )
        if sensor_cfg.optical_field_mm is not None:
            from omni.usd import get_context
            from pxr import UsdGeom

            units = UsdGeom.GetStageMetersPerUnit(get_context().get_stage())
            width, height = sensor_cfg.optical_field_mm
            optical_cfg = optical_cfg.replace(sensor_camera_cfg=optical_cfg.sensor_camera_cfg.replace(
                horizontal_aperture=width * 0.01 / units,
                vertical_aperture=height * 0.01 / units,
            ))
        if optical_cfg.optical_sim_cfg is None:
            raise TactileDependencyError("The selected GelSight Mini Taxim configuration has no optical simulator.")
        if optical_input == "object":
            from omni.usd import get_context
            from .surface import surface_optical_height

            height = surface_optical_height(get_context().get_stage(), sensor_cfg)
            optical_cfg = optical_cfg.replace(
                gelpad_dimensions=optical_cfg.gelpad_dimensions.replace(height=height),
            )
        optical_sim_cfg = optical_cfg.optical_sim_cfg.replace(
            tactile_img_res=(resolution[1], resolution[0]), device=device
        )
        if calibration_dir:
            optical_sim_cfg = optical_sim_cfg.replace(calib_folder_path=str(Path(calibration_dir).expanduser()))
        optical_cfg = optical_cfg.replace(optical_sim_cfg=optical_sim_cfg)
        if optical_backend == "taxim":
            if not hasattr(optical_cfg.optical_sim_cfg, "input_mode"):
                raise TactileDependencyError(
                    "The installed TacEx lacks selectable optical inputs. "
                    "Install the current RoboDojo tactile source patches before enabling tactile mode."
                )
            optical_cfg = optical_cfg.replace(
                optical_sim_cfg=optical_cfg.optical_sim_cfg.replace(
                    with_shadow=False, input_mode=optical_input
                ),
                compute_indentation_depth_class="optical_sim",
            )

        gelpad_cfg = UipcDeformableObjectCfg(
            prim_path=sensor_cfg.gelpad_prim_path,
            constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e3,
            constraint_cfg=UipcIsaacAttachmentsCfg(
                # Backing-plane penalty relative to nodal mass, not a hard clamp.
                constraint_strength_ratio=sensor_cfg.attachment_strength_ratio,
                body_name=sensor_cfg.body_name,
                isaaclab_rigid_body_prim_path=rigid_body_prim_path,
                attachment_point_indices=_mounted_backplate_indices(sensor_cfg.gelpad_prim_path),
                debug_vis=False,
                **attachment_options,
            ),
        )
        self.gelpad = UipcDeformableObject(gelpad_cfg, uipc_sim)
        self.attachment = self.gelpad.constraint
        self.attachment.isaaclab_rigid_object = parent_robot
        if sensor_cfg.attachment_stiffness_multiplier is not None:
            self._set_attachment_stiffness(gelpad_cfg)
        self.sensor = GelSightSensor(optical_cfg, self.gelpad)

    def _set_attachment_stiffness(self, gelpad_cfg) -> None:
        """Set material-relative backing penalties before UIPC world initialization."""
        import numpy as np
        from uipc import view

        from .attachment import material_backing_strengths

        geometry = self.gelpad.geo_slot_list[0].geometry()
        points = np.asarray(geometry.positions().view()).reshape(-1, 3)
        tetrahedra = np.asarray(geometry.tetrahedra().topo().view()).reshape(-1, 4)
        indices = np.asarray(self.attachment.attachment_points_idx)
        strengths = material_backing_strengths(
            points,
            tetrahedra,
            indices,
            youngs_modulus_pa=gelpad_cfg.constitution_cfg.youngs_modulus * 1e6,  # TacEx config uses MPa.
            poisson_ratio=gelpad_cfg.constitution_cfg.poisson_rate,
            mass_density=gelpad_cfg.mass_density,
            dt=self.gelpad.uipc_sim.cfg.dt,
            multiplier=self.sensor_cfg.attachment_stiffness_multiplier,
        )
        attribute = geometry.vertices().find("strength_ratio")
        if attribute is None:
            raise RuntimeError("UIPC backing constraint has no strength_ratio attribute")
        # AttributeSlot.view() is const in PyUIPC; the top-level view API
        # explicitly obtains mutable storage owned by the scene geometry.
        values = np.asarray(view(attribute)).reshape(-1)
        if len(values) != len(points) or values.dtype.kind != "f":
            raise RuntimeError("UIPC backing strength_ratio must have one float per vertex")
        representable = strengths.astype(values.dtype)
        if not np.isfinite(representable).all() or np.any(representable <= 0):
            raise RuntimeError("Computed backing strengths are outside the UIPC attribute's numeric range")
        values[indices] = representable

    def sync_mount_pose(self) -> None:
        """Publish the virtual camera pose from the measured finger transform."""

        import numpy as np
        import omni.usd
        from transforms3d.quaternions import qmult, quat2mat

        stage = omni.usd.get_context().get_stage()
        sensor_path = self.sensor_cfg.sensor_prim_path.replace("/World/envs/env_.*/", "/World/envs/env_0/")
        body_path = sensor_path.rsplit("/", 1)[0]
        body = stage.GetPrimAtPath(body_path)
        housing = stage.GetPrimAtPath(sensor_path)
        if not body.IsValid() or not housing.IsValid():
            return
        # Isaac Lab's PhysX articulation view is authoritative during a
        # joint-state write.  The USD stage transform can remain at its
        # authored value until the next Fabric synchronization, which is too
        # late for the same render fence.  TacEx's attachment has already
        # sampled that live link pose in ``obj_pose`` for its aim positions;
        # use it here as the camera mount pose as well.
        live_pose = getattr(self.attachment, "obj_pose", None)
        if live_pose is not None:
            try:
                live = np.asarray(live_pose[0, 0].detach().cpu(), dtype=float).reshape(7)
            except Exception:
                live = None
        else:
            live = None
        if live is not None:
            body_pos = live[:3]
            body_quat = live[3:7]
        else:
            body_matrix = omni.usd.get_world_transform_matrix(body)
            body_pos = np.asarray(body_matrix.ExtractTranslation(), dtype=float)
            body_quat_obj = body_matrix.ExtractRotation().GetQuat()
            body_quat = [
                float(body_quat_obj.GetReal()),
                *[float(v) for v in body_quat_obj.GetImaginary()],
            ]
        local_pose = np.asarray(self.sensor_cfg.sensor_pose, dtype=float)
        world_pos = body_pos + quat2mat(body_quat) @ local_pose[:3]
        world_quat = qmult(body_quat, local_pose[3:])
        from .mount import _set_world_pose

        _set_world_pose(stage, sensor_path, [*world_pos, *world_quat])

    def ensure_initialized(self) -> None:
        """Ensure TacEx's PLAY callback has initialized the sensor handles.

        RoboDojo creates the adapter during scene setup, while TacEx normally
        receives its initialization from the next Isaac timeline PLAY event.
        The explicit fallback makes the boundary deterministic when the
        adapter is created after PLAY (for example after an in-process reset).
        """
        self._ensure_gelpad_initialized()
        self._ensure_attachment_initialized()

        # The normal PLAY event initializes the camera and optical simulator.
        # A manager can also be created after PLAY (for example after an
        # in-process stage reset), so use TacEx's own initialization hook when
        # the event did not reach this sensor. Checking ``_is_initialized`` is
        # important here: the private buffers can survive a STOP event while
        # their Isaac handles have already been invalidated.
        if getattr(self.sensor, "_is_initialized", False):
            self._configure_surface_optics()
            self._remember_nominal_state()
            return
        try:
            import omni.timeline

            is_playing = omni.timeline.get_timeline_interface().is_playing()
        except Exception:
            is_playing = True
        if not is_playing:
            raise RuntimeError(
                f"TacEx sensor {self.sensor.cfg.prim_path!r} is not initialized; "
                "start Isaac timeline before capturing tactile data."
            )
        self.sensor._initialize_impl()
        self.sensor._is_initialized = True
        self._configure_surface_optics()
        self._remember_nominal_state()

    def _configure_surface_optics(self) -> None:
        """Use physical slopes with the inherited calibrated RGB response.

        The polynomial retains its original image coordinates. Only the
        height-to-normal scale follows this camera's field in millimetres.
        This defines a virtual sensor, not a new hardware calibration.
        """
        field = self.sensor_cfg.optical_field_mm
        if field is None:
            return
        model = self.sensor.optical_simulator._taxim
        width, height = field
        if not math.isclose(width / model.width, height / model.height, rel_tol=1e-6):
            raise ValueError("Custom optical field must match the Taxim calibration aspect ratio")
        model.sensor_params.pixmm = width / model.width

    def _ensure_gelpad_initialized(self) -> None:
        if getattr(self.gelpad, "_is_initialized", False) and getattr(self.gelpad, "_state_accessor", None) is not None:
            return
        initialize_impl = getattr(self.gelpad, "_initialize_impl", None)
        if not callable(initialize_impl):
            raise RuntimeError("TacEx UIPC gelpad does not expose an initialization hook.")
        _set_manual_runtime_device(self.gelpad, str(self.gelpad.uipc_sim.cfg.device))
        initialize_impl()
        self.gelpad._is_initialized = True

    def _ensure_attachment_initialized(self) -> None:
        """Initialize the Isaac attachment when the PLAY callback was missed."""

        attachment = self.attachment
        if getattr(attachment, "_is_initialized", False):
            return
        initialize_impl = getattr(attachment, "_initialize_impl", None)
        if not callable(initialize_impl):
            raise RuntimeError("TacEx UIPC gelpad attachment does not expose an initialization hook.")
        # Attachments transform Isaac body poses, whose device can differ
        # from the CUDA-only contact solver (RoboDojo defaults to CPU PhysX).
        _set_manual_runtime_device(attachment, str(attachment.isaaclab_rigid_object.device))
        initialize_impl()
        attachment._is_initialized = True

    def _remember_nominal_state(self) -> None:
        if self._nominal_state is not None:
            return
        try:
            nominal_state = self.gelpad.default_nodal_state_w
        except (AttributeError, RuntimeError):
            return
        if nominal_state is not None:
            self._nominal_state = nominal_state.clone()

    def reset(self) -> None:
        self._valid = False
        self._last_rgb = None
        self._reference_depth = None
        self._last_update_time = None
        self._last_update_ms = 0.0
        reset = getattr(self.gelpad, "reset", None)
        if callable(reset):
            reset()
        # The sensor is constructed before PLAY during scene setup. Its
        # buffers do not exist yet, so defer reset until the first live
        # episode if Isaac has not initialized it.
        if getattr(self.sensor, "_is_initialized", False):
            reset_sensor = getattr(self.sensor, "reset", None)
            if callable(reset_sensor):
                reset_sensor()
        self._remember_nominal_state()
        write_state = getattr(self.gelpad, "write_nodal_state_to_sim", None)
        # UIPC creates the default nodal tensor while authoring the object, but
        # its state accessor is allocated only by the PLAY callback.  The
        # manager is reset once during scene setup (before PLAY), so defer the
        # explicit state write until the accessor exists.
        if (
            callable(write_state)
            and self._nominal_state is not None
            and getattr(self.gelpad, "_state_accessor", None) is not None
        ):
            write_state(self._nominal_state.clone())

    def physics_step(self, dt: float) -> None:
        # UIPC advances from its Isaac physics callback even before the
        # adapter's first live capture.  The deformable object's data buffers
        # are allocated lazily, so do not dereference ``_data`` during the
        # pre-PLAY/reset steps.
        if not getattr(self.gelpad, "_is_initialized", False) or getattr(self.gelpad, "_data", None) is None:
            return
        self.gelpad.update(dt=dt)

    def capture(self, dt: float, force_recompute: bool = True) -> None:
        start = time.perf_counter()
        self.ensure_initialized()
        self.sensor.update(dt=dt, force_recompute=force_recompute)
        output = self.sensor.data.output
        # A later vision-only render changes the underlying camera buffers.
        # Keep this capture independent of backend storage and future renders.
        self._last_rgb = _to_numpy_rgb(output["tactile_rgb"], self._resolution).copy()
        camera = getattr(self.sensor, "camera", None)
        camera_output = camera.data.output if camera is not None else {}
        if getattr(self, "optical_input", "deformed_surface") == "object" and "depth" in camera_output:
            import torch

            depth = camera_output["depth"][..., 0]
            near, far = self.sensor_cfg.camera_clipping_range
            self.sensor._robodojo_depth_valid = torch.isfinite(depth) & (depth > near) & (depth < far - 1e-6)
        self._valid = True
        self._last_update_ms = (time.perf_counter() - start) * 1000.0
        self._last_update_time = time.time()

    def setup_render_proxy(self, proxy_path: str) -> None:
        """Create a world-space USD surface used by the RTX tactile camera.

        TacEx/UIPC keeps the authoritative deformable state and writes its
        Fabric mesh with a world identity transform.  RoboDojo's GelSight
        asset is nested below an X5 finger, which can leave that Fabric mesh
        outside a newly-created RTX render product.  The proxy copies the
        same UIPC surface points/topology into a root-level USD mesh; it is
        refreshed once per physics/render generation.
        """

        if self._render_proxy_mesh is not None:
            return
        import numpy as np
        from omni.usd import get_context
        from pxr import Gf, UsdGeom, Vt

        stage = get_context().get_stage()
        root = stage.DefinePrim(proxy_path, "Xform")
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        xform.SetResetXformStack(True)
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        mesh = UsdGeom.Mesh.Define(stage, f"{proxy_path}/mesh")
        from uipc.geometry import extract_surface

        # USD/Fabric can retain the source visual topology (984 vertices),
        # while this asset's actual FEM surface has 169. Read topology and
        # positions together from the registered solver geometry.
        surface = extract_surface(self.gelpad.geo_slot_list[0].geometry())
        points = np.asarray(surface.positions().view()).reshape(-1, 3)
        triangles = np.asarray(surface.triangles().topo().view(), dtype=np.int64).reshape(-1, 3)
        initial = self.gelpad.init_world_transform.detach().cpu().numpy().reshape(4, 4)
        inverse = np.linalg.inv(initial)
        local = points @ inverse[:3, :3].T + inverse[:3, 3]
        # The camera looks through the gel's back. Render the deforming skin,
        # not the fixed attachment plate that would occlude every indentation.
        triangles = triangles[local[triangles, 2].min(axis=1) > 1e-5]
        if not len(triangles):
            raise RuntimeError("The calibrated gel asset has no visible front-surface triangles.")
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(triangles)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(triangles.reshape(-1).astype(np.int32).tolist()))
        mesh.CreateDisplayColorAttr([(0.6, 0.7, 0.8)])
        mesh.CreateDoubleSidedAttr(True)
        self._render_proxy_mesh = mesh
        self._render_proxy_path = proxy_path
        UsdGeom.Imageable(self.gelpad._usd_mesh_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        self.update_render_proxy()

    def configure_object_rendering(self) -> None:
        """Keep gel contact physics while the optical camera sees task geometry."""
        from pxr import UsdGeom

        mesh = UsdGeom.Imageable(self.gelpad._usd_mesh_prim)
        mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        proxy = getattr(self, "_render_proxy_mesh", None)
        if proxy is not None:
            proxy.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        # UIPC publishes points through Fabric. Mirror visibility there when
        # the mesh has already been populated, without changing solver data.
        fabric_prim = getattr(self.gelpad, "fabric_prim", None)
        if fabric_prim:
            import usdrt

            visibility = fabric_prim.GetAttribute("visibility")
            if not visibility:
                visibility = fabric_prim.CreateAttribute("visibility", usdrt.Sdf.ValueTypeNames.Token)
            visibility.Set("invisible")

    @contextmanager
    def show_visuals_in_vision(self):
        """Show the CAD gel in visual cameras, then restore optical visibility."""
        from omni.usd import get_context
        from pxr import UsdGeom
        import usdrt

        prims = [self.gelpad._usd_mesh_prim]
        fabric_stage = usdrt.Usd.Stage.Attach(get_context().get_stage_id())
        attributes = []
        for prim in prims:
            attributes.append(UsdGeom.Imageable(prim).GetVisibilityAttr())
            fabric_prim = fabric_stage.GetPrimAtPath(usdrt.Sdf.Path(str(prim.GetPath())))
            if fabric_prim:
                attribute = fabric_prim.GetAttribute("visibility")
                if attribute:
                    attributes.append(attribute)

        def restore(attribute, value, authored):
            clear = getattr(attribute, "Clear", None)
            if not authored and callable(clear):
                clear()
            else:
                attribute.Set(value)

        changed = False
        with ExitStack() as stack:
            for attribute in attributes:
                value = attribute.Get()
                if value == "inherited":
                    continue
                has_opinion = getattr(attribute, "HasAuthoredValueOpinion", None)
                authored = has_opinion() if callable(has_opinion) else True
                stack.callback(restore, attribute, value, authored)
                attribute.Set("inherited")
                changed = True
            yield changed

    def update_render_proxy(self) -> None:
        if self._render_proxy_mesh is None:
            return
        import numpy as np
        from pxr import Gf, Vt
        from uipc.geometry import extract_surface

        surface = extract_surface(self.gelpad.geo_slot_list[0].geometry())
        points = np.asarray(surface.positions().view()).reshape(-1, 3)
        self._render_proxy_mesh.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])
        )


    def observe(self) -> dict[str, Any]:
        return {"rgb": self._last_rgb, "valid": self._valid}



    def calibrate_nominal(self) -> None:
        """Only called after a reset settle with the robot clear of all props."""
        import torch

        self.attachment._compute_aim_positions()
        pose = self.attachment.obj_pose.reshape(-1, 7)[0]
        nodes = self.gelpad.data.nodal_pos_w.reshape(-1, 3)
        self.sensor.camera.update(dt=0.0, force_recompute=True)
        depth = self.sensor.camera.data.output["depth"][..., 0].clone()
        self._reference_depth = depth
        if self.optical_input == "object":
            # Object depth uses Taxim's fixed near + gel-thickness reference.
            # A clear camera can legitimately see no object at reset. Never
            # turn that empty frame into a per-pixel subtraction baseline.
            for attribute in ("_robodojo_nominal_depth", "_robodojo_nominal_mask", "_robodojo_depth_valid"):
                if hasattr(self.sensor, attribute):
                    delattr(self.sensor, attribute)
            return
        near, far = self.sensor_cfg.camera_clipping_range
        valid = torch.isfinite(depth) & (depth > near) & (depth < far - 1e-6)
        minimum_coverage = 0.2
        if float(valid.float().mean()) < minimum_coverage:
            raise RuntimeError(
                f"{self.sensor_cfg.name}: nominal depth coverage={float(valid.float().mean()):.3f}; range={float(depth.min())},{float(depth.max())}; mount={pose.detach().cpu().tolist()}; nodes={nodes.mean(0).detach().cpu().tolist()}; cam={self.sensor.camera.data.pos_w.detach().cpu().tolist()}"
            )
        self.sensor._robodojo_nominal_depth = depth
        self.sensor._robodojo_nominal_mask = valid

    def reference_depth(self):
        """Return the reset capture for recording identity, not policy inputs."""
        return self._reference_depth


    def invalidate(self) -> None:
        self._valid = False
        # Never expose an old episode/frame as if it were the current sensor
        # value after a failed render or backend update.  Callers can still
        # inspect ``valid=False`` and the manager diagnostics for the error.
        self._last_rgb = None

    def close(self) -> None:
        """Unsubscribe TacEx/Isaac callbacks before the stage is destroyed.

        TacEx 0.1 does not expose a public ``close`` method.  Its callback
        handles are intentionally private, so the adapter owns this small
        cleanup shim to make repeated in-process episodes deterministic.
        """

        self.invalidate()
        # TiledCamera owns Replicator annotators.  Let it detach those
        # annotators while the USD stage and Kit app are still alive; waiting
        # for Python's cyclic GC makes Isaac Sim 5.1 call ``__del__`` during
        # app shutdown, where its weak references can already be invalid.
        camera = getattr(self.sensor, "camera", None)
        if camera is not None:
            try:
                camera.__del__()
            except Exception:
                pass
            try:
                camera._annotators = {}
                camera._render_product_paths = []
            except Exception:
                pass
            try:
                self.sensor.camera = None
            except Exception:
                pass
        callback_objects = (self.sensor, self.gelpad, self.attachment)
        for obj in callback_objects:
            if obj is None:
                continue
            clear_callbacks = getattr(obj, "_clear_callbacks", None)
            if callable(clear_callbacks):
                try:
                    clear_callbacks()
                except Exception:
                    pass
            for handle_name in ("_initialize_handle", "_invalidate_initialize_handle", "_debug_vis_handle"):
                handle = getattr(obj, handle_name, None)
                if handle is not None:
                    try:
                        handle.unsubscribe()
                    except Exception:
                        pass
                    try:
                        setattr(obj, handle_name, None)
                    except Exception:
                        pass
        self._valid = False
        self._last_rgb = None
        self._last_update_time = None
        self._last_update_ms = 0.0
        self._nominal_state = None
        self._reference_depth = None
        proxy_path = getattr(self, "_render_proxy_path", None)
        if proxy_path:
            try:
                from omni.usd import get_context

                stage = get_context().get_stage()
                if stage is not None and stage.GetPrimAtPath(proxy_path).IsValid():
                    stage.RemovePrim(proxy_path)
            except Exception:
                pass
        self._render_proxy_mesh = None
        self._render_proxy_path = None

        # The manager has already erased the UIPC animation closures. Break
        # TacEx's optical and gelpad/constraint Python cycles without waiting
        # for cyclic GC. Keep the adapter's gelpad and attachment references
        # intact so a later manager cleanup retry can still traverse them.
        if self.sensor is not None:
            self.sensor.compute_indentation_depth_func = None
            self.sensor.optical_simulator = None
        if self.gelpad is not None:
            self.gelpad.constraint = None

    @property
    def last_update_ms(self) -> float:
        return self._last_update_ms

    @property
    def last_update_time(self) -> float | None:
        return self._last_update_time
