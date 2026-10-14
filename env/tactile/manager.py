"""Own optional tactile simulation over the native environment lifecycle."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
import importlib
import importlib.metadata
import os
import time
from typing import Any, Iterable

from .config import (
    TactileConfig,
    load_tactile_config,
    validate_calibration_dir,
    validate_tactile_config,
)
from .gelsight import _TacExGelSight
from .mount import _mount_sensor_assets
from .runtime import (
    TactileDependencyError as TactileDependencyError,
    _default_calibration_dir,
    _require_backend_modules,
    _resolve_robot_for_sensor,
    _set_manual_runtime_device,
    _validate_runtime_versions,
    _validate_sensor_prims,
)


class RoboDojoTactileManager:
    """Owns the two v1 GelSight sensors and their simulation lifecycle."""

    def __init__(self, config: TactileConfig | dict[str, Any] | None, *, workspace: str | None = None):
        if isinstance(config, TactileConfig):
            validate_tactile_config(config)
            self.config = config
        else:
            self.config = load_tactile_config(config)
        self.workspace = workspace or self.config.uipc_workspace
        self.enabled = self.config.enabled
        self.sensors: dict[str, _TacExGelSight] = {}
        self.uipc_rigid_objects: dict[str, Any] = {}
        self.uipc_sim: Any | None = None
        self.sim: Any | None = None
        self.robot_manager: Any | None = None
        self.initialized = False
        self._last_error: str | None = None
        self._physics_generation = 0
        self._render_generation = -1
        self._last_render_ms = 0.0
        self._render_sync_count = 0
        self._reset_count = 0
        self._capture_count = 0
        self._uipc_callback_names: set[str] = set()
        self._uipc_step_callback_registered = False
        # Scene reloads can execute internal physics steps. Keep the gel/UIPC
        # world frozen while RoboDojo resets the robot and task assets, then
        # explicitly resume it before settling the next episode.
        self._suspended = False
        self._mounted = False
        self._resolved_robots = []
        self._prepared_scene = None
        self._scene_bodies = {}
        self._robot_bodies = {}
        self._robot_render_bridges = []
        self._capture_generation = -1
        self._capture_sim_time = None
        self._step_timings = {"physx_s": 0.0, "uipc_s": 0.0, "state_sync_s": 0.0}
        self._references = {}
        self._vision_rendering = False
        self._optical_render_pending = False
        self._solver_profiler = None
        self._initialization_report = None
        self._initialization_report_path = None
        if self.enabled and self.config.extra.get("profile_solver", False):
            from .profiling import UipcStepProfiler

            self._solver_profiler = UipcStepProfiler(
                self.config.extra.get(
                    "profile_solver_output", os.path.join(self.workspace or "tactile_workspace", "profiles")
                ),
                steps=self.config.extra.get("profile_solver_steps", [300, 301, 600, 601]),
                fine_steps=self.config.extra.get("profile_solver_fine_steps", [301, 601]),
            )

    def mount(self, sim: Any, robot_manager: Any) -> None:
        """Mount surface cameras before the articulation starts playing."""
        if not self.enabled:
            return
        if self._mounted:
            return
        print("[RoboDojo tactile] initializing attachments", flush=True)
        if int(getattr(robot_manager, "num_envs", 1)) != 1:
            raise ValueError(
                "RoboDojo tactile v1 supports exactly one environment. "
                "Set sim.scene.num_envs=1 until batched UIPC outputs are implemented."
            )
        # Report the complete detected stack before any compatibility or USD
        # validation failure, so a launch log is actionable even when setup
        # stops before the first tactile render.
        self._print_versions()
        _validate_runtime_versions()
        _require_backend_modules()
        if self.config.left is None or self.config.right is None:
            raise ValueError("Enabled tactile mode requires left and right sensor configuration.")
        resolved_robots = []
        default_calibration_dir = _default_calibration_dir()
        for sensor in (self.config.left, self.config.right):
            print(f"[RoboDojo tactile] resolving {sensor.name} body={sensor.body_name}", flush=True)
            validate_calibration_dir(
                sensor.calibration_dir or default_calibration_dir,
                resolution=self.config.resolution,
            )
            print(f"[RoboDojo tactile] validated config {sensor.name}", flush=True)
            robot, robot_key = _resolve_robot_for_sensor(robot_manager, sensor)
            print(
                f"[RoboDojo tactile] resolved robot {sensor.name} root={getattr(getattr(robot_key, 'cfg', None), 'prim_path', '')}",
                flush=True,
            )
            _mount_sensor_assets(sensor, robot_key)
            from omni.usd import get_context
            from .mount import _single_env_path

            camera_mount = get_context().get_stage().GetPrimAtPath(_single_env_path(sensor.sensor_prim_path))
            sensor = replace(sensor, sensor_pose=tuple(camera_mount.GetAttribute("robodojo:mountPose").Get()))
            _validate_sensor_prims(sensor)
            print(f"[RoboDojo tactile] validated {sensor.name}", flush=True)
            resolved_robots.append((sensor, robot, robot_key))

        self._resolved_robots = resolved_robots
        self.sim = sim
        self.robot_manager = robot_manager
        self._mounted = True

    def initialize(self, sim: Any, robot_manager: Any, scene_manager: Any) -> None:
        """Build one complete episode, releasing partial ownership on failure."""
        if not self.enabled or self.initialized:
            return
        if scene_manager is None:
            raise ValueError("Tactile initialization requires the native SceneManager.")
        # A previous failed cleanup retains ownership rather than hiding live
        # callbacks. It must finish before a fresh world can be constructed.
        if self.uipc_sim is not None or self._prepared_scene is not None:
            self.close()
        self._initialization_report = None
        self._initialization_report_path = None
        try:
            self._initialize_episode(sim, robot_manager, scene_manager)
        except Exception as initialization_error:
            self.initialized = False
            try:
                self.close()
            except Exception as cleanup_error:
                raise ExceptionGroup(
                    "Tactile initialization and cleanup both failed",
                    [initialization_error, cleanup_error],
                ) from None
            raise
        # Observations and manual physics may use this world only after every
        # sensor, native follower and episode buffer has successfully reset.
        self.initialized = True

    def _initialize_episode(self, sim: Any, robot_manager: Any, scene_manager: Any) -> None:
        self.mount(sim, robot_manager)
        resolved_robots = self._resolved_robots
        default_calibration_dir = _default_calibration_dir()
        self._restore_gel_assets()
        from .dynamics import ArticulationRenderBridge
        from .scene import collect_robot_colliders, prepare_scene

        stage = scene_manager.stage
        robots = {id(robot): robot for robot in getattr(robot_manager, "robot_key", ())}
        self._robot_render_bridges = [ArticulationRenderBridge(robot, stage) for robot in robots.values()]
        excluded = [
            sensor.gelpad_prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
            for sensor, _, _ in self._resolved_robots
        ]
        robot_colliders = collect_robot_colliders(robot_manager, scene_manager.stage, exclude_prim_paths=excluded)

        self._prepared_scene = prepare_scene(
            scene_manager,
            cache_dir=self.config.extra.get("mesh_cache", ".cache/tactile/native_collision"),
            allow_surface_repair=bool(self.config.extra.get("allow_surface_repair", False)),
            max_surface_error=float(self.config.extra.get("max_surface_error", 0.0003)),
            extra_objects=robot_colliders,
            source_meshes=self.config.extra.get("source_meshes", {}),
        )

        print("[RoboDojo tactile] creating UipcSim", flush=True)
        from tacex_uipc import UipcSim, UipcSimCfg

        self.sim = sim
        self.robot_manager = robot_manager
        workspace = self.workspace or os.path.join(os.getcwd(), "tactile_workspace")
        os.makedirs(workspace, exist_ok=True)
        uipc_cfg = UipcSimCfg(
            device=self.config.device,
            dt=float(getattr(sim, "physics_dt", self.config.period)),
            strict_mode=True,
            update_render_meshes=False,
            workspace=workspace,
            collision_detection_method=self.config.collision_detection_method,
            newton=UipcSimCfg.Newton(
                velocity_tol=self.config.solver_velocity_tolerance,
                max_iter=self.config.solver_max_iterations,
            ),
            linear_system=UipcSimCfg.LinearSystem(tol_rate=self.config.solver_linear_tolerance),
            ground_height=float(self.config.extra.get("ground_height", 0.0)),
            contact=UipcSimCfg.Contact(
                d_hat=float(self.config.extra.get("d_hat", 0.001)),
                default_friction_ratio=float(self.config.extra.get("friction", 0.5)),
                eps_velocity=float(self.config.extra.get("eps_velocity", 0.01)),
            ),
        )
        self.uipc_sim = UipcSim(uipc_cfg)
        # TacEx enables every nested libuipc timer by default. Each timer
        # synchronizes CUDA and retains a record until explicitly reported.
        # Normal evaluation needs neither that overhead nor unbounded traces.
        from uipc import Timer

        Timer.disable_all()
        Timer.report_as_json()
        print("[RoboDojo tactile] UipcSim created", flush=True)
        # Task-owned rigid meshes must be registered before setup_sim so they
        # participate in the same IPC collision world as the gelpads.
        for spec in self._prepared_scene.specs:
            self._create_uipc_rigid_object(spec)
        if self._prepared_scene is not None:
            from .dynamics import UipcRobotBody

            for name, binding in self._prepared_scene.bindings.items():
                if binding.robot_key is not None:
                    self._robot_bodies[name] = UipcRobotBody(binding, self.uipc_rigid_objects[name], self.uipc_sim)
        self.sensors.clear()
        for sensor_cfg, _robot, robot_key in resolved_robots:
            print(f"[RoboDojo tactile] constructing GelSight {sensor_cfg.name}", flush=True)
            prim_path = sensor_cfg.rigid_body_prim_path or getattr(getattr(robot_key, "cfg", None), "prim_path", "")
            if not prim_path:
                raise ValueError(
                    f"Cannot determine Isaac rigid-body prim path for tactile body {sensor_cfg.body_name!r}."
                )
            # Own the configured callback before any sensor constructor can
            # fail. Partially built adapters also register their gel resources.
            self._uipc_callback_names.add(f"{sensor_cfg.gelpad_prim_path}_X_{prim_path}_attachment_update")
            sensor = _TacExGelSight(
                sensor_cfg=sensor_cfg,
                # ``robot_key`` is the live Isaac Lab Articulation from
                # RobotManager.robot_key; ``_robot`` is RoboDojo's static
                # embodiment description and cannot serve attachments.
                parent_robot=robot_key,
                rigid_body_prim_path=prim_path,
                uipc_sim=self.uipc_sim,
                resolution=self.config.resolution,
                update_period=self.config.period,
                optical_backend=self.config.optical_backend,
                optical_input=self.config.optical_input,
                device=self.config.device,
                calibration_dir=sensor_cfg.calibration_dir or default_calibration_dir,
                owner_registry=self.sensors,
            )
            self.sensors[sensor_cfg.name] = sensor
            print(f"[RoboDojo tactile] GelSight {sensor_cfg.name} constructed", flush=True)
            rigid_expr = sensor_cfg.rigid_body_prim_path or getattr(
                getattr(robot_key, "cfg", None), "prim_path", ""
            )
            if rigid_expr:
                self._uipc_callback_names.add(f"{sensor_cfg.gelpad_prim_path}_X_{rigid_expr}_attachment_update")
            # The attachment callback uses the concrete Isaac Lab
            # articulation expression from the live object. Keep that
            # exact name as well as the user-facing configured expression
            # so close() can remove callbacks after either path form.
            live_rigid_expr = getattr(
                getattr(getattr(sensor.attachment, "isaaclab_rigid_object", None), "cfg", None),
                "prim_path",
                "",
            )
            if live_rigid_expr:
                self._uipc_callback_names.add(
                    f"{sensor.gelpad.cfg.prim_path}_X_{live_rigid_expr}_attachment_update"
                )
        # TacEx/UIPC constructs all objects before ``setup_sim``. The
        # world then initializes against the complete scene graph, which
        # also keeps callback ordering consistent with UniVTAC's native
        # IsaacLab workflow.
        # ``update_render_meshes=False`` is deliberate: RoboDojo calls it
        # exactly once in before_render() after physics synchronization.
        # TacEx only registers ``uicp_step`` in this mode.
        self._uipc_callback_names.add("uicp_step")
        self._configure_contacts()
        from .initialization import commit_initialization, prepare_initialization, record_world_initialization

        proposal = prepare_initialization(self, scene_manager)
        commit_initialization(self, proposal)
        print("[RoboDojo tactile] setting up UIPC world", flush=True)
        self.uipc_sim.setup_sim()
        record_world_initialization(self, valid=self.uipc_sim.world.is_valid())
        if not self.uipc_sim.world.is_valid():
            raise RuntimeError(
                "UIPC rejected the initial scene. Check the preceding sanity-check "
                "errors for mesh overlap or ground penetration; sensor buffers "
                "cannot be initialized from an invalid world."
            )
        print("[RoboDojo tactile] UIPC world ready", flush=True)
        # TaskEnv may enter reset with tactile suspended. UipcSim installs
        # its global physics callback during setup_sim, so honor that state
        # immediately instead of advancing a previous episode while the
        # scene is being rebuilt.
        self._uipc_step_callback_registered = True
        # DirectRLEnv advances UIPC explicitly after refreshing robot
        # state. A global PhysX callback otherwise runs on stale buffers.
        self._pause_uipc_step_callback()
        # UipcSim registers global Isaac physics callbacks. Keep their
        # names so close() can unregister them before RoboDojo destroys
        # the stage; otherwise a second environment can call into a stale
        # UIPC world.
        self._initialize_handles()
        if self._prepared_scene is not None:
            from .dynamics import UipcSceneBody

            for name, binding in self._prepared_scene.bindings.items():
                if binding.scene_object.dynamic:
                    self._scene_bodies[name] = UipcSceneBody(
                        binding, self.uipc_rigid_objects[name], self.uipc_sim.world, self.config.device
                    )
        self.reset()

    def _restore_gel_assets(self) -> None:
        """Restore the CAD pad's undeformed geometry without moving rigid links."""
        from omni.usd import get_context
        from .surface import restore_surface_geometry

        stage = get_context().get_stage()
        for sensor, _, robot_key in self._resolved_robots:
            restore_surface_geometry(stage, sensor.gelpad_prim_path,
                                     robot_key.cfg.spawn.usd_path,
                                     sensor.rigid_body_prim_path or robot_key.cfg.prim_path)

    def begin_reset(self) -> None:
        """Dispose the previous contact world before SceneManager reloads USD."""
        if not self.enabled:
            return
        self.suspend()
        self._close_world()

    def _print_versions(self) -> None:
        versions = []
        distributions = {
            "isaacsim": ("isaacsim", "isaacsim"),
            "isaaclab": ("isaaclab", "isaaclab"),
            "tacex": ("tacex", "tacex"),
            "tacex_assets": ("tacex_assets", "tacex-assets"),
            "tacex_uipc": ("tacex_uipc", "tacex-uipc"),
            "libuipc": ("uipc", "pyuipc"),
            "torch_scatter": ("torch_scatter", "torch-scatter"),
        }
        for label, (module_name, distribution_name) in distributions.items():
            try:
                module = importlib.import_module(module_name)
                module_version = getattr(module, "__version__", None)
                package_version = importlib.metadata.version(distribution_name)
                versions.append(f"{label}={module_version or package_version}")
            except Exception as exc:  # pragma: no cover - Isaac-only path
                versions.append(f"{label}=unavailable({exc})")
        try:
            import torch

            versions.append(f"torch={torch.__version__}")
            versions.append(f"cuda={torch.version.cuda or 'none'}")
        except Exception as exc:  # pragma: no cover - Isaac-only path
            versions.append(f"torch=unavailable({exc})")
        print("[RoboDojo tactile] " + ", ".join(versions))

    def _create_uipc_rigid_object(self, spec: dict[str, Any]) -> None:
        """Register one task-owned rigid mesh in the TacEx/UIPC world."""

        from tacex_uipc import UipcRigidObject, UipcRigidObjectCfg

        name = str(spec.get("name", spec["prim_path"]))
        binding = self._prepared_scene.bindings.get(name) if self._prepared_scene is not None else None
        driven = binding is not None and binding.robot_key is not None
        cfg = UipcRigidObjectCfg(
            prim_path=str(spec["prim_path"]),
            mesh_cfg=spec.get("mesh_cfg"),
            mass_density=float(spec.get("mass_density", 1e3)),
            usd_mesh_prim_name=spec.get("usd_mesh_prim_name"),
            constitution_cfg=UipcRigidObjectCfg.AffineBodyConstitutionCfg(
                kinematic=bool(spec.get("kinematic", True)) and not driven
            ),
        )
        if driven:
            from .dynamics import create_driven_body

            self.uipc_rigid_objects[name] = create_driven_body(cfg, self.uipc_sim)
        else:
            self.uipc_rigid_objects[name] = UipcRigidObject(cfg, self.uipc_sim)

    def _configure_contacts(self) -> None:
        """Map material friction and exclude contacts already solved by PhysX."""
        if self._prepared_scene is None:
            return
        tabular = self.uipc_sim.scene.contact_tabular()
        resistance = float(self.uipc_sim.cfg.contact.default_contact_resistance) * 1e9
        elements = []
        for name, binding in self._prepared_scene.bindings.items():
            descriptor = binding.scene_object
            kind = "robot" if binding.robot_key is not None else ("dynamic" if descriptor.dynamic else "static")
            physics = getattr(descriptor.wrapper, "physics_config", {})
            default = 0.8 if descriptor.object_type == "table" else (1.5 if descriptor.dynamic else 0.0)
            friction = float(physics.get("dynamic_friction", default))
            element = tabular.create(name)
            element.apply_to(self.uipc_rigid_objects[name].geo_slot_list[0].geometry())
            elements.append((element, kind, friction))
        for name, sensor in self.sensors.items():
            element = tabular.create(f"gel:{name}")
            element.apply_to(sensor.gelpad.geo_slot_list[0].geometry())
            elements.append((element, "gel", float(self.config.extra.get("friction", 0.5))))
        for index, (left, kind_left, friction_left) in enumerate(elements):
            for right, kind_right, friction_right in elements[index:]:
                kinds = {kind_left, kind_right}
                # Robot/static and robot self contacts remain PhysX's job.
                # Gel attachments already constrain pads to their housings.
                enabled = "robot" not in kinds or "dynamic" in kinds
                if kinds == {"static"}:
                    enabled = False
                tabular.insert(left, right, (friction_left + friction_right) / 2, resistance, enable=enabled)

    def _ensure_uipc_rigid_initialized(self) -> None:
        """Initialize task-owned UIPC rigid buffers after the PLAY event."""

        for obj in self.uipc_rigid_objects.values():
            if getattr(obj, "_is_initialized", False):
                continue
            initialize_impl = getattr(obj, "_initialize_impl", None)
            if not callable(initialize_impl):
                raise RuntimeError("TacEx UIPC rigid object does not expose an initialization hook.")
            _set_manual_runtime_device(obj, str(self.uipc_sim.cfg.device))
            initialize_impl()
            obj._is_initialized = True

    def reset(self) -> None:
        if not self.enabled:
            return
        self._physics_generation = 0
        self._render_generation = -1
        self._last_render_ms = 0.0
        self._last_error = None
        self._reset_count += 1
        self._capture_count = 0
        self._capture_generation = -1
        self._capture_sim_time = None
        self._step_timings = {"physx_s": 0.0, "uipc_s": 0.0, "state_sync_s": 0.0}
        self._references.clear()
        for sensor in self.sensors.values():
            sensor.reset()

    @property
    def suspended(self) -> bool:
        """Whether tactile physics/render updates are temporarily paused."""

        return self._suspended

    def suspend(self) -> None:
        """Pause tactile updates during a RoboDojo scene/robot reset."""

        self._suspended = True
        self._pause_uipc_step_callback()

    def resume(self) -> None:
        """Resume tactile updates after reset and scene stabilization."""

        self._suspended = False

    def _pause_uipc_step_callback(self) -> None:
        """Stop UIPC's world.advance callback while reset owns the timeline."""

        if not self._uipc_step_callback_registered:
            return
        isaac_sim = getattr(self.uipc_sim, "isaac_sim", None)
        exists = getattr(isaac_sim, "physics_callback_exists", None)
        if not callable(exists):
            raise RuntimeError("Cannot verify removal of UIPC automatic stepping: physics_callback_exists is unavailable")
        if not exists("uicp_step"):
            self._uipc_step_callback_registered = False
            return
        remove_callback = getattr(isaac_sim, "remove_physics_callback", None)
        if not callable(remove_callback):
            raise RuntimeError("Cannot stop UIPC automatic stepping: remove_physics_callback is unavailable")
        # Manual stepping is safe only after confirmed removal. Keep the
        # registered flag on every failure, including a no-op backend removal.
        remove_callback("uicp_step")
        if exists("uicp_step"):
            raise RuntimeError("UIPC automatic step callback remains registered after removal")
        self._uipc_step_callback_registered = False

    def ensure_initialized(self) -> None:
        """Ensure all TacEx handles exist before the next physics step."""

        if not self.enabled or not self.initialized:
            return
        self._initialize_handles()

    def _initialize_handles(self) -> None:
        """Initialize constructed handles before publishing the ready flag."""
        self._ensure_uipc_rigid_initialized()
        for sensor in self.sensors.values():
            sensor.ensure_initialized()
        self._ensure_optical_geometry()

    def _ensure_optical_geometry(self) -> None:
        """Select the optical geometry after UIPC has authored its surface."""

        if not self.enabled or self.uipc_sim is None or self._vision_rendering:
            return
        for sensor_cfg in (self.config.left, self.config.right):
            if sensor_cfg is None:
                continue
            sensor = self.sensors.get(sensor_cfg.name)
            if sensor is None:
                continue
            if self.config.optical_input == "object":
                sensor.configure_object_rendering()
                continue
            if getattr(sensor, "_render_proxy_mesh", None) is not None:
                continue
            proxy_path = f"/World/RoboDojoTactileRenderProxy/{sensor_cfg.name}"
            sensor.setup_render_proxy(proxy_path)


    def calibrate_reference(self, render) -> None:
        """Capture a reference only after proving separation from task objects."""
        if not self.enabled:
            return
        import hashlib

        import numpy as np

        if self._prepared_scene is None:
            raise RuntimeError("Reference calibration requires a registered native scene.")
        bounds = []
        for name, binding in self._prepared_scene.bindings.items():
            if binding.robot_key is not None:
                continue
            body = self.uipc_rigid_objects[name]
            mesh = body.geo_slot_list[0].geometry()
            vertices = np.asarray(mesh.positions().view()).reshape(-1, 3)
            transform = np.asarray(mesh.transforms().view()).reshape(-1, 4, 4)[0]
            vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
            bounds.append((name, vertices.min(axis=0), vertices.max(axis=0)))
        for name, sensor in self.sensors.items():
            nodes = sensor.gelpad.data.nodal_pos_w.detach().cpu().numpy().reshape(-1, 3)
            low, high = nodes.min(axis=0), nodes.max(axis=0)
            for object_name, object_low, object_high in bounds:
                gap = np.maximum(np.maximum(low - object_high, object_low - high), 0)
                if float(np.linalg.norm(gap)) < 0.005:
                    raise RuntimeError(
                        f"Cannot calibrate {name}: its bounds are within 5 mm of {object_name}. "
                        "A reset frame in possible contact is not a valid reference."
                    )
        self.render_for_tactile(render)
        for sensor in self.sensors.values():
            sensor.calibrate_nominal()
        self._capture_generation = -1
        self.capture()
        for name, sensor in self.sensors.items():
            rgb = sensor.observe()["rgb"].copy()
            depth = sensor.reference_depth().detach().cpu().numpy()
            digest = hashlib.sha256(rgb.tobytes() + depth.tobytes()).hexdigest()
            self._references[name] = {"reference_rgb": rgb, "reference_id": digest}

    def physics_step(self, dt: float, *, physx_elapsed_s: float = 0.0) -> None:
        if not self.enabled or not self.initialized or self._suspended:
            return
        started = time.perf_counter()
        self._ensure_uipc_rigid_initialized()
        link_transforms = {}
        for body in self._robot_bodies.values():
            key = id(body.robot)
            if key not in link_transforms:
                link_transforms[key] = body.robot.root_physx_view.get_link_transforms().detach().cpu().numpy()
            body.set_target(link_transforms[key])
        for sensor in self.sensors.values():
            sensor.attachment._compute_aim_positions()
        prepared = time.perf_counter()
        try:
            if self._solver_profiler is None:
                self.uipc_sim.step(dt)
            else:
                self._solver_profiler.step(
                    self.uipc_sim, dt, physics_generation=self._physics_generation, reset_count=self._reset_count
                )
            if not self.uipc_sim.world.is_valid():
                raise RuntimeError("Tactile contact solver failed; close and rebuild the environment.")
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._suspended = True
            for sensor in self.sensors.values():
                sensor.invalidate()
            raise
        solved = time.perf_counter()
        for obj in self.uipc_rigid_objects.values():
            obj.update(dt=dt)
        for body in self._scene_bodies.values():
            body.sync_visual()
        for sensor in self.sensors.values():
            sensor.physics_step(dt)
        self._physics_generation += 1
        self._step_timings["physx_s"] += physx_elapsed_s
        self._step_timings["uipc_s"] += solved - prepared
        self._step_timings["state_sync_s"] += prepared - started + time.perf_counter() - solved

    def before_render(self) -> None:
        if (
            self.enabled
            and not self._suspended
            and self.uipc_sim is not None
            and self._render_generation != self._physics_generation
        ):
            self._ensure_optical_geometry()
            for bridge in self._robot_render_bridges:
                bridge.sync()
            for sensor in self.sensors.values():
                sensor.sync_mount_pose()
            start = time.perf_counter()
            self.uipc_sim.update_render_meshes()
            for sensor in self.sensors.values():
                sensor.update_render_proxy()
            self._render_generation = self._physics_generation
            self._render_sync_count += 1
            self._last_render_ms = (time.perf_counter() - start) * 1000.0

    @contextmanager
    def vision_render_pass(self):
        """Select sensor visibility for an ordinary camera pass.

        Object-depth optics hide the gel and housing; the vision pass can show
        both when requested. Surface-mode proxies retain the existing setting.
        Yield whether another render is needed, restoring opinions on failure.
        """
        if not self.enabled or not self.initialized or self._suspended:
            yield False
            return
        if self.config.optical_input == "object" and self.config.render_in_vision:
            if self._vision_rendering:
                raise RuntimeError("An ordinary sensor-visibility pass is already active.")
            self._vision_rendering = True
            changed = False
            try:
                with ExitStack() as stack:
                    for sensor in self.sensors.values():
                        changed = stack.enter_context(sensor.show_visuals_in_vision()) or changed
                    yield changed
            finally:
                self._vision_rendering = False
                # A vision pass overwrites the same RTX buffers used by the
                # optical cameras. The hidden geometry needs a render fence
                # before any later tactile capture can consume those buffers.
                self._optical_render_pending = self._optical_render_pending or changed
            return
        if self.config.render_in_vision:
            yield False
            return
        previous = []
        try:
            for sensor in self.sensors.values():
                mesh = getattr(sensor, "_render_proxy_mesh", None)
                if mesh is None:
                    continue
                visibility = mesh.GetVisibilityAttr()
                value = visibility.Get()
                if value == "invisible":
                    continue
                previous.append((visibility, visibility.HasAuthoredValueOpinion(), value))
                visibility.Set("invisible")
            yield bool(previous)
        finally:
            for visibility, authored, value in reversed(previous):
                if authored:
                    visibility.Set(value)
                else:
                    visibility.Clear()

    def render_for_tactile(self, render) -> None:
        """Fence hidden optical geometry without advancing physics or capture."""
        if self._vision_rendering:
            raise RuntimeError("Cannot render tactile depth during an ordinary camera pass.")
        self._ensure_optical_geometry()
        self.before_render()
        render()
        self._optical_render_pending = False

    def capture(self, dt: float | None = None) -> None:
        if not self.enabled or not self.initialized or self._suspended:
            return
        if self._vision_rendering:
            raise RuntimeError("Cannot capture tactile data during an ordinary camera pass.")
        if self._capture_generation == self._physics_generation:
            return
        dt = float(dt or self.config.period)
        try:
            if self._optical_render_pending:
                from env.camera_manager.capture.render_sync import wait_for_latest_cameras

                self.render_for_tactile(lambda: wait_for_latest_cameras(self.sim))
            for sensor in self.sensors.values():
                sensor.capture(dt=dt, force_recompute=True)
            self._capture_count += 1
            self._capture_generation = self._physics_generation
            step = int(getattr(self.sim, "_sim_step_counter", self._physics_generation))
            self._capture_sim_time = step * float(getattr(self.sim, "physics_dt", self.config.period))
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            for sensor in self.sensors.values():
                sensor.invalidate()
            raise

    def observe(self, env_ids: Iterable[int] | None = None) -> dict[int, dict[str, dict[str, Any]]]:
        if not self.enabled:
            return {}
        if env_ids is None:
            env_ids = range(getattr(self.sim, "num_envs", 1))
        single = {
            name: {
                **sensor.observe(),
                **self._references.get(name, {}),
                "sensor_id": name,
                "frame_id": self._capture_count - 1,
                "capture_sim_time": self._capture_sim_time,
                "age_steps": max(0, self._physics_generation - self._capture_generation),
                "calibration_id": sensor.sensor_cfg.calibration_id,
            }
            for name, sensor in self.sensors.items()
        }
        return {int(env_id): {name: dict(value) for name, value in single.items()} for env_id in env_ids}

    def diagnostics(self) -> dict[str, Any]:
        """Report inexpensive lifecycle status and timing without solver snapshots."""
        return {
            "enabled": self.enabled,
            "initialized": self.initialized,
            "suspended": self._suspended,
            "resolution": list(self.config.resolution),
            "sensor_names": sorted(self.sensors),
            "sensors": {
                name: {"valid": sensor.observe()["valid"],
                       "last_update_ms": sensor.last_update_ms,
                       "last_update_time": sensor.last_update_time}
                for name, sensor in self.sensors.items()
            },
            "physics_steps": self._physics_generation,
            "reset_count": self._reset_count,
            "capture_count": self._capture_count,
            "last_render_sync_ms": self._last_render_ms,
            "timing": dict(self._step_timings),
            "last_error": self._last_error,
        }


    def _remove_owned_physics_callbacks(self) -> None:
        """Confirm callback removal before releasing anything they can reference."""
        isaac_sim = getattr(self.uipc_sim, "isaac_sim", None)
        # setup_sim installs uicp_step before initialize marks registration.
        # With no isaac_sim and no registration flag, setup did not reach the
        # native callback installation; the configured names need not exist.
        if isaac_sim is not None:
            self._uipc_step_callback_registered = True
        errors = []
        try:
            self._pause_uipc_step_callback()
        except Exception as error:
            errors.append(error)
        if isaac_sim is not None:
            exists = getattr(isaac_sim, "physics_callback_exists", None)
            remove = getattr(isaac_sim, "remove_physics_callback", None)
            for name in sorted(self._uipc_callback_names - {"uicp_step"}):
                try:
                    if not callable(exists):
                        raise RuntimeError(
                            f"Cannot verify removal of tactile callback {name!r}: "
                            "physics_callback_exists is unavailable"
                        )
                    if not exists(name):
                        continue
                    if not callable(remove):
                        raise RuntimeError(
                            f"Cannot remove tactile callback {name!r}: "
                            "remove_physics_callback is unavailable"
                        )
                    remove(name)
                    if exists(name):
                        raise RuntimeError(f"Tactile callback {name!r} remains registered after removal")
                except Exception as error:
                    # Try the remaining independent removals, but retain world
                    # ownership and every resource while any removal is uncertain.
                    errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Tactile physics callback cleanup failed", errors)

    def _close_world(self) -> None:
        self._remove_owned_physics_callbacks()
        if self.uipc_sim is not None:
            # C++ animation callbacks hold Python closures. Erase them before
            # releasing objects to break the scene -> callback -> scene cycle.
            animator = self.uipc_sim.scene.animator()
            actors = list(self.uipc_rigid_objects.values())
            actors.extend(sensor.gelpad for sensor in self.sensors.values() if sensor.gelpad is not None)
            for actor in actors:
                for scene_object in actor.uipc_scene_objects:
                    animator.erase(scene_object.id())
        for body in self._scene_bodies.values():
            body.close()
        self._scene_bodies.clear()
        self._robot_bodies.clear()
        self._robot_render_bridges.clear()
        for sensor in self.sensors.values():
            sensor.close()
        if self.uipc_sim is not None:
            # The UIPC world keeps strong references to all objects. Clear
            # those registries before releasing the world so a stage restart
            # cannot retain a stale gelpad/attachment graph.
            for attr in ("uipc_objects", "_fabric_meshes"):
                value = getattr(self.uipc_sim, attr, None)
                if hasattr(value, "clear"):
                    value.clear()
            offsets = getattr(self.uipc_sim, "_surf_vertex_offsets", None)
            if isinstance(offsets, list):
                offsets[:] = [0]
        for obj in self.uipc_rigid_objects.values():
            clear_callbacks = getattr(obj, "_clear_callbacks", None)
            if callable(clear_callbacks):
                clear_callbacks()
        if self._prepared_scene is not None:
            stage = self._prepared_scene.bindings[next(iter(self._prepared_scene.bindings))].scene_object.wrapper.stage
            if stage.GetPrimAtPath(self._prepared_scene.root_path).IsValid():
                stage.RemovePrim(self._prepared_scene.root_path)
            self._prepared_scene = None
        self.sensors.clear()
        self.uipc_rigid_objects.clear()
        self.uipc_sim = None
        self.initialized = False
        self._physics_generation = 0
        self._render_generation = -1
        self._last_render_ms = 0.0
        self._render_sync_count = 0
        self._capture_count = 0
        self._last_error = None
        self._uipc_callback_names.clear()
        self._uipc_step_callback_registered = False
        self._capture_generation = -1
        self._capture_sim_time = None
        self._references.clear()
        self._vision_rendering = False
        self._optical_render_pending = False

    def close(self) -> None:
        self._close_world()
        self.sim = None
        self.robot_manager = None
        self._resolved_robots.clear()
        self._mounted = False
        self._suspended = False
