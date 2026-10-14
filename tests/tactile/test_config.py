from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - depends on the host runtime
    np = None

from env.tactile.config import (
    TactileConfigError,
    load_tactile_config,
    validate_calibration_dir,
    validate_tactile_config,
)
from env.tactile.manager import RoboDojoTactileManager
from env.tactile.runtime import (
    TactileDependencyError,
    _require_backend_modules,
    _resolve_robot_for_sensor,
    _set_manual_runtime_device,
    _to_numpy_rgb,
)


class TactileConfigTests(unittest.TestCase):
    def test_attachment_target_dtype_defaults_and_routes_per_sensor(self):
        from utils.load_file import load_yaml

        raw = load_yaml(Path(__file__).resolve().parents[2] / "env_cfg/arx_x5_pad_tactile.yml")["tactile"]
        config = load_tactile_config(raw)
        self.assertEqual((config.left.attachment_target_dtype, config.right.attachment_target_dtype),
                         ("float32", "float32"))
        raw["left"]["attachment_target_dtype"] = "float64"
        config = load_tactile_config(raw)
        self.assertEqual((config.left.attachment_target_dtype, config.right.attachment_target_dtype),
                         ("float64", "float32"))

    def test_attachment_target_dtype_rejects_invalid_enabled_disabled_and_dataclass_values(self):
        from utils.load_file import load_yaml

        raw = load_yaml(Path(__file__).resolve().parents[2] / "env_cfg/arx_x5_pad_tactile.yml")["tactile"]
        config = load_tactile_config(raw)
        for value in (None, True, 64, "Float64", "float16"):
            with self.subTest(value=value):
                for enabled in (True, False):
                    data = deepcopy(raw)
                    data["enabled"] = enabled
                    data["left"]["attachment_target_dtype"] = value
                    with self.assertRaisesRegex(TactileConfigError, "left.attachment_target_dtype"):
                        load_tactile_config(data)
                with self.assertRaisesRegex(TactileConfigError, "left.attachment_target_dtype"):
                    validate_tactile_config(replace(config, left=replace(config.left, attachment_target_dtype=value)))

    def test_solver_max_iterations_preserves_default_and_accepts_int32_range(self):
        from omegaconf import OmegaConf

        self.assertEqual(load_tactile_config(None).solver_max_iterations, 1024)
        for value in (1, 1024, 2048, 2**31 - 1):
            with self.subTest(value=value):
                config = load_tactile_config(OmegaConf.create({"solver_max_iterations": value}))
                self.assertEqual(config.solver_max_iterations, value)
                self.assertNotIn("solver_max_iterations", config.extra)
                validate_tactile_config(config)

    def test_solver_max_iterations_presets_keep_legacy_default(self):
        from utils.load_file import load_yaml

        root = Path(__file__).resolve().parents[2]
        for name in ("arx_x5_pad_tactile",):
            with self.subTest(preset=name):
                raw = load_yaml(root / "env_cfg" / f"{name}.yml")["tactile"]
                self.assertNotIn("solver_max_iterations", raw)
                self.assertEqual(load_tactile_config(raw).solver_max_iterations, 1024)

    def test_solver_max_iterations_rejects_invalid_mapping_dataclass_and_manager(self):
        from utils.load_file import load_yaml

        raw = load_yaml(Path(__file__).resolve().parents[2] / "env_cfg/arx_x5_pad_tactile.yml")["tactile"]
        enabled = load_tactile_config(raw)
        self.assertTrue(enabled.enabled)
        self.assertIs(RoboDojoTactileManager(enabled).config, enabled)
        backend = ModuleType("tacex_uipc")
        backend.UipcSimCfg = Mock()
        backend.UipcSim = Mock()
        invalid = (0, -1, True, False, 1024.0, "2048", None, float("nan"), float("inf"), 2**31, 2**63)
        for value in invalid:
            for base in (load_tactile_config(None), enabled):
                with self.subTest(value=value, enabled=base.enabled):
                    data = {"enabled": base.enabled, "solver_max_iterations": value}
                    with self.assertRaisesRegex(TactileConfigError, "solver_max_iterations"):
                        load_tactile_config(data)
                    direct = replace(base, solver_max_iterations=value)
                    with self.assertRaisesRegex(TactileConfigError, "solver_max_iterations"):
                        validate_tactile_config(direct)
                    with patch.dict(sys.modules, {"tacex_uipc": backend}), self.assertRaisesRegex(
                        TactileConfigError, "solver_max_iterations"
                    ):
                        RoboDojoTactileManager(direct)
                    backend.UipcSimCfg.assert_not_called()
                    backend.UipcSim.assert_not_called()

    def test_solver_linear_tolerance_preserves_default_and_accepts_range(self):
        self.assertEqual(load_tactile_config(None).solver_linear_tolerance, 0.001)
        for value in (0.000001, 0.1, 1, "0.000001"):
            with self.subTest(value=value):
                config = load_tactile_config({"solver_linear_tolerance": value})
                self.assertEqual(config.solver_linear_tolerance, float(value))
                self.assertNotIn("solver_linear_tolerance", config.extra)

    def test_solver_linear_tolerance_rejects_invalid_mapping_and_dataclass(self):
        for value in (0, -1, 1.000001, float("nan"), float("inf"), -float("inf"), None, "invalid", True, False):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TactileConfigError, "solver_linear_tolerance"):
                    load_tactile_config({"solver_linear_tolerance": value})
                with self.assertRaisesRegex(TactileConfigError, "solver_linear_tolerance"):
                    validate_tactile_config(replace(load_tactile_config(None), solver_linear_tolerance=value))

    def test_presets_share_original_solver_defaults(self):
        from utils.load_file import load_yaml

        root = Path(__file__).resolve().parents[2]
        for preset in ("arx_x5_pad_tactile",):
            with self.subTest(preset=preset):
                raw = load_yaml(root / "env_cfg" / f"{preset}.yml")["tactile"]
                config = load_tactile_config(raw)
                self.assertEqual(config.solver_velocity_tolerance, 0.05)
                self.assertEqual(config.solver_linear_tolerance, 0.001)
                self.assertEqual(config.solver_max_iterations, 1024)

    def test_solver_velocity_tolerance_preserves_default_and_accepts_override(self):
        self.assertEqual(load_tactile_config(None).solver_velocity_tolerance, 0.05)
        config = load_tactile_config({"solver_velocity_tolerance": 0.0005})
        self.assertEqual(config.solver_velocity_tolerance, 0.0005)
        self.assertNotIn("solver_velocity_tolerance", config.extra)

    def test_solver_velocity_tolerance_rejects_invalid_values(self):
        for value in (0, -1, float("nan"), float("inf"), -float("inf"), None, "invalid"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TactileConfigError, "solver_velocity_tolerance.*positive and finite"):
                    load_tactile_config({"solver_velocity_tolerance": value})

    def test_manager_passes_solver_tolerances_to_backend_construction(self):
        class StopBeforeEngine(Exception):
            pass

        class BackendConfig(SimpleNamespace):
            Newton = SimpleNamespace
            LinearSystem = SimpleNamespace
            Contact = SimpleNamespace

        backend = ModuleType("tacex_uipc")
        backend.UipcSimCfg = BackendConfig
        backend.UipcSim = Mock(side_effect=StopBeforeEngine)
        cases = (
            ({}, 0.05, 0.001, 1024),
            ({"solver_velocity_tolerance": 0.0005}, 0.0005, 0.001, 1024),
            ({"solver_linear_tolerance": 0.000001}, 0.05, 0.000001, 1024),
            ({"solver_max_iterations": 2048}, 0.05, 0.001, 2048),
            (replace(load_tactile_config(None), solver_max_iterations=2048), 0.05, 0.001, 2048),
        )
        for raw, expected, linear_expected, iteration_expected in cases:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as workspace:
                manager = RoboDojoTactileManager(raw, workspace=workspace)
                simulation = SimpleNamespace(physics_dt=0.004)
                robots = SimpleNamespace(robot_key=[])
                scene = SimpleNamespace(stage=object())
                with (
                    patch.object(manager, "mount"),
                    patch.object(manager, "_restore_gel_assets"),
                    patch("env.tactile.manager._default_calibration_dir", return_value=None),
                    patch("env.tactile.scene.collect_robot_colliders", return_value=[]),
                    patch("env.tactile.scene.prepare_scene"),
                    patch.dict(sys.modules, {"tacex_uipc": backend}),
                    self.assertRaises(StopBeforeEngine),
                ):
                    manager._initialize_episode(simulation, robots, scene)
                config = backend.UipcSim.call_args.args[0]
                self.assertEqual(config.newton.velocity_tol, expected)
                self.assertEqual(config.linear_system.tol_rate, linear_expected)
                self.assertEqual(config.newton.max_iter, iteration_expected)
                self.assertTrue(config.strict_mode)
                self.assertEqual(config.dt, simulation.physics_dt)

    def test_surface_preset_uses_source_pads_and_explicit_physical_field(self):
        from utils.load_file import load_yaml

        preset = load_yaml(Path(__file__).resolve().parents[2] / "env_cfg/arx_x5_pad_tactile.yml")
        config = load_tactile_config(preset["tactile"])
        self.assertEqual(config.optical_input, "object")
        self.assertEqual(config.resolution, (240, 320))
        for sensor, body in ((config.left, "link7"), (config.right, "link8")):
            self.assertEqual(sensor.mounting, "surface")
            self.assertEqual(sensor.body_name, body)
            self.assertEqual(sensor.optical_field_mm, (20.4, 15.3))
            self.assertAlmostEqual(sensor.camera_clipping_range[0] + 0.001, sensor.camera_focus_distance)
            self.assertEqual(sensor.attachment_strength_ratio, 1e5)
            self.assertIsNone(sensor.attachment_stiffness_multiplier)
            self.assertEqual(sensor.attachment_target_dtype, "float32")
            self.assertEqual(sensor.rigid_body_prim_path, "/World/envs/env_.*/robot0")
            self.assertEqual(sensor.sensor_pose, (0, 0, 0, 1, 0, 0, 0))

    def test_surface_requires_field_and_supports_both_optical_inputs(self):
        from utils.load_file import load_yaml

        raw = load_yaml(Path(__file__).resolve().parents[2] / "env_cfg/arx_x5_pad_tactile.yml")["tactile"]
        missing = deepcopy(raw)
        del missing["left"]["optical_field_mm"]
        with self.assertRaisesRegex(TactileConfigError, "requires optical_field_mm"):
            load_tactile_config(missing)
        for mode in ("object", "deformed_surface"):
            with self.subTest(mode=mode):
                raw["optical_input"] = mode
                self.assertEqual(load_tactile_config(raw).optical_input, mode)

    def test_surface_field_rejects_invalid_dimensions_and_image_aspect(self):
        from utils.load_file import load_yaml

        raw = load_yaml(Path(__file__).resolve().parents[2] / "env_cfg/arx_x5_pad_tactile.yml")["tactile"]
        for value in ([0, 15.3], [-20.4, 15.3], [float("nan"), 15.3], [20.4, float("inf")], [20.4]):
            with self.subTest(field=value):
                data = deepcopy(raw)
                data["left"]["optical_field_mm"] = value
                with self.assertRaisesRegex(TactileConfigError, "positive"):
                    load_tactile_config(data)
        data = deepcopy(raw)
        data["left"]["optical_field_mm"] = [20.4, 10.2]
        with self.assertRaisesRegex(TactileConfigError, "image aspect ratio"):
            load_tactile_config(data)


    def test_attachment_strength_ratio_is_per_sensor_with_compatible_default(self):
        from utils.load_file import load_yaml

        root = Path(__file__).resolve().parents[2]
        raw = load_yaml(root / "env_cfg/arx_x5_pad_tactile.yml")["tactile"]
        raw["left"]["attachment_strength_ratio"] = 1e7
        config = load_tactile_config(raw)
        self.assertEqual(config.left.attachment_strength_ratio, 1e7)
        self.assertEqual(config.right.attachment_strength_ratio, 1e5)

    def test_attachment_strength_ratio_rejects_nonpositive_and_nonfinite_values(self):
        from utils.load_file import load_yaml

        root = Path(__file__).resolve().parents[2]
        raw = load_yaml(root / "env_cfg/arx_x5_pad_tactile.yml")["tactile"]
        for value in (0, -1, float("nan"), float("inf"), -float("inf"), None, "invalid"):
            with self.subTest(value=value):
                data = deepcopy(raw)
                data["left"]["attachment_strength_ratio"] = value
                with self.assertRaisesRegex(TactileConfigError, "left.attachment_strength_ratio"):
                    load_tactile_config(data)


    def test_surface_mount_rejects_pose_overrides(self):
        from utils.load_file import load_yaml

        preset = load_yaml(Path(__file__).resolve().parents[2] / "env_cfg/arx_x5_pad_tactile.yml")
        for field in ("pose", "gelpad_pose", "sensor_pose", "camera_pose"):
            for value in (None, [0, 0, 0, 1, 0, 0, 0]):
                with self.subTest(field=field, value=value):
                    data = dict(preset["tactile"])
                    data["left"] = dict(data["left"], **{field: value})
                    with self.assertRaisesRegex(TactileConfigError, "omit pose overrides"):
                        load_tactile_config(data)

    def test_removed_mounting_modes_are_rejected(self):
        from utils.load_file import load_yaml

        raw = load_yaml(Path(__file__).resolve().parents[2] / "env_cfg/arx_x5_pad_tactile.yml")["tactile"]
        for mounting in ("reference", "existing"):
            with self.subTest(mounting=mounting):
                changed = deepcopy(raw)
                changed["left"]["mounting"] = mounting
                with self.assertRaisesRegex(TactileConfigError, "must be 'surface'"):
                    load_tactile_config(changed)
                config = load_tactile_config(raw)
                with self.assertRaisesRegex(TactileConfigError, "must be 'surface'"):
                    validate_tactile_config(replace(config, left=replace(config.left, mounting=mounting)))

    def test_initial_pose_correction_defaults_strict_and_validates_budget(self):
        self.assertEqual(load_tactile_config({}).max_initial_pose_correction_m, 0.0)
        cfg = load_tactile_config({"max_initial_pose_correction_m": .0003})
        self.assertEqual(cfg.max_initial_pose_correction_m, .0003)
        self.assertNotIn("max_initial_pose_correction_m", cfg.extra)
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(TactileConfigError, "max_initial_pose_correction_m"):
                load_tactile_config({"max_initial_pose_correction_m": value})

    def test_collision_detection_method_default_and_opt_in(self):
        self.assertEqual(load_tactile_config({}).collision_detection_method, "linear_bvh")
        cfg = load_tactile_config({"collision_detection_method": "stackless_bvh"})
        self.assertEqual(cfg.collision_detection_method, "stackless_bvh")
        self.assertNotIn("collision_detection_method", cfg.extra)

    def test_collision_detection_method_rejects_unknown_values(self):
        for value in ("unknown", "informative_linear_bvh", None, True):
            with self.subTest(value=value), self.assertRaisesRegex(TactileConfigError, "collision_detection_method"):
                load_tactile_config({"collision_detection_method": value})


    def test_duplicate_body_names_use_explicit_articulation_root(self):
        class Cfg:
            def __init__(self, prim_path):
                self.prim_path = prim_path

        class Key:
            def __init__(self, prim_path):
                self.cfg = Cfg(prim_path)
                self.body_names = ["link7", "link8"]

        class RobotManager:
            robot_list = [object(), object()]
            robot_key = [Key("{ENV_REGEX_NS}/robot0"), Key("{ENV_REGEX_NS}/robot1")]

        cfg = load_tactile_config(
            {
                "enabled": True,
                "left": {
                    "body_name": "link7",
                    "rigid_body_prim_path": "/World/envs/env_.*/robot1",
                    "gelpad_prim_path": "/World/envs/env_.*/robot1/gelpad",
                    "sensor_prim_path": "/World/envs/env_.*/robot1/sensor",
                    "optical_field_mm": [20, 20],
                },
                "right": {
                    "body_name": "link8",
                    "rigid_body_prim_path": "/World/envs/env_.*/robot0",
                    "gelpad_prim_path": "/World/envs/env_.*/robot0/gelpad",
                    "sensor_prim_path": "/World/envs/env_.*/robot0/sensor",
                    "optical_field_mm": [20, 20],
                },
            }
        )
        _, left_key = _resolve_robot_for_sensor(RobotManager(), cfg.left)
        _, right_key = _resolve_robot_for_sensor(RobotManager(), cfg.right)
        self.assertEqual(left_key.cfg.prim_path, "{ENV_REGEX_NS}/robot1")
        self.assertEqual(right_key.cfg.prim_path, "{ENV_REGEX_NS}/robot0")

    def test_duplicate_body_names_without_root_are_rejected(self):
        class Key:
            body_names = ["link7"]
            cfg = type("Cfg", (), {"prim_path": "{ENV_REGEX_NS}/robot0"})()

        class RobotManager:
            robot_list = [object(), object()]
            robot_key = [Key(), Key()]

        cfg = load_tactile_config(
            {
                "enabled": True,
                "left": {
                    "body_name": "link7",
                    "gelpad_prim_path": "/World/gelpad_left",
                    "sensor_prim_path": "/World/sensor_left",
                    "optical_field_mm": [20, 20],
                },
                "right": {
                    "body_name": "link8",
                    "gelpad_prim_path": "/World/gelpad_right",
                    "sensor_prim_path": "/World/sensor_right",
                    "optical_field_mm": [20, 20],
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "multiple robots"):
            _resolve_robot_for_sensor(RobotManager(), cfg.left)

    def test_render_sync_is_idempotent_for_one_physics_generation(self):
        class FakeUipc:
            def __init__(self):
                self.calls = 0

            def update_render_meshes(self):
                self.calls += 1

        manager = RoboDojoTactileManager({"enabled": False})
        manager.enabled = True
        manager.uipc_sim = FakeUipc()
        manager._physics_generation = 1
        manager.before_render()
        manager.before_render()
        self.assertEqual(manager.uipc_sim.calls, 1)
        manager._physics_generation = 2
        manager.before_render()
        self.assertEqual(manager.uipc_sim.calls, 2)

    def test_enabled_initialization_requires_native_scene_manager(self):
        manager = RoboDojoTactileManager({"enabled": False})
        manager.initialize(None, None, None)
        self.assertFalse(manager.initialized)
        manager.enabled = True
        with self.assertRaisesRegex(ValueError, "native SceneManager"):
            manager.initialize(None, None, None)

    def test_disabled_manager_diagnostics_are_safe(self):
        manager = RoboDojoTactileManager({"enabled": False})
        diagnostics = manager.diagnostics()
        self.assertFalse(diagnostics["enabled"])
        self.assertFalse(diagnostics["suspended"])
        self.assertEqual(diagnostics["sensor_names"], [])
        self.assertEqual(diagnostics["reset_count"], 0)
        self.assertEqual(diagnostics["capture_count"], 0)
        manager.close()


    def test_obs_manager_assembles_tactile_observation(self):
        from env.observation_manager.obs_manager import ObsManager

        class FakeTactileManager:
            enabled = True
            initialized = True
            suspended = False

            def before_render(self):
                pass

            def capture(self, dt):
                self.capture_dt = dt

            def observe(self, env_ids):
                return {
                    int(env_id): {
                        "left": {"rgb": "left_rgb", "valid": True},
                        "right": {"rgb": "right_rgb", "valid": True},
                    }
                    for env_id in env_ids
                }

        obs_manager = ObsManager.__new__(ObsManager)
        obs_manager.num_envs = 1
        obs_manager.dt = 0.004
        obs_manager.collect_freq = 25
        obs_manager.instruction = ["touch"]
        obs_manager.robot_manager = None
        obs_manager.robot_cfg = {}
        obs_manager.camera_manager = None
        obs_manager.capture_manager = None
        obs_manager.tactile_manager = FakeTactileManager()

        observation = obs_manager.get_obs(env_idx_list=[0])
        self.assertEqual(observation[0]["tactile"]["left"]["rgb"], "left_rgb")
        self.assertTrue(observation[0]["tactile"]["right"]["valid"])

    def test_manager_can_be_suspended_during_reset(self):
        manager = RoboDojoTactileManager({"enabled": False})
        manager.suspend()
        self.assertTrue(manager.suspended)
        self.assertTrue(manager.diagnostics()["suspended"])
        manager.resume()
        self.assertFalse(manager.suspended)
        manager.close()

    def test_resume_leaves_world_stepping_to_the_environment(self):
        class FakeIsaacSim:
            def __init__(self):
                self.removed = []
                self.added = []
                self.callbacks = {"uicp_step"}

            def physics_callback_exists(self, name):
                return name in self.callbacks

            def remove_physics_callback(self, name):
                self.removed.append(name)
                self.callbacks.remove(name)

            def add_physics_callback(self, name, callback):
                self.added.append((name, callback))
                self.callbacks.add(name)

        class FakeUipc:
            def __init__(self):
                self.isaac_sim = FakeIsaacSim()
                self.scene = SimpleNamespace(animator=lambda: SimpleNamespace(erase=lambda object_id: None))

            def step(self, dt=0.0):
                return dt

        manager = RoboDojoTactileManager({"enabled": False})
        manager.enabled = True
        manager.initialized = True
        manager.uipc_sim = FakeUipc()
        manager._uipc_step_callback_registered = True
        manager.suspend()
        self.assertEqual(manager.uipc_sim.isaac_sim.removed, ["uicp_step"])
        self.assertFalse(manager._uipc_step_callback_registered)
        manager.resume()
        self.assertEqual(manager.uipc_sim.isaac_sim.added, [])
        self.assertFalse(manager._uipc_step_callback_registered)
        manager.close()

    def test_manual_runtime_device_is_set_for_play_callback_fallback(self):
        class Obj:
            _device = None

        obj = Obj()
        _set_manual_runtime_device(obj, "cuda")
        self.assertEqual(obj._device, "cuda")
        # Do not overwrite a device selected by the backend/runtime.
        _set_manual_runtime_device(obj, "cuda:1")
        self.assertEqual(obj._device, "cuda")

    def test_missing_optional_backend_has_actionable_error(self):
        if all(
            importlib.util.find_spec(name) is not None
            for name in ("tacex", "tacex_assets", "tacex_uipc", "torch_scatter")
        ):
            self.skipTest("TacEx optional dependencies are installed in this runtime")
        try:
            _require_backend_modules()
        except TactileDependencyError as exc:
            ctx = exc
        else:
            self.fail("missing-backend test unexpectedly loaded the optional backend")
        self.assertIn("tactile.enabled=true", str(ctx))

    def test_disabled_config_is_import_light_and_does_not_require_sensor_paths(self):
        before = set(sys.modules)
        cfg = load_tactile_config({"enabled": False})
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.left)
        self.assertIsNone(cfg.right)
        self.assertFalse({"tacex", "tacex_assets", "tacex_uipc", "uipc"} & (set(sys.modules) - before))

    def test_enabled_config_requires_both_sensors(self):
        with self.assertRaisesRegex(TactileConfigError, "left.body_name"):
            load_tactile_config({"enabled": True})

    def test_enabled_config_requires_physical_field(self):
        with self.assertRaisesRegex(TactileConfigError, "requires optical_field_mm"):
            load_tactile_config(
                {
                    "enabled": True,
                    "left": {
                        "body_name": "left_finger",
                        "gelpad_prim_path": "/World/gelpad_left",
                        "sensor_prim_path": "/World/sensor_left",
                    },
                    "right": {
                        "body_name": "right_finger",
                        "gelpad_prim_path": "/World/gelpad_right",
                        "sensor_prim_path": "/World/sensor_right",
                        "optical_field_mm": [20, 20],
                    },
                }
            )

    def test_enabled_config_accepts_explicit_paths(self):
        cfg = load_tactile_config(
            {
                "enabled": True,
                "left": {
                    "body_name": "left_finger",
                    "gelpad_prim_path": "/World/gelpad_left",
                    "sensor_prim_path": "/World/sensor_left",
                    "optical_field_mm": [20, 20],
                },
                "right": {
                    "body_name": "right_finger",
                    "gelpad_prim_path": "/World/gelpad_right",
                    "sensor_prim_path": "/World/sensor_right",
                    "optical_field_mm": [20, 20],
                },
            }
        )
        self.assertEqual(cfg.resolution, (128, 128))
        self.assertEqual(cfg.left.body_name, "left_finger")
        self.assertEqual(cfg.left.calibration_id, "left")
        self.assertEqual(cfg.left.camera_clipping_range, (0.024, 0.034))
        self.assertEqual(cfg.left.camera_focal_length, 1.05)


    def test_invalid_v1_values_are_rejected(self):
        with self.assertRaises(TactileConfigError):
            load_tactile_config({"resolution": [64, 64]})
        with self.assertRaises(TactileConfigError):
            load_tactile_config({"optical_backend": "pix2pix"})
        with self.assertRaises(TactileConfigError):
            load_tactile_config(
                {
                    "enabled": True,
                    "left": {
                        "body_name": "left_finger",
                        "gelpad_prim_path": "/World/gelpad_left",
                        "sensor_prim_path": "/World/sensor_left",
                        "optical_field_mm": [20, 20],
                        "camera_clipping_range": [0.03, 0.02],
                    },
                    "right": {
                        "body_name": "right_finger",
                        "gelpad_prim_path": "/World/gelpad_right",
                        "sensor_prim_path": "/World/sensor_right",
                        "optical_field_mm": [20, 20],
                    },
                }
            )

    def test_enabled_taxim_requires_cuda(self):
        with self.assertRaisesRegex(TactileConfigError, "requires a CUDA device"):
            load_tactile_config(
                {
                    "enabled": True,
                    "device": "cpu",
                    "left": {
                        "body_name": "left_finger",
                        "gelpad_prim_path": "/World/gelpad_left",
                        "sensor_prim_path": "/World/sensor_left",
                        "optical_field_mm": [20, 20],
                    },
                    "right": {
                        "body_name": "right_finger",
                        "gelpad_prim_path": "/World/gelpad_right",
                        "sensor_prim_path": "/World/sensor_right",
                        "optical_field_mm": [20, 20],
                    },
                }
            )

    def test_calibration_files_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(TactileConfigError):
                validate_calibration_dir(root)
            (root / "params.json").write_text("{}", encoding="utf-8")
            if np is None:
                (root / "gelmap.npy").touch()
            else:
                np.save(root / "gelmap.npy", np.ones((2, 2), dtype=np.float32))
            if np is None:
                for name in ("shadowTable.npz", "polycalib.npz", "dataPack.npz"):
                    (root / name).touch()
            else:
                np.savez(root / "shadowTable.npz")
                np.savez(
                    root / "polycalib.npz",
                    grad_b=np.ones((1,), dtype=np.float32),
                    grad_g=np.ones((1,), dtype=np.float32),
                    grad_r=np.ones((1,), dtype=np.float32),
                )
                np.savez(root / "dataPack.npz", f0=np.ones((1,), dtype=np.float32))
            validate_calibration_dir(root)


@unittest.skipIf(np is None, "numpy is provided by the Isaac/TacEx runtime")
class TactileOutputTests(unittest.TestCase):
    def test_float_hwc_is_scaled_to_uint8_rgb(self):
        value = np.zeros((128, 128, 4), dtype=np.float32)
        value[..., 0] = 1.0
        rgb = _to_numpy_rgb(value, (128, 128))
        self.assertEqual(rgb.shape, (128, 128, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(tuple(rgb[0, 0]), (255, 0, 0))

    def test_float_output_is_rounded_before_cast(self):
        value = np.full((128, 128, 3), 12.6, dtype=np.float32)
        rgb = _to_numpy_rgb(value, (128, 128))
        self.assertEqual(int(rgb[0, 0, 0]), 13)

    def test_chw_output_is_transposed(self):
        value = np.zeros((3, 128, 128), dtype=np.uint8)
        value[1] = 77
        rgb = _to_numpy_rgb(value, (128, 128))
        self.assertEqual(tuple(rgb[0, 0]), (0, 77, 0))

    def test_wrong_resolution_is_rejected(self):
        with self.assertRaises(ValueError):
            _to_numpy_rgb(np.zeros((64, 64, 3), dtype=np.uint8), (128, 128))

    def test_non_finite_float_output_is_rejected(self):
        value = np.zeros((128, 128, 3), dtype=np.float32)
        value[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            _to_numpy_rgb(value, (128, 128))


if __name__ == "__main__":
    unittest.main()
