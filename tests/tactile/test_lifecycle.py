"""Check temporal ordering and episode isolation without launching Isaac."""

import ast
from dataclasses import replace
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from env.tactile.config import TactileSensorConfig
from env.tactile.dynamics import rigid_pose
from env.tactile.gelsight import _TacExGelSight
from env.tactile.manager import RoboDojoTactileManager
from env.tactile.runtime import TactileDependencyError


class LifecycleTests(unittest.TestCase):
    def test_close_breaks_python_owner_cycles_without_gc(self):
        import gc
        import weakref

        class Resource:
            def _clear_callbacks(self):
                pass

        class OpticalSimulator:
            def __init__(self, sensor):
                self.sensor = sensor

            def compute_indentation_depth(self):
                return None

        class Sensor(Resource):
            def compute_indentation_depth(self):
                return None

        def closed_world_reference(depth_owner):
            world = Resource()
            simulation = Resource()
            simulation.world = world
            gelpad = Resource()
            gelpad._uipc_sim = simulation
            gelpad.uipc_scene_objects = [Resource()]
            attachment = Resource()
            attachment.uipc_object = gelpad
            gelpad.constraint = attachment
            sensor = Sensor()
            sensor.camera = None
            sensor.gelpad_obj = gelpad
            sensor.marker_motion_simulator = None
            sensor.optical_simulator = OpticalSimulator(sensor)
            depth_source = sensor.optical_simulator if depth_owner == "optical" else sensor
            sensor.compute_indentation_depth_func = depth_source.compute_indentation_depth
            adapter = _TacExGelSight.__new__(_TacExGelSight)
            adapter.sensor = sensor
            adapter.gelpad = gelpad
            adapter.attachment = attachment
            world_reference = weakref.ref(world)
            scene_objects = gelpad.uipc_scene_objects

            # These are the actual adapter cleanup calls. The manager has
            # already erased animator callbacks and will clear its registries.
            adapter.close()
            adapter.close()
            self.assertIs(adapter.gelpad, gelpad)
            self.assertIs(adapter.attachment, attachment)
            self.assertIs(adapter.gelpad.uipc_scene_objects, scene_objects)
            self.assertTrue(callable(adapter.gelpad._clear_callbacks))
            self.assertTrue(callable(adapter.attachment._clear_callbacks))
            return world_reference

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for depth_owner in ("optical", "sensor"):
                with self.subTest(depth_owner=depth_owner):
                    world_reference = closed_world_reference(depth_owner)
                    self.assertIsNone(world_reference(), "Closed TacEx Python cycles retain the previous world")
        finally:
            # Restore process GC state and collect failing-baseline cycles
            # only after observing whether ordinary reference release worked.
            if gc_was_enabled:
                gc.enable()
            gc.collect()

    def test_failed_second_follower_releases_first_and_preserves_cleanup_errors(self):
        import tempfile

        from env.tactile import dynamics, initialization, manager as manager_module, scene

        for cleanup_fails in (False, True):
            with self.subTest(cleanup_fails=cleanup_fails), tempfile.TemporaryDirectory() as directory:
                manager = RoboDojoTactileManager(None, workspace=directory)
                manager.enabled = True
                manager.mount = lambda *_: None
                manager._restore_gel_assets = lambda: None
                manager._configure_contacts = lambda: None
                manager._pause_uipc_step_callback = lambda: None
                ready_flags = []
                manager._initialize_handles = lambda: ready_flags.append(manager.initialized)
                wrappers = {name: SimpleNamespace(backend=None) for name in ("first", "second")}
                removed = []
                stage = SimpleNamespace(
                    GetPrimAtPath=lambda _: SimpleNamespace(IsValid=lambda: True),
                    RemovePrim=lambda path: removed.append(path),
                )
                bindings = {name: SimpleNamespace(name=name, robot_key=None,
                    scene_object=SimpleNamespace(dynamic=True, wrapper=SimpleNamespace(stage=stage)))
                    for name in wrappers}
                prepared = SimpleNamespace(specs=[], bindings=bindings, root_path="/hidden")
                manager.uipc_rigid_objects = {name: SimpleNamespace(uipc_scene_objects=[]) for name in wrappers}
                fake_sim = SimpleNamespace(
                    world=SimpleNamespace(is_valid=lambda: True), setup_sim=lambda: None,
                    scene=SimpleNamespace(animator=lambda: SimpleNamespace(erase=lambda _: None)),
                )
                def fake_cfg(**_):
                    return SimpleNamespace()

                fake_cfg.Contact = lambda **_: SimpleNamespace()
                fake_cfg.Newton = lambda **_: SimpleNamespace()
                fake_cfg.LinearSystem = lambda **_: SimpleNamespace()
                initialization_error = RuntimeError("second follower rejected")
                cleanup_error = RuntimeError("cleanup rejected")

                def follower(binding, *_):
                    ready_flags.append(manager.initialized)
                    if binding.name == "second":
                        raise initialization_error
                    wrapper = wrappers[binding.name]
                    wrapper.backend = "owned"
                    return SimpleNamespace(close=lambda: setattr(wrapper, "backend", None))

                close = manager.close

                def cleanup():
                    close()
                    if cleanup_fails:
                        raise cleanup_error

                manager.close = cleanup
                modules = {
                    "uipc": SimpleNamespace(
                        Timer=SimpleNamespace(disable_all=lambda: None, report_as_json=lambda: None)
                    ),
                    "tacex_uipc": SimpleNamespace(UipcSim=lambda _: fake_sim, UipcSimCfg=fake_cfg),
                }
                with patch.object(manager_module, "_default_calibration_dir", return_value=None), \
                     patch.object(scene, "collect_robot_colliders", return_value=[]), \
                     patch.object(scene, "prepare_scene", return_value=prepared), \
                     patch.object(initialization, "prepare_initialization", return_value=None), \
                     patch.object(initialization, "commit_initialization"), \
                     patch.object(initialization, "record_world_initialization"), \
                     patch.object(dynamics, "UipcSceneBody", side_effect=follower), \
                     patch.dict(sys.modules, modules):
                    if cleanup_fails:
                        with self.assertRaises(ExceptionGroup) as caught:
                            manager.initialize(
                                SimpleNamespace(physics_dt=.01), SimpleNamespace(), SimpleNamespace(stage=stage)
                            )
                        self.assertEqual(caught.exception.exceptions, (initialization_error, cleanup_error))
                    else:
                        with self.assertRaises(RuntimeError) as caught:
                            manager.initialize(
                                SimpleNamespace(physics_dt=.01), SimpleNamespace(), SimpleNamespace(stage=stage)
                            )
                        self.assertIs(caught.exception, initialization_error)
                self.assertEqual(ready_flags, [False, False, False])
                self.assertFalse(manager.initialized)
                self.assertIsNone(wrappers["first"].backend)
                self.assertFalse(manager._scene_bodies)
                self.assertFalse(manager.sensors)
                self.assertIsNone(manager.uipc_sim)
                self.assertEqual(removed, ["/hidden"])

    def test_absent_or_removed_callback_is_confirmed_and_idempotent(self):
        for initially_present in (False, True):
            with self.subTest(initially_present=initially_present):
                callbacks = {"uicp_step"} if initially_present else set()
                calls = []

                def remove(name):
                    calls.append(name)
                    callbacks.remove(name)

                manager = RoboDojoTactileManager(None)
                manager._uipc_step_callback_registered = True
                interface = SimpleNamespace(physics_callback_exists=lambda name: name in callbacks)
                if initially_present:
                    interface.remove_physics_callback = remove
                manager.uipc_sim = SimpleNamespace(isaac_sim=interface)
                manager._pause_uipc_step_callback()
                manager._pause_uipc_step_callback()
                self.assertFalse(manager._uipc_step_callback_registered)
                self.assertNotIn("uicp_step", callbacks)
                self.assertEqual(calls, ["uicp_step"] if initially_present else [])

        # Pinned TacEx setup_sim sets isaac_sim then installs uicp_step. The
        # manager checks world validity before setting its registered flag.
        for partial_stage in (
            "invalid_world_after_setup", "error_after_callback_install", "error_before_callback_install"
        ):
            with self.subTest(partial_stage=partial_stage):
                callbacks = set()
                calls = []

                def remove_partial(name):
                    calls.append("remove")
                    callbacks.remove(name)

                interface = SimpleNamespace(
                    physics_callback_exists=lambda name: name in callbacks,
                    remove_physics_callback=remove_partial,
                )
                manager = RoboDojoTactileManager(None)
                manager._uipc_callback_names.add("uicp_step")
                manager.uipc_sim = SimpleNamespace(
                    isaac_sim=None,
                    scene=SimpleNamespace(animator=lambda: calls.append("dispose") or SimpleNamespace()),
                )
                if partial_stage != "error_before_callback_install":
                    manager.uipc_sim.isaac_sim = interface
                    callbacks.add("uicp_step")
                self.assertFalse(manager._uipc_step_callback_registered)
                manager.close()
                self.assertFalse(manager._uipc_step_callback_registered)
                self.assertNotIn("uicp_step", callbacks)
                self.assertIsNone(manager.uipc_sim)
                expected = ["dispose"] if partial_stage == "error_before_callback_install" else ["remove", "dispose"]
                self.assertEqual(calls, expected)

    def test_uncertain_removal_keeps_ownership_and_blocks_cleanup(self):
        for failure in (
            "remove_error", "remove_noop", "missing_remove", "missing_exists",
            "exists_error", "missing_world", "partial_setup_remove_error",
        ):
            with self.subTest(failure=failure):
                callbacks = {"uicp_step"}
                destroyed = []

                def remove(name):
                    if failure in ("remove_error", "partial_setup_remove_error"):
                        raise RuntimeError("callback registry failed")
                    if failure != "remove_noop":
                        callbacks.remove(name)

                def exists(name):
                    if failure == "exists_error":
                        raise RuntimeError("callback lookup failed")
                    return name in callbacks

                manager = RoboDojoTactileManager(None)
                manager.enabled = True
                manager.initialized = failure != "partial_setup_remove_error"
                manager._uipc_callback_names.add("uicp_step")
                manager._uipc_step_callback_registered = failure != "partial_setup_remove_error"
                interface = SimpleNamespace()
                if failure != "missing_remove":
                    interface.remove_physics_callback = remove
                if failure != "missing_exists":
                    interface.physics_callback_exists = exists
                manager.uipc_sim = None if failure == "missing_world" else SimpleNamespace(
                    isaac_sim=interface,
                    scene=SimpleNamespace(animator=lambda: destroyed.append("world")),
                )
                original_world = manager.uipc_sim
                if failure != "partial_setup_remove_error":
                    with self.assertRaises(RuntimeError):
                        manager._pause_uipc_step_callback()
                    self.assertTrue(manager._uipc_step_callback_registered)
                else:
                    # Callback installation preceded the manager's flag update.
                    self.assertFalse(manager._uipc_step_callback_registered)
                # initialize() calls close() on failure: it must not mask the
                # failed removal, discard ownership, or release a live world.
                with self.assertRaises(RuntimeError):
                    manager.close()
                self.assertTrue(manager._uipc_step_callback_registered)
                self.assertEqual(manager.initialized, failure != "partial_setup_remove_error")
                self.assertIs(manager.uipc_sim, original_world)
                self.assertEqual(destroyed, [])
                self.assertIn("uicp_step", callbacks)


    def test_all_owned_callbacks_are_verified_before_world_disposal(self):
        for initially_present in (set(), {"uicp_step", "a_attachment", "z_attachment"}):
            with self.subTest(initially_present=initially_present):
                callbacks = initially_present.copy()
                events = []

                def remove(name):
                    events.append(("remove", name))
                    callbacks.remove(name)

                manager = RoboDojoTactileManager(None)
                manager.enabled = manager.initialized = True
                manager._uipc_callback_names = {"uicp_step", "a_attachment", "z_attachment", "unused_alias"}
                manager.uipc_sim = SimpleNamespace(
                    isaac_sim=SimpleNamespace(physics_callback_exists=lambda name: name in callbacks,
                                             remove_physics_callback=remove),
                    scene=SimpleNamespace(animator=lambda: events.append("dispose") or SimpleNamespace()),
                )
                manager.close()
                manager.close()
                self.assertFalse(callbacks)
                self.assertIsNone(manager.uipc_sim)
                self.assertFalse(manager._uipc_callback_names)
                self.assertEqual(events[-1], "dispose")
                self.assertEqual([event[1] for event in events[:-1]],
                                 ["uicp_step", "a_attachment", "z_attachment"] if initially_present else [])

    def test_uncertain_attachment_removal_preserves_world_and_tries_remaining_callbacks(self):
        for failure in (
            "remove_error", "remove_noop", "exists_error", "missing_remove", "missing_exists", "step_remove_error"
        ):
            with self.subTest(failure=failure):
                callbacks = {"uicp_step", "a_attachment", "z_attachment"}
                events = []

                def remove(name):
                    events.append(("remove", name))
                    if (
                        (failure == "remove_error" and name == "a_attachment")
                        or (failure == "step_remove_error" and name == "uicp_step")
                    ):
                        raise RuntimeError("callback removal failed")
                    if failure != "remove_noop" or name != "a_attachment":
                        callbacks.remove(name)

                def exists(name):
                    if failure == "exists_error" and name == "a_attachment":
                        raise RuntimeError("callback lookup failed")
                    return name in callbacks

                interface = SimpleNamespace()
                if failure != "missing_remove":
                    interface.remove_physics_callback = remove
                if failure != "missing_exists":
                    interface.physics_callback_exists = exists
                manager = RoboDojoTactileManager(None)
                manager.enabled = manager.initialized = True
                manager._uipc_callback_names = callbacks.copy()
                manager.sensors["left"] = SimpleNamespace(
                    close=lambda: events.append("sensor disposed"),
                    gelpad=SimpleNamespace(uipc_scene_objects=[]),
                )
                manager.uipc_sim = SimpleNamespace(
                    isaac_sim=interface,
                    scene=SimpleNamespace(animator=lambda: events.append("world disposed") or SimpleNamespace()),
                )
                original_world = manager.uipc_sim
                with self.assertRaises((RuntimeError, ExceptionGroup)):
                    manager.close()
                self.assertIs(manager.uipc_sim, original_world)
                self.assertTrue(manager.initialized)
                self.assertEqual(manager._uipc_callback_names, {"uicp_step", "a_attachment", "z_attachment"})
                self.assertIn("left", manager.sensors)
                self.assertNotIn("world disposed", events)
                self.assertNotIn("sensor disposed", events)
                if failure not in {"missing_remove", "missing_exists"}:
                    self.assertIn(("remove", "z_attachment"), events)
                    self.assertNotIn("z_attachment", callbacks)
                self.assertEqual(
                    manager._uipc_step_callback_registered,
                    failure in {"missing_remove", "missing_exists", "step_remove_error"},
                )
                # Once the registry works again, retry can safely finish the
                # same cleanup instead of abandoning a callback-owned world.
                interface.physics_callback_exists = lambda name: name in callbacks
                interface.remove_physics_callback = lambda name: callbacks.remove(name)
                manager.close()
                self.assertFalse(callbacks)
                self.assertIsNone(manager.uipc_sim)
                self.assertFalse(manager.sensors)
                self.assertIn("world disposed", events)
                self.assertIn("sensor disposed", events)

    def test_attachment_uses_robot_device_when_contact_solver_is_on_cuda(self):
        sensor = _TacExGelSight.__new__(_TacExGelSight)
        attachment = SimpleNamespace(
            _is_initialized=False, isaaclab_rigid_object=SimpleNamespace(device="cpu")
        )
        initialized_devices = []
        attachment._initialize_impl = lambda: initialized_devices.append(attachment._device)
        sensor.attachment = attachment
        sensor.gelpad = SimpleNamespace(uipc_sim=SimpleNamespace(cfg=SimpleNamespace(device="cuda")))
        sensor._ensure_attachment_initialized()
        self.assertEqual(initialized_devices, ["cpu"])
        self.assertTrue(attachment._is_initialized)

    def test_physics_uses_current_robot_pose_before_advancing_contacts(self):
        events = []
        manager = RoboDojoTactileManager(None)
        manager.enabled = manager.initialized = True
        transforms = SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: [42])))
        robot = SimpleNamespace(root_physx_view=SimpleNamespace(get_link_transforms=lambda: transforms))
        manager._robot_bodies["finger"] = SimpleNamespace(
            robot=robot, set_target=lambda value: events.append(("robot", value))
        )
        manager.sensors["left"] = SimpleNamespace(
            attachment=SimpleNamespace(_compute_aim_positions=lambda: events.append("attachment")),
            physics_step=lambda dt: events.append("sensor"),
        )
        manager.uipc_rigid_objects["object"] = SimpleNamespace(
            _is_initialized=True, update=lambda dt: events.append("body")
        )
        manager._scene_bodies["object"] = SimpleNamespace(sync_visual=lambda: events.append("visual"))
        manager.uipc_sim = SimpleNamespace(
            step=lambda dt: events.append("advance"), world=SimpleNamespace(is_valid=lambda: True)
        )
        manager.physics_step(0.004, physx_elapsed_s=0.125)
        self.assertEqual(events, [("robot", [42]), "attachment", "advance", "body", "visual", "sensor"])
        manager.suspend()
        manager.physics_step(0.004)
        self.assertEqual(len(events), 6)
        self.assertEqual(manager._physics_generation, 1)
        self.assertEqual(manager._step_timings["physx_s"], 0.125)

    def test_solver_failure_stops_publication_and_preserves_error_for_cleanup(self):
        for profiled in (False, True):
            for raises in (False, True):
                with self.subTest(profiled=profiled, raises=raises):
                    events = []
                    manager = RoboDojoTactileManager(None)
                    manager.enabled = manager.initialized = True
                    manager.sensors["left"] = SimpleNamespace(
                        attachment=SimpleNamespace(_compute_aim_positions=lambda: None),
                        physics_step=lambda dt: events.append("sensor"),
                        invalidate=lambda: events.append("invalidate"),
                    )
                    manager.uipc_rigid_objects["object"] = SimpleNamespace(
                        _is_initialized=True, update=lambda dt: events.append("body")
                    )
                    manager._scene_bodies["object"] = SimpleNamespace(
                        sync_visual=lambda: events.append("visual")
                    )
                    failure = RuntimeError("solver rejected step")

                    def step(*args, **kwargs):
                        events.append("advance")
                        if raises:
                            raise failure

                    manager.uipc_sim = SimpleNamespace(
                        step=step, world=SimpleNamespace(is_valid=lambda: False)
                    )
                    if profiled:
                        manager._solver_profiler = SimpleNamespace(step=step)
                    with self.assertRaises(RuntimeError) as caught:
                        manager.physics_step(0.004)
                    if raises:
                        self.assertIs(caught.exception, failure)
                    else:
                        self.assertIn("close and rebuild", str(caught.exception))
                    self.assertEqual(events, ["advance", "invalidate"])
                    self.assertEqual(manager._physics_generation, 0)
                    self.assertTrue(manager._suspended)
                    self.assertTrue(manager.initialized)  # Cleanup still owns this world.
                    self.assertIsNotNone(manager.uipc_sim)
                    self.assertIn(str(caught.exception), manager._last_error)
                    manager.physics_step(0.004)
                    manager.capture()
                    self.assertEqual(events, ["advance", "invalidate"])

    def test_capture_is_one_frame_per_physics_state_with_simulation_timestamp(self):
        calls = []
        manager = RoboDojoTactileManager(None)
        manager.enabled = manager.initialized = True
        manager.sim = SimpleNamespace(_sim_step_counter=77, physics_dt=0.004)
        sensor = SimpleNamespace(
            sensor_cfg=SimpleNamespace(calibration_id="calibrated"),
            capture=lambda **kwargs: calls.append(kwargs),
            observe=lambda: {"rgb": np.zeros((2, 2, 3), dtype=np.uint8), "valid": True},
            restore_render_occluder=lambda: None,
            reset=lambda: None,
        )
        manager.sensors["left"] = sensor
        manager.capture()
        manager.capture()
        observation = manager.observe([0])[0]["left"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(observation["frame_id"], 0)
        self.assertAlmostEqual(observation["capture_sim_time"], 0.308)
        self.assertEqual(observation["age_steps"], 0)
        manager._physics_generation += 2
        self.assertEqual(manager.observe([0])[0]["left"]["age_steps"], 2)
        manager.reset()
        self.assertIsNone(manager.observe([0])[0]["left"]["capture_sim_time"])

    def test_begin_reset_disposes_old_state_and_callbacks_once(self):
        closed = []
        manager = RoboDojoTactileManager(None)
        manager.enabled = manager.initialized = True
        manager.sensors["left"] = SimpleNamespace(close=lambda: closed.append("sensor"))
        manager._scene_bodies["object"] = SimpleNamespace(close=lambda: closed.append("native_binding"))
        manager.uipc_rigid_objects["object"] = SimpleNamespace(_clear_callbacks=lambda: closed.append("body"))
        manager._capture_generation = 42
        manager._capture_sim_time = 12.0
        manager.begin_reset()
        manager.begin_reset()
        self.assertEqual(closed, ["native_binding", "sensor", "body"])
        self.assertFalse(manager.initialized)
        self.assertTrue(manager.suspended)
        self.assertFalse(manager.sensors)
        self.assertEqual(manager._capture_generation, -1)
        self.assertIsNone(manager._capture_sim_time)


class EvaluationCloseTests(unittest.TestCase):
    def test_setup_scene_keeps_native_order_without_reapplying_enabled_layout(self):
        with patch.object(sys, "path", [*sys.path, str(Path(__file__).resolve().parents[2] / "XPolicyLab")]):
            from src.eval_client import eval_env
        from task.RoboDojo import task_registry

        class WithoutSimulation(type):
            def __call__(cls, *args, **kwargs):
                return object.__new__(cls)

        class NativeTask(metaclass=WithoutSimulation):
            pass

        for enabled in (False, True):
            with self.subTest(tactile_enabled=enabled):
                calls = []
                with patch.object(task_registry, "load_task_class", return_value=("reset_fixture", NativeTask)):
                    environment = eval_env.create_eval_env(
                        SimpleNamespace(eval_cfg={"task_name": "reset_fixture"}), None
                    )

                def stability(actual):
                    self.assertIs(actual, environment)
                    calls.append("stability")
                    return True, []

                environment.num_envs = environment.episode_nums = 1
                environment.unstable_nums = 0
                environment.tactile_manager = SimpleNamespace(enabled=enabled, initialized=enabled)
                environment.scene_manager = SimpleNamespace(
                    apply_saved_poses=lambda *, env_idx_list: calls.append(("restore", env_idx_list)),
                    layout_manager=SimpleNamespace(check_layout_stability=stability),
                )
                environment._align_layout_success = lambda: calls.append("align")
                environment.render = lambda: calls.append("render")
                environment.sim_step = lambda: calls.append("step")
                environment.obs_manager = SimpleNamespace(
                    render_for_capture=lambda: calls.append("capture_fence"),
                    get_obs=lambda: calls.append("observation"),
                )
                environment.physx_monitor_enabled = True
                environment._check_physx_broken_envs = lambda: calls.append("physx_monitor")
                environment.setup_scene()
                expected = ([] if enabled else [("restore", [0])]) + ["align", "stability"] + ["render"] * 10
                for index in range(200):
                    expected.append("step")
                    if index % 5 == 0:
                        expected.extend(("capture_fence", "observation"))
                expected.append("physx_monitor")
                self.assertEqual(calls, expected)
                self.assertEqual(environment.episode_nums, 1)
                self.assertEqual(environment.unstable_nums, 0)

    def test_video_failure_still_closes_native_environment_and_preserves_errors(self):
        with patch.object(sys, "path", [*sys.path, str(Path(__file__).resolve().parents[2] / "XPolicyLab")]):
            from src.eval_client import eval_env
        from task.RoboDojo import task_registry

        class WithoutSimulation(type):
            def __call__(cls, *args, **kwargs):
                # Keep the real factory-generated EvalEnv methods and super()
                # relationship; only skip its simulator-heavy constructor.
                return object.__new__(cls)

        for scenario in ("normal", "video_only", "multiple_failures"):
            with self.subTest(scenario=scenario):
                calls = []
                video_error = RuntimeError("video cleanup failed")
                native_error = RuntimeError("native close failed")

                def abort_video():
                    calls.append("video")
                    if scenario != "normal":
                        raise video_error

                class NativeTask(metaclass=WithoutSimulation):
                    def close(self):
                        calls.append("native")
                        if scenario == "multiple_failures":
                            raise native_error

                with patch.object(task_registry, "load_task_class", return_value=("close_fixture", NativeTask)):
                    environment = eval_env.create_eval_env(
                        SimpleNamespace(eval_cfg={"task_name": "close_fixture"}), None
                    )
                environment._abort_video_writers = abort_video
                environment.obs_manager = SimpleNamespace(reset=lambda: calls.append("observation"))
                if scenario == "normal":
                    self.assertIsNone(environment.close())
                elif scenario == "video_only":
                    with self.assertRaises(RuntimeError) as caught:
                        environment.close()
                    self.assertIs(caught.exception, video_error)
                else:
                    with self.assertRaises(ExceptionGroup) as caught:
                        environment.close()
                    self.assertEqual(caught.exception.exceptions, (video_error, native_error))
                self.assertEqual(calls, ["video", "observation", "native"])


class DynamicsMathTests(unittest.TestCase):
    def test_pose_removes_scale_without_changing_orientation(self):
        transform = np.array([[0, -3, 0, 0.1], [2, 0, 0, 0.2], [0, 0, 4, 0.3], [0, 0, 0, 1]])
        position, quaternion = rigid_pose(transform)
        np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(np.abs(quaternion), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-12)

    def test_nonfinite_solver_state_is_reported(self):
        transform = np.eye(4)
        transform[0, 3] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            rigid_pose(transform)


class InitializationProjectionTests(unittest.TestCase):
    def test_broadphase_constructs_only_intersecting_tetrahedra(self):
        from env.tactile.initialization import Body, ConvexPiece, LazyPieces, pair_witnesses

        tet = np.array([[0, 0, 0], [.001, 0, 0], [0, .001, 0], [0, 0, .001]])
        vertices = np.concatenate([tet + [index * .01, 0, 0] for index in range(100)])
        tets = np.arange(len(vertices)).reshape(-1, 4)
        pieces = LazyPieces(vertices, tets)
        pieces.add_tetrahedra(range(len(tets)))
        pose = np.eye(4).reshape(-1).tolist()
        large = Body("large", False, pieces, pose, "fixture", (vertices.min(0), vertices.max(0)))
        tiny = Body("tiny", True, [ConvexPiece.from_vertices(tet + [.0001, 0, 0])], pose, "fixture")
        witnesses, candidates = pair_witnesses(large, tiny, np.zeros(3), np.zeros(3), clearance=2e-6, tolerance=1e-10)
        self.assertEqual(candidates, 1)
        self.assertEqual(len(witnesses), 1)
        self.assertEqual(set(pieces.cache), {0})

    def test_detector_catches_crossing_and_containment_but_rejects_aabb_false_positive(self):
        from env.tactile.initialization import ConvexPiece, sat_witness

        crossed = np.array([[x, y, z] for x in [-.01, .01] for y in [-.0001, .0001] for z in [-.0001, .0001]])
        self.assertIsNotNone(
            sat_witness(ConvexPiece.from_vertices(crossed), ConvexPiece.from_vertices(crossed[:, [1, 0, 2]]))
        )
        self.assertIsNotNone(sat_witness(ConvexPiece.from_vertices(crossed), ConvexPiece.from_vertices(crossed * 100)))
        left = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        right = np.array([[1, 1, 1], [.4, 1, 1], [1, .4, 1], [1, 1, .4]], float)
        self.assertTrue(np.all(left.max(0) >= right.min(0)))
        self.assertIsNone(sat_witness(ConvexPiece.from_vertices(left), ConvexPiece.from_vertices(right)))

    def test_initialization_commit_rejects_stale_geometry_and_rolls_back_all_edits(self):
        import hashlib
        import tempfile

        from env.tactile.initialization import Body, commit_initialization

        class Attribute:
            def __init__(self, value):
                self.value = value
                self.fail_once = False
                self.reject_once = False

            def Get(self):
                return self.value

            def Set(self, value):
                if self.fail_once:
                    self.fail_once = False
                    raise RuntimeError("USD commit rejected")
                if self.reject_once:
                    self.reject_once = False
                    return False
                self.value = value

        for mode in ("success", "stale_fixed_geometry", "commit_failure", "commit_false"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                manager = SimpleNamespace(
                    uipc_rigid_objects={}, _prepared_scene=SimpleNamespace(bindings={}),
                    _initialization_report_path=str(Path(directory) / "initialization.json"),
                    _initialization_report={"bodies": {}, "contact_table": []},
                )
                bodies, meshes, slots, translations = [], {}, {}, {}
                for index, name in enumerate(("first", "second", "fixed")):
                    matrix = np.eye(4)[None]
                    rest = matrix.copy()
                    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
                    tets = np.array([[0, 1, 2, 3]])
                    def attr(value):
                        return SimpleNamespace(view=lambda: value)

                    mesh = SimpleNamespace(
                        positions=lambda p=points: attr(p),
                        tetrahedra=lambda t=tets: SimpleNamespace(topo=lambda: attr(t)),
                        transforms=lambda m=matrix: attr(m),
                        meta=lambda: SimpleNamespace(find=lambda _: attr(np.array([0]))),
                    )
                    rest_mesh = SimpleNamespace(transforms=lambda m=rest: attr(m))
                    meshes[name] = mesh
                    slots[index] = (SimpleNamespace(geometry=lambda m=mesh: m),
                                    SimpleNamespace(geometry=lambda m=rest_mesh: m))
                    effective = np.eye(4)
                    if name != "fixed":
                        effective[:3, 3] = [.0002, 0, 0]
                    digest = hashlib.sha256(points.tobytes() + tets.tobytes() + matrix.tobytes()).hexdigest()
                    manager._initialization_report["bodies"][name] = {
                        "effective_matrix": effective.tolist(), "live_mesh_sha256": digest
                    }
                    bodies.append(Body(name, name != "fixed", [], np.eye(4).reshape(-1).tolist(), "fixture"))
                    translations[name] = Attribute((0, 0, 0))
                    stage = SimpleNamespace(
                        GetPrimAtPath=lambda _, a=translations[name]: SimpleNamespace(GetAttribute=lambda _: a)
                    )
                    binding = SimpleNamespace(prim_path=f"/hidden/{name}", initial_pose=(0, 0, 0, 1, 0, 0, 0),
                                              scene_object=SimpleNamespace(wrapper=SimpleNamespace(stage=stage)))
                    manager._prepared_scene.bindings[name] = binding
                    manager.uipc_rigid_objects[name] = SimpleNamespace(
                        cfg=SimpleNamespace(init_state=SimpleNamespace(pos=(0, 0, 0), rot=(1, 0, 0, 0))),
                        geo_slot_list=[SimpleNamespace(id=lambda i=index: i)],
                    )
                scene = SimpleNamespace(contact_tabular=lambda: None,
                                        geometries=lambda: SimpleNamespace(find=lambda index: slots[index]))
                manager.uipc_sim = SimpleNamespace(scene=scene)
                if mode == "stale_fixed_geometry":
                    meshes["fixed"].positions().view()[0, 0] += .01
                if mode == "commit_failure":
                    translations["second"].fail_once = True
                if mode == "commit_false":
                    translations["second"].reject_once = True
                modules = {"uipc": SimpleNamespace(view=lambda value: value.view()),
                           "pxr": SimpleNamespace(Gf=SimpleNamespace(Vec3d=lambda *value: tuple(value)))}
                with patch.dict(sys.modules, modules):
                    if mode == "success":
                        commit_initialization(manager, (bodies, meshes, "fingerprint"))
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError, "changed before|USD commit rejected|transform commit was rejected"
                        ):
                            commit_initialization(manager, (bodies, meshes, "fingerprint"))
                for index, name in enumerate(("first", "second", "fixed")):
                    expected = [.0002, 0, 0] if mode == "success" and name != "fixed" else [0, 0, 0]
                    for slot in slots[index]:
                        np.testing.assert_allclose(slot.geometry().transforms().view()[0, :3, 3], expected)
                    np.testing.assert_allclose(translations[name].Get(), expected)


class FailureCleanupTests(unittest.TestCase):
    def test_attachment_dtype_routes_only_explicit_float64_to_supported_vendor(self):
        for dtype, supported in (("float32", False), ("float64", True), ("float64", False)):
            with self.subTest(dtype=dtype, supported=supported):
                (manager, manager_module, scene, prepared, modules, events, _, _,
                 created, camera_error, _) = self._failed_camera_fixture()
                sensor_cfg, robot, key = manager._resolved_robots[0]
                manager._resolved_robots = [(replace(sensor_cfg, attachment_target_dtype=dtype), robot, key)]
                attachment_cfg = Mock(side_effect=lambda **values: SimpleNamespace(**values))
                attachment_cfg.__dataclass_fields__ = {"target_dtype": None} if supported else {}
                modules["tacex_uipc"].UipcIsaacAttachmentsCfg = attachment_cfg
                incompatible = dtype == "float64" and not supported
                error_type = TactileDependencyError if incompatible else RuntimeError
                with (
                    patch.object(manager_module, "_default_calibration_dir", return_value=None),
                    patch.object(scene, "collect_robot_colliders", return_value=[]),
                    patch.object(scene, "prepare_scene", return_value=prepared),
                    patch.dict(sys.modules, modules),
                    self.assertRaises(error_type) as caught,
                ):
                    manager.initialize(SimpleNamespace(physics_dt=.004), SimpleNamespace(), SimpleNamespace(stage=None))
                if incompatible:
                    self.assertIn("target_dtype patch", str(caught.exception))
                    attachment_cfg.assert_not_called()
                    self.assertFalse(created)
                    self.assertFalse(events)
                else:
                    self.assertIs(caught.exception, camera_error)
                    options = attachment_cfg.call_args.kwargs
                    if dtype == "float64":
                        self.assertEqual(options["target_dtype"], "float64")
                    else:
                        self.assertNotIn("target_dtype", options)

    def _failed_camera_fixture(self, *, callback_removal_fails=False):
        from env.tactile import manager as manager_module, scene

        camera_error = RuntimeError("optical camera constructor failed")
        callback_error = RuntimeError("owned attachment callback removal failed")
        events, animations, physics_callbacks, created = [], {}, set(), []
        sensor_cfg = TactileSensorConfig(
            name="left", body_name="link7", sensor_prim_path="/robot/case",
            gelpad_prim_path="/robot/gel", rigid_body_prim_path="/robot",
            sensor_pose=(0, 0, 0, 1, 0, 0, 0),
            calibration_dir=None, camera_clipping_range=(.02, .04),
            camera_focal_length=1.05, camera_focus_distance=.028,
        )
        callback_name = "/robot/gel_X_/robot_attachment_update"
        if callback_removal_fails:
            physics_callbacks.add(callback_name)

        def remove_callback(name):
            if callback_removal_fails:
                raise callback_error
            physics_callbacks.remove(name)

        def erase(actor_id):
            events.append(("erase", actor_id))
            del animations[actor_id]

        simulation = SimpleNamespace(
            scene=SimpleNamespace(animator=lambda: SimpleNamespace(erase=erase)),
            uipc_objects=[],
            isaac_sim=SimpleNamespace(
                physics_callback_exists=lambda name: name in physics_callbacks,
                remove_physics_callback=remove_callback,
            ),
        )

        class OpticalConfig:
            optical_sim_cfg = None
            input_mode = "object"

            def __init__(self):
                self.optical_sim_cfg = self

            def copy(self):
                return self

            def replace(self, **_):
                return self

            def SensorCameraCfg(self, **values):
                return SimpleNamespace(**values)

        class Gel:
            def __init__(self, cfg, owner):
                self.cfg, self._uipc_sim = cfg, owner
                self.uipc_scene_objects = [SimpleNamespace(id=lambda: 7)]
                self.constraint = SimpleNamespace(
                    uipc_object=self,
                    _clear_callbacks=lambda: events.append("attachment clear"),
                )
                self._clear_callbacks = lambda: events.append("gel clear")
                self._initialize_handle = SimpleNamespace(
                    unsubscribe=lambda: events.append("gel unsubscribe")
                )
                self.constraint._initialize_handle = SimpleNamespace(
                    unsubscribe=lambda: events.append("attachment unsubscribe")
                )
                animations[7] = self.constraint
                owner.uipc_objects.append(self)
                created.append(self)

        def optical_sensor(*_):
            events.append("camera constructor")
            raise camera_error

        def deformable_cfg(**values):
            return SimpleNamespace(**values)

        deformable_cfg.StableNeoHookeanCfg = lambda **values: SimpleNamespace(**values)

        def sim_cfg(**_):
            return SimpleNamespace()

        sim_cfg.Contact = lambda **_: SimpleNamespace()
        sim_cfg.Newton = lambda **_: SimpleNamespace()
        sim_cfg.LinearSystem = lambda **_: SimpleNamespace()
        optical = OpticalConfig()
        # This fixture models one attached node and an optical-constructor
        # failure; mounted USD topology is covered by the attachment tests.
        self.enterContext(patch("env.tactile.gelsight._mounted_backplate_indices", return_value=[0]))
        modules = {
            "uipc": SimpleNamespace(Timer=SimpleNamespace(
                disable_all=lambda: None, report_as_json=lambda: None)),
            "tacex": SimpleNamespace(GelSightSensor=optical_sensor),
            "tacex.simulation_approaches.fem_based.sim.gelpad_info": SimpleNamespace(
                CONSTRAIN_PTS={"gsmini": np.asarray([0])}),
            "tacex_assets.sensors.gelsight_mini": SimpleNamespace(
                GELSIGHT_MINI_PIX2PIX_CFG=optical, GELSIGHT_MINI_TAXIM_CFG=optical),
            "tacex_uipc": SimpleNamespace(
                UipcSim=lambda _: simulation, UipcSimCfg=sim_cfg,
                UipcDeformableObject=Gel, UipcDeformableObjectCfg=deformable_cfg,
                UipcIsaacAttachmentsCfg=lambda **values: SimpleNamespace(**values)),
        }
        manager = RoboDojoTactileManager(None)
        manager.enabled = True
        manager.mount = lambda *_: None
        manager._restore_gel_assets = lambda: None
        manager._resolved_robots = [(sensor_cfg, None, SimpleNamespace(cfg=SimpleNamespace(prim_path="/robot")))]
        stage = SimpleNamespace(GetPrimAtPath=lambda _: SimpleNamespace(IsValid=lambda: False))
        prepared = SimpleNamespace(
            specs=[], root_path="/hidden",
            bindings={"fixed": SimpleNamespace(
                robot_key=None, scene_object=SimpleNamespace(wrapper=SimpleNamespace(stage=stage)))},
        )
        # Keep the actual sensor constructor and manager initialization/cleanup.
        # Only SDK components and scene conversion are replaced with owned CPU resources.
        return (manager, manager_module, scene, prepared, modules, events,
                animations, physics_callbacks, created, camera_error, callback_error)

    def test_failed_optical_constructor_releases_owned_gel_animation(self):
        (manager, manager_module, scene, prepared, modules, events, animations,
         physics_callbacks, created, camera_error, _) = self._failed_camera_fixture()
        with patch.object(manager_module, "_default_calibration_dir", return_value=None), \
             patch.object(scene, "collect_robot_colliders", return_value=[]), \
             patch.object(scene, "prepare_scene", return_value=prepared), \
             patch.dict(sys.modules, modules), self.assertRaises(RuntimeError) as caught:
            manager.initialize(SimpleNamespace(physics_dt=.004), SimpleNamespace(), SimpleNamespace(stage=None))
        self.assertIs(caught.exception, camera_error)
        self.assertEqual(events, ["camera constructor", ("erase", 7), "gel clear",
                                  "gel unsubscribe", "attachment clear", "attachment unsubscribe"])
        self.assertFalse(animations)
        self.assertFalse(physics_callbacks)
        self.assertFalse(manager.sensors)
        self.assertIsNone(manager.uipc_sim)
        self.assertIsNone(created[0].constraint)
        self.assertFalse(created[0]._uipc_sim.uipc_objects)

    def test_partial_sensor_waits_for_owned_physics_callback_removal(self):
        (manager, manager_module, scene, prepared, modules, events, animations,
         physics_callbacks, created, camera_error, callback_error) = self._failed_camera_fixture(
             callback_removal_fails=True)
        with patch.object(manager_module, "_default_calibration_dir", return_value=None), \
             patch.object(scene, "collect_robot_colliders", return_value=[]), \
             patch.object(scene, "prepare_scene", return_value=prepared), \
             patch.dict(sys.modules, modules), self.assertRaises(ExceptionGroup) as caught:
            manager.initialize(SimpleNamespace(physics_dt=.004), SimpleNamespace(), SimpleNamespace(stage=None))
        self.assertEqual(caught.exception.exceptions, (camera_error, callback_error))
        self.assertEqual(events, ["camera constructor"])
        self.assertIn(7, animations)
        self.assertTrue(physics_callbacks)
        self.assertIs(manager.sensors["left"].gelpad, created[0])
        self.assertIsNotNone(manager.uipc_sim)
        self.assertIsNotNone(created[0].constraint)

    def _entry_functions(self, *, simulation_app, run_evaluation=None):
        # Importing main.py would construct Kit. Execute only its actual shutdown
        # functions, with CPU fault-injection dependencies, from the checked-in AST.
        source = Path(__file__).resolve().parents[2] / "src/eval_client/main.py"
        functions = {node.name: node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.FunctionDef)}
        namespace = {
            "simulation_app": simulation_app,
            "os": SimpleNamespace(environ={}),
            "sys": SimpleNamespace(stdout=io.StringIO(), stderr=io.StringIO()),
            "_close_model_client": lambda env: None,
            "_run_evaluation": run_evaluation,
        }
        for name in ("_finish_evaluation", "_restart_or_exit", "main"):
            exec(compile(ast.Module(body=[functions[name]], type_ignores=[]), str(source), "exec"), namespace)
        return namespace

    def test_final_cleanup_keeps_evaluation_and_environment_and_application_errors(self):
        calls = []
        original = RuntimeError("solver failed")
        environment_error = OSError("archive cleanup failed")
        application_error = RuntimeError("application close failed")

        def close_environment():
            calls.append("environment close")
            raise environment_error

        def close_application():
            calls.append("application close")
            raise application_error

        app = SimpleNamespace(app=SimpleNamespace(post_quit=lambda code: calls.append(("quit", code))))
        functions = self._entry_functions(simulation_app=app)
        with self.assertRaises(ExceptionGroup) as caught:
            functions["_finish_evaluation"](SimpleNamespace(close=close_environment), app,
                                            close_application, original)
        self.assertEqual(caught.exception.exceptions, (original, environment_error, application_error))
        self.assertEqual(calls, [("quit", 1), "environment close", "application close"])

    def test_normal_final_environment_close_failure_still_closes_application_once(self):
        calls = []
        cleanup_error = RuntimeError("environment close failed")

        def close_environment():
            calls.append("environment close")
            raise cleanup_error

        env = SimpleNamespace(close=close_environment)
        app = SimpleNamespace(app=SimpleNamespace(post_quit=lambda code: calls.append(("quit", code))),
                              close=lambda: calls.append("application close"))
        functions = self._entry_functions(
            simulation_app=app, run_evaluation=lambda lifecycle, close: lifecycle.update(env=env))
        with self.assertRaises(RuntimeError) as caught:
            functions["main"]()
        self.assertIs(caught.exception, cleanup_error)
        self.assertEqual(calls, ["environment close", ("quit", 1), "application close"])

    def test_normal_entry_environment_close_failure_still_finishes_application(self):
        source = getattr(self, "_entry_source", Path(__file__).resolve().parents[2] / "src/eval_client/main.py")
        functions = {node.name: node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.FunctionDef)}
        calls = []
        cleanup_error = RuntimeError("final environment close failed")

        def close_environment():
            calls.append("environment close")
            raise cleanup_error

        env = SimpleNamespace(success_nums=0, fail_nums=0, close=close_environment)
        app = SimpleNamespace(
            app=SimpleNamespace(post_quit=lambda code: calls.append(("quit", code))),
            close=lambda: calls.append("application close"),
        )
        namespace = {
            "simulation_app": app, "args_cli": SimpleNamespace(), "enable_monitor": False,
            "os": SimpleNamespace(environ={"ROBODOJO_RUN_ID": "cpu"}),
            "sys": SimpleNamespace(stdout=io.StringIO(), stderr=io.StringIO()),
            "build_eval_config": lambda *_args, **_kwargs: (SimpleNamespace(eval_cfg={}), 0),
            "_load_resume_manifest": lambda *_: None,
            "create_eval_env": lambda *_args, **_kwargs: env,
            "_delete_resume_manifest": lambda _: None,
            "_close_model_client": lambda _: calls.append("client close"),
        }
        names = ("_finish_evaluation", "_run_evaluation", "main") if "_run_evaluation" in functions else ("main",)
        for name in names:
            exec(compile(ast.Module(body=[functions[name]], type_ignores=[]), str(source), "exec"), namespace)
        with self.assertRaises(RuntimeError) as caught:
            namespace["main"]()
        self.assertIs(caught.exception, cleanup_error)
        self.assertEqual(calls, ["client close", "environment close", ("quit", 1), "application close"])

    def test_factory_failure_before_environment_return_still_closes_application(self):
        calls = []
        original = RuntimeError("factory failed before returning environment")

        def run_evaluation(lifecycle, close_application):
            raise original

        app = SimpleNamespace(app=SimpleNamespace(post_quit=lambda code: calls.append(("quit", code))),
                              close=lambda: calls.append("application close"))
        functions = self._entry_functions(simulation_app=app, run_evaluation=run_evaluation)
        with self.assertRaises(RuntimeError) as caught:
            functions["main"]()
        self.assertIs(caught.exception, original)
        self.assertEqual(calls, [("quit", 1), "application close"])

    def _assert_error_visible_before_application_close(self, *, factory_fails):
        observed_at_close = []
        original = RuntimeError("factory failure") if factory_fails else OSError("final environment cleanup failure")

        class Stderr(io.StringIO):
            visible = ""

            def flush(self):
                self.visible = self.getvalue()
                super().flush()

        def close_environment():
            raise original

        def run_evaluation(lifecycle, close_application):
            if factory_fails:
                raise original
            lifecycle["env"] = SimpleNamespace(close=close_environment)

        stderr = Stderr()
        app = SimpleNamespace(
            app=SimpleNamespace(post_quit=lambda code: None),
            close=lambda: observed_at_close.append(stderr.visible),
        )
        functions = self._entry_functions(simulation_app=app, run_evaluation=run_evaluation)
        functions["sys"].stderr = stderr
        with self.assertRaises(type(original)) as caught:
            functions["main"]()
        self.assertIs(caught.exception, original)
        self.assertEqual(len(observed_at_close), 1)
        self.assertIn("Traceback (most recent call last)", observed_at_close[0])
        self.assertIn(f"{type(original).__name__}: {original}", observed_at_close[0])

    def test_factory_failure_traceback_is_flushed_before_application_close(self):
        self._assert_error_visible_before_application_close(factory_fails=True)

    def test_final_cleanup_failure_traceback_is_flushed_before_application_close(self):
        self._assert_error_visible_before_application_close(factory_fails=False)

    def test_restart_exit_status_retained_without_closing_application_twice(self):
        calls = []
        env = SimpleNamespace(close=lambda: calls.append("unexpected environment close"))
        app = SimpleNamespace(app=SimpleNamespace(post_quit=lambda code: calls.append(("unexpected quit", code))),
                              close=lambda: calls.append("application close"))

        def run_evaluation(lifecycle, close_application):
            lifecycle["env"] = env
            close_application()
            close_application()
            raise SystemExit(99)

        functions = self._entry_functions(simulation_app=app, run_evaluation=run_evaluation)
        with self.assertRaises(SystemExit) as caught:
            functions["main"]()
        self.assertEqual(caught.exception.code, 99)
        self.assertEqual(calls, ["application close"])

    def test_actual_restart_cap_retains_exit99_and_application_close_once(self):
        calls = []
        env = SimpleNamespace(
            close=lambda: calls.append("unexpected environment close"),
            persist_resume_manifest=lambda **values: calls.append(("persist", values["restart_count"])),
        )
        app = SimpleNamespace(
            app=SimpleNamespace(post_quit=lambda code: calls.append(("unexpected quit", code))),
            close=lambda: calls.append("application close"),
        )

        def run_evaluation(lifecycle, close_application):
            lifecycle["env"] = env
            functions["_restart_or_exit"](env, close_application, "injected fatal failure")

        def exit_with_status(code):
            raise SystemExit(code)

        functions = self._entry_functions(simulation_app=app, run_evaluation=run_evaluation)
        functions["MAX_INPROC_RESTARTS"] = 3
        functions["os"] = SimpleNamespace(environ={"ROBODOJO_FATAL_RESTART_COUNT": "3"})
        functions["sys"].exit = exit_with_status
        with self.assertRaises(SystemExit) as caught:
            functions["main"]()
        self.assertEqual(caught.exception.code, 99)
        self.assertEqual(calls, [("persist", 4), "application close"])


if __name__ == "__main__":
    unittest.main()
