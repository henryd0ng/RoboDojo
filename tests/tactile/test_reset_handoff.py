"""Check real reset methods using CPU objects, without importing Isaac/Kit."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def source_class(relative, name, **namespace):
    # Compile the actual class, leaving simulator-heavy imports/constructors
    # out of this CPU fixture. Methods and their super() relationships are real.
    tree = ast.parse((ROOT / relative).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / relative), 'exec'), namespace)
    return namespace[name]


class NativeBase:
    def reset(self, **_):
        self.events.append(('native_reset', self.tick))


Task = source_class('env/environment/task_env.py', 'TaskEnv', BaseEnv=NativeBase)
Evaluation = source_class('src/eval_client/eval_env.py', 'EvalEnv',
                          task_class=Task, UnStableError=RuntimeError)
Direct = source_class('env/environment/isaac/direct_rl_env.py', 'CustomDirectRLEnv',
                      DirectRLEnv=object, time=time)
Observation = source_class('env/observation_manager/obs_manager.py', 'ObsManager', deepcopy=deepcopy)


class Manager:
    def __init__(self, env, *, enabled=True, failure=None, stabilise_steps=10):
        self.env, self.enabled, self.failure = env, enabled, failure
        self.initialized = self.suspended = False
        self.config = SimpleNamespace(stabilise_steps=stabilise_steps)
        self.uipc_steps = self.captures = 0

    def begin_reset(self):
        self.initialized = False
        self.suspended = True
        self.env.events.append(('begin_reset', self.env.tick))

    def initialize(self, *_):
        self.env.events.append(('initialize', self.env.tick))
        if self.failure == 'initialize':
            raise RuntimeError('initialization rejected')
        self.initialized = True
        self.env.phase = 'warmup'

    def resume(self):
        self.suspended = False
        self.env.events.append(('resume', self.env.tick))

    def ensure_initialized(self):
        assert self.initialized and not self.suspended

    def physics_step(self, *_args, **_kwargs):
        assert self.initialized and not self.suspended
        if self.failure == 'warmup':
            raise RuntimeError('warmup rejected')
        self.uipc_steps += 1

    def before_render(self):
        pass

    def capture(self, **_):
        if not self.initialized or self.suspended:
            return
        self.captures += 1

    def observe(self, env_ids):
        if not self.initialized:
            return {int(i): {} for i in env_ids}
        return {int(i): {'left': {'rgb': 'frame', 'valid': True}} for i in env_ids}

    def calibrate_reference(self, render):
        assert self.initialized and not self.suspended
        self.env.events.append(('calibrate', self.env.tick))
        if self.failure == 'calibrate':
            raise RuntimeError('calibration rejected')
        render()


def fixture(cls=Evaluation, *, enabled=True, failure=None, stabilise_steps=10):
    env = object.__new__(cls)
    env.events, env.ticks, env.warmup_observations = [], [], []
    env.tick, env.phase = 0, 'task'
    env.num_envs = 1
    env._episode_namespace, env._episode_number, env.episode_ids = 'fixture', 0, []
    manager = env.tactile_manager = Manager(env, enabled=enabled, failure=failure, stabilise_steps=stabilise_steps)

    def tick(**_):
        owner = 'uipc' if manager.enabled and manager.initialized and not manager.suspended else 'native'
        env.tick += 1
        env.ticks.append((env.phase, owner))
        if owner == 'uipc':
            manager.physics_step(.004)

    env.sim_step = tick
    env.sim = SimpleNamespace(_sim_step_counter=0, physics_dt=.004)
    env.robot_manager = SimpleNamespace(
        reset=lambda: env.events.append(('robot_reset', env.tick)),
        set_origin_endpose=lambda: env.events.append(('robot_baseline', env.tick)),
        set_robot_init_state=lambda: env.events.append(('control_baseline', env.tick)))

    def saved_poses(env_idx_list):
        assert list(env_idx_list) == [0]
        previous = env.phase
        for phase in ('table', 'objects'):
            env.phase = phase
            for _ in range(20):
                env.sim_step(render=False)
        env.phase = previous
        env.events.append(('saved_poses_complete', env.tick))

    def stability(actual):
        assert actual is env
        env.phase = 'layout'
        for _ in range(300):
            env.sim_step(render=False)
        env.phase = 'capture'
        env.events.append(('layout_complete', env.tick))
        return True, []

    env.scene_manager = SimpleNamespace(
        reload_scene=lambda: env.events.append(('reload', env.tick)),
        apply_saved_poses=saved_poses,
        layout_manager=SimpleNamespace(set_saved_layout=lambda *_: None, check_layout_stability=stability))
    env.camera_manager = SimpleNamespace(reset=lambda: env.events.append(('camera_reset', env.tick)))
    env.capture_manager = SimpleNamespace(reset=lambda: env.events.append(('capture_reset', env.tick)))
    env.render = lambda: None
    obs = env.obs_manager = object.__new__(Observation)
    obs.num_envs, obs.dt, obs.collect_freq, obs.instruction = 1, .004, 25, ['fixture']
    obs.robot_manager = obs.camera_manager = obs.capture_manager = None
    obs.tactile_manager = manager
    obs.desc_manager = SimpleNamespace(reset=lambda: None, get_one_description=lambda: ['fixture'])
    obs.render_for_capture = lambda: None
    get_obs = obs.get_obs

    def recorded_obs(*args, **kwargs):
        value = get_obs(*args, **kwargs)
        env.warmup_observations.append(value)
        return value

    obs.get_obs = recorded_obs
    env._abort_video_writers = lambda: None
    env.seed_manager = SimpleNamespace(get_seed_scene_info=lambda _: {})
    env._align_layout_success = lambda: None
    env.physx_monitor_enabled = False
    env.unstable_nums = 0
    env.reward_manager = SimpleNamespace(init_state=lambda: env.events.append(('reward_baseline', env.tick)))
    env.model_client = SimpleNamespace(call=lambda **_: env.events.append(('policy_reset', env.tick)))
    return env


class ResetHandoffTests(unittest.TestCase):
    def simulator_imports(self, env):
        return patch.dict('sys.modules', {
            'env.tactile.dynamics': SimpleNamespace(
                restore_robot_state=lambda *_: env.events.append(('robot_restore', env.tick))),
            'omni.physx': SimpleNamespace(acquire_physx_interface=lambda: SimpleNamespace(
                update_transformations=lambda *_: env.events.append(('publish', env.tick)))),
            'env.camera_manager.capture.render_sync': SimpleNamespace(wait_for_latest_cameras=lambda _: None),
        })

    def test_eval_preserves_full_native_reset_before_handoff_and_policy(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                env = fixture(enabled=enabled)
                with self.simulator_imports(env):
                    env.reset(seed=[0])
                expected_ticks = ([('task', 'native')] * 300
                    + [('table', 'native')] * 20 + [('objects', 'native')] * 20
                    + [('layout', 'native')] * 300 + [('capture', 'native')] * 200)
                self.assertTrue(env.ticks[:840] == expected_ticks, 'Native reset phase order/ownership changed')
                self.assertEqual(len(env.warmup_observations), 40)
                self.assertTrue(all('tactile' not in obs[0] for obs in env.warmup_observations))
                self.assertEqual(env.episode_ids, ['fixture:1:0'])
                final = 850 if enabled else 840
                self.assertEqual(env.tick, final)
                self.assertEqual(env.events[-4:], [('robot_baseline', final), ('control_baseline', final),
                                                  ('reward_baseline', final), ('policy_reset', final)])
                if enabled:
                    self.assertIn(('robot_restore', 0), env.events)
                    self.assertNotIn(('resume', 0), env.events)
                    self.assertIn(('resume', 840), env.events)
                    self.assertEqual(env.ticks[840:], [('warmup', 'uipc')] * 10)
                    self.assertLess(env.events.index(('publish', 840)), env.events.index(('initialize', 840)))
                    self.assertLess(env.events.index(('initialize', 840)), env.events.index(('calibrate', 850)))
                    self.assertEqual(env.tactile_manager.captures, 0)
                    self.assertIn('tactile', env.obs_manager.get_obs()[0])
                else:
                    self.assertFalse(any(e[0] in ('robot_restore', 'initialize', 'calibrate') for e in env.events))

    def test_large_configured_warmup_is_consumed_once_after_native_reset(self):
        for steps in (301, 302):
            with self.subTest(stabilise_steps=steps):
                env = fixture(stabilise_steps=steps)
                with self.simulator_imports(env):
                    env.reset(seed=[0])
                self.assertEqual(sum(owner == 'native' for _, owner in env.ticks), 840)
                self.assertEqual(env.tactile_manager.uipc_steps, steps)
                self.assertEqual(env.events[-1], ('policy_reset', 840 + steps))
                self.assertNotIn(('resume', 0), env.events)
                self.assertIn(('resume', 840), env.events)
                standalone = fixture(Task, stabilise_steps=steps)
                with self.simulator_imports(standalone):
                    standalone.reset(seed=[0])
                self.assertEqual(standalone.tick, 40 + steps)
                self.assertEqual(standalone.tactile_manager.uipc_steps, steps)

    def test_failed_init_warmup_or_calibration_cannot_reset_policy(self):
        for failure in ('initialize', 'warmup', 'calibrate'):
            with self.subTest(failure=failure):
                env = fixture(failure=failure)
                with self.simulator_imports(env), self.assertRaises(RuntimeError):
                    env.reset(seed=[0])
                self.assertEqual(sum(owner == 'native' for _, owner in env.ticks), 840)
                self.assertFalse(any(e[0].endswith('baseline') or e[0] == 'policy_reset' for e in env.events))

    def test_standalone_keeps_early_handoff_and_existing_settle(self):
        env = fixture(Task)
        with self.simulator_imports(env):
            env.reset(seed=[0])
        self.assertEqual(env.ticks[:40], [('table', 'native')] * 20 + [('objects', 'native')] * 20)
        self.assertEqual(len(env.ticks), 340)
        self.assertTrue(all(owner == 'uipc' for _, owner in env.ticks[40:]))
        self.assertIn(('robot_restore', 0), env.events)
        self.assertIn(('initialize', 40), env.events)
        self.assertIn(('calibrate', 340), env.events)

    def test_direct_step_order_for_pending_disabled_active_and_suspended(self):
        for name, enabled, initialized, suspended in (
            ('pending', True, False, False), ('disabled', False, False, False),
            ('active', True, True, False), ('suspended', True, True, True),
        ):
            with self.subTest(state=name):
                events = []
                active = enabled and initialized and not suspended
                manager = SimpleNamespace(enabled=enabled, initialized=initialized, suspended=suspended,
                    ensure_initialized=lambda: events.append('ensure') if enabled and initialized else None,
                    physics_step=lambda *_, **__: events.append('uipc') if active else None,
                    before_render=lambda: events.append('before_render') if active else None)
                driver = object.__new__(Direct)
                driver.tactile_manager = manager
                driver.sim = SimpleNamespace(has_gui=lambda: False, has_rtx_sensors=lambda: True,
                    step=lambda **_: events.append('physics'), render=lambda: events.append('render'))
                driver.scene = SimpleNamespace(write_data_to_sim=lambda: events.append('write'),
                    update=lambda **_: events.append('update'))
                driver.cfg = SimpleNamespace(decimation=1, sim=SimpleNamespace(render_interval=1), events=None)
                driver._sim_step_counter = driver.episode_length_buf = driver.common_step_counter = 0
                driver.physics_dt, driver.extras = .004, {}
                driver.sim_step()
                expected = (['ensure', 'write', 'physics', 'update', 'uipc', 'before_render', 'render']
                    if name == 'active' else ['write', 'physics', 'render', 'update'])
                self.assertEqual(events, expected)

    def test_pending_disabled_and_suspended_observations_have_no_tactile(self):
        for enabled, initialized, suspended in ((True, False, False), (False, False, False), (True, True, True)):
            with self.subTest(state=(enabled, initialized, suspended)):
                env = fixture(enabled=enabled)
                env.tactile_manager.initialized = initialized
                env.tactile_manager.suspended = suspended
                self.assertNotIn('tactile', env.obs_manager.get_obs()[0])
                self.assertEqual(env.tactile_manager.captures, 0)


if __name__ == '__main__':
    unittest.main()
