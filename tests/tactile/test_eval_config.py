"""Normal configuration semantics without importing the application entrypoint."""

import contextlib
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.eval_client.config import build_eval_config, eval_run_overrides
from utils.load_file import load_yaml


def arguments(**overrides):
    fields = dict(task_name="insert_tubes", env_cfg_type="arx_x5", num_envs=3,
                  device_id=2, policy_name="demo_policy", additional_info="configuration-test",
                  seed=42, host="127.0.0.1", port=9999, protocol="ws", policy_server_url="")
    return SimpleNamespace(**(fields | overrides))


class EvaluationConfigurationTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"ROBODOJO_RUN_ID": "configuration-test", "EVAL_NUM": "3"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_native_task_identity_and_episode_limits_for_both_presets(self):
        for task in ("insert_tubes", "plug_in_charger", "deposit_coin"):
            for preset in ("arx_x5", "arx_x5_pad_tactile"):
                with self.subTest(task=task, preset=preset):
                    config, episodes = build_eval_config(arguments(task_name=task, env_cfg_type=preset))
                    self.assertEqual(config.eval_cfg.task_name, task)
                    self.assertEqual(config.eval_cfg.config_name, "arx_x5")
                    self.assertEqual(config.eval_cfg.eval_num, episodes)
                    self.assertEqual(episodes, 3)
                    self.assertEqual(config.eval_cfg.num_envs, 1)
                    self.assertEqual(config.sim.scene.num_envs, 1)
                    self.assertEqual(list(config.sim.seed), [0])
                    self.assertEqual(config.tactile.enabled, preset.endswith("_tactile"))
                    self.assertEqual([robot.robot_name for robot in config.robot.robots],
                                     ["x5_pad", "x5"] if preset.endswith("_tactile") else ["x5", "x5"])
                    self.assertEqual(config.camera.default_frequency, config.eval_cfg.observation.collect_freq)

    def test_deployment_metadata_keeps_cli_overrides_and_excludes_credentials(self):
        def load(path):
            if str(path).endswith("deploy.yml"):
                return {"eval_batch": True, "request_timeout_s": 240, "api_key": "must-not-copy",
                        "host": "ignored-policy-host", "port": 123}
            return load_yaml(path)

        with patch("src.eval_client.config.load_yaml", side_effect=load):
            config, _ = build_eval_config(arguments(policy_server_url="ws://explicit:4444"), physx_monitor_enabled=True)
        self.assertEqual(config.eval_cfg.device_id, 2)
        self.assertEqual(config.eval_cfg.seed, 42)
        self.assertEqual(config.eval_cfg.num_envs, 3)
        self.assertEqual(list(config.sim.seed), [0, 0, 0])
        self.assertTrue(config.eval_cfg.physx_monitor_enabled)
        self.assertNotIn("request_timeout_s", config.deploy_cfg)
        self.assertEqual(config.deploy_cfg.host, "127.0.0.1")
        self.assertEqual(config.deploy_cfg.port, 9999)
        self.assertEqual(config.deploy_cfg.policy_server_url, "ws://explicit:4444")
        self.assertEqual(config.deploy_cfg.evaluation_id, "configuration-test")
        self.assertNotIn("api_key", config.deploy_cfg)

    def test_batch_policy_uses_one_tactile_environment_but_stock_keeps_requested_count(self):
        def load(path):
            if str(path).endswith("deploy.yml"):
                return {"eval_batch": True}
            return load_yaml(path)

        with patch("src.eval_client.config.load_yaml", side_effect=load):
            for task in ("insert_tubes", "plug_in_charger", "deposit_coin"):
                for preset, expected_count in (("arx_x5_pad_tactile", 1), ("arx_x5", 10)):
                    with self.subTest(task=task, preset=preset), contextlib.redirect_stdout(io.StringIO()) as output:
                        config, episodes = build_eval_config(arguments(
                            task_name=task, env_cfg_type=preset, num_envs=10))
                        self.assertTrue(config.eval_cfg.eval_batch)
                        self.assertEqual(config.sim.scene.num_envs, expected_count)
                        self.assertEqual(config.eval_cfg.num_envs, expected_count)
                        self.assertEqual(list(config.sim.seed), [0] * expected_count)
                        self.assertEqual(episodes, 3)
                        self.assertEqual(config.eval_cfg.config_name, "arx_x5")
                        if preset.endswith("_tactile"):
                            self.assertIn("tactile.enabled=true currently supports one", output.getvalue())
                            self.assertIn("10 -> 1", output.getvalue())
                        else:
                            self.assertNotIn("tactile.enabled", output.getvalue())

    def test_native_episode_count_and_numeric_cap(self):
        with patch.dict(os.environ, {"EVAL_NUM": "NaTiVe"}):
            native, count = build_eval_config(arguments(num_envs=1))
        with patch.dict(os.environ, {"EVAL_NUM": str(count + 100)}):
            capped, episodes = build_eval_config(arguments(num_envs=1))
        self.assertEqual(episodes, count)
        self.assertEqual(capped.eval_cfg.eval_num, native.eval_cfg.eval_num)

    def test_optional_output_and_action_overrides_route_tactile_scratch(self):
        with patch.dict(os.environ, {"ROBODOJO_EVAL_OUTPUT_ROOT": "~/robodojo-test-output",
                                     "ROBODOJO_EVAL_MAX_ACTIONS": "220"}):
            config, _ = build_eval_config(arguments(env_cfg_type="arx_x5_pad_tactile"))
        root = os.path.abspath(os.path.expanduser("~/robodojo-test-output"))
        self.assertEqual(config.eval_cfg.output_root, root)
        self.assertEqual(config.eval_cfg.max_actions, 220)
        self.assertEqual(config.tactile.uipc_workspace, os.path.join(root, "_tactile_workspace", "configuration-test"))
        self.assertEqual(config.tactile.mesh_cache, os.path.join(root, "_cache", "native_collision"))

    def test_output_override_preserves_explicit_tactile_scratch_paths(self):
        def load(path):
            data = load_yaml(path)
            if str(path).endswith("arx_x5_pad_tactile.yml"):
                data["tactile"].update(uipc_workspace="/tmp/explicit-workspace", mesh_cache="/tmp/explicit-cache")
            return data

        with patch.dict(os.environ, {"ROBODOJO_EVAL_OUTPUT_ROOT": "/tmp/robodojo-test-output"}), \
                patch("src.eval_client.config.load_yaml", side_effect=load):
            config, _ = build_eval_config(arguments(env_cfg_type="arx_x5_pad_tactile"))
        self.assertEqual(config.tactile.uipc_workspace, "/tmp/explicit-workspace")
        self.assertEqual(config.tactile.mesh_cache, "/tmp/explicit-cache")

    def test_invalid_run_overrides_fail_before_simulation(self):
        for value in ("", "0", "-1", "1.5", "abc"):
            with self.subTest(value=value), patch.dict(os.environ, {"ROBODOJO_EVAL_MAX_ACTIONS": value}):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    eval_run_overrides()
        with patch.dict(os.environ, {"ROBODOJO_EVAL_OUTPUT_ROOT": ""}):
            with self.assertRaisesRegex(ValueError, "nonempty path"):
                eval_run_overrides()


if __name__ == "__main__":
    unittest.main()
