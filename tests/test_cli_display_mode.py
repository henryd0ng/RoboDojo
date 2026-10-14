"""CPU shell-entry guards; no simulator, policy server, or network is started."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeCliDisplayModeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="robodojo-cli-display-")
        self.addCleanup(temporary.cleanup)
        self.checkout = Path(temporary.name) / "checkout"
        scripts = self.checkout / "scripts"
        (scripts / "internal").mkdir(parents=True)
        for name in ("robodojo.sh", "eval_policy.sh"):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        self.capture = self.checkout / "main-call.json"
        self.write("env_cfg/arx_x5_pad_tactile.yml", "config:\n  sim: sim_config\n")
        self.write("env_cfg/sim/sim_config.yml", "render_interval: 1\nscene:\n  num_envs: 1\n")
        self.write("XPolicyLab/policy/demo_policy/deploy.py", "# Never imported by these shell tests.\n")
        self.write("XPolicyLab/policy/demo_policy/deploy.yml", "protocol: ws\n")
        self.write("XPolicyLab/policy/demo_policy/eval.sh", "#!/usr/bin/env bash\nexit 91\n")
        self.write("XPolicyLab/policy/demo_policy/setup_eval_policy_server.sh", "exit 92\n")

        # Substitute only eval's server/conda orchestration. The public CLI and
        # lower eval_policy.sh remain exact copies of the source under review.
        self.write("scripts/internal/run_policy_eval.sh", """#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
shift
dataset=$1 task=$2 ckpt=$3 config=$4 action=$5 seed=$6 device=$8
bash "$root/scripts/eval_policy.sh" \
  --root_dir "$root" --task_name "$task" --env_cfg_type "$config" \
  --device_id "$device" --policy_name demo_policy --port 19999 \
  --host 127.0.0.1 --protocol ws --seed "$seed" \
  --additional_info "ckpt_name=$ckpt,action_type=$action"
""")

        stub = f"#!{sys.executable}\n" + """import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if args and args[0] == '-c':
    query = args[1]
    if "['config']['sim']" in query:
        print('sim_config')
    elif 'render_interval' in query or "get('scene'" in query:
        print(1)
    elif "get('protocol'" in query:
        print('ws')
    else:
        raise SystemExit('Unexpected config query in CLI CPU guard')
elif len(args) >= 2 and args[0] == '-u' and args[1].endswith('src/eval_client/main.py'):
    Path(os.environ['CLI_CPU_CAPTURE']).write_text(json.dumps({
        'argv': args,
        'HEADLESS': os.environ.get('HEADLESS'),
        'ROBODOJO_HEADLESS': os.environ.get('ROBODOJO_HEADLESS'),
        'EVAL_NUM': os.environ.get('EVAL_NUM'),
    }))
else:
    raise SystemExit('Unexpected Python invocation in CLI CPU guard')
"""
        for name in ("python", "python3"):
            path = self.write(f"bin/{name}", stub)
            path.chmod(0o755)
        # probe_tcp invokes timeout; this prevents even a /dev/tcp connection.
        self.write("bin/timeout", "#!/usr/bin/env bash\nexit 0\n").chmod(0o755)
        self.environment = {
            **os.environ,
            "PATH": f"{self.checkout / 'bin'}:{os.environ.get('PATH', '')}",
            "CLI_CPU_CAPTURE": str(self.capture),
            "ROBODOJO_RUN_ID": "cpu-display-mode-test",
        }
        for name in ("HEADLESS", "ROBODOJO_HEADLESS", "EVAL_NUM"):
            self.environment.pop(name, None)

    def write(self, relative, text):
        path = self.checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def run_shell(self, relative, arguments, environment=None):
        self.capture.unlink(missing_ok=True)
        return subprocess.run(
            ["bash", str(self.checkout / relative), *arguments],
            cwd=self.checkout,
            env=self.environment | (environment or {}),
            capture_output=True,
            text=True,
            timeout=10,
        )

    def public_arguments(self, command, mode=()):
        common = ["--task", "insert_tubes", "--env-cfg", "arx_x5_pad_tactile",
                  "--action-type", "ee", "--seed", "0", "--eval-num", "1", "--ckpt", "cpu-test"]
        if command == "client":
            common += ["--policy-name", "demo_policy", "--policy-host", "127.0.0.1", "--policy-port", "19999"]
        else:
            common += ["--policy-dir", str(self.checkout / "XPolicyLab/policy/demo_policy"),
                       "--policy-env", "unused-cpu-fixture"]
        return [command, *common, *mode]

    def assert_main_mode(self, command, mode, expected):
        result = self.run_shell("scripts/robodojo.sh", self.public_arguments(command, mode))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        call = json.loads(self.capture.read_text())
        self.assertEqual(call["HEADLESS"], str(expected))
        self.assertEqual(call["argv"].count("--headless"), expected)
        self.assertEqual(call["EVAL_NUM"], "1")
        self.assertIn("--enable_cameras", call["argv"])
        kit_args = call["argv"][call["argv"].index("--kit_args") + 1]
        self.assertIn("isaacsim.replicator.behavior", kit_args)
        self.assertIn("isaacsim.sensors.camera", kit_args)
        if expected == 0:
            self.assertEqual(call["argv"][call["argv"].index("--livestream") + 1], "0")
        if command == "eval":
            self.assertEqual(call["ROBODOJO_HEADLESS"], str(expected))

    def test_client_default_is_headless(self):
        self.assert_main_mode("client", (), 1)

    def test_client_gui_reaches_main_without_headless(self):
        self.assert_main_mode("client", ("--gui",), 0)

    def test_client_explicit_headless_reaches_main(self):
        self.assert_main_mode("client", ("--headless",), 1)

    def test_eval_default_is_headless(self):
        self.assert_main_mode("eval", (), 1)

    def test_eval_gui_environment_reaches_main_without_headless(self):
        self.assert_main_mode("eval", ("--gui",), 0)

    def test_eval_explicit_headless_environment_reaches_main(self):
        self.assert_main_mode("eval", ("--headless",), 1)

    def test_lower_runner_rejects_unknown_mode(self):
        arguments = ["--root_dir", str(self.checkout), "--task_name", "insert_tubes",
                     "--env_cfg_type", "arx_x5_pad_tactile", "--device_id", "0",
                     "--policy_name", "demo_policy", "--port", "19999",
                     "--additional_info", "cpu-test", "--seed", "0"]
        result = self.run_shell("scripts/eval_policy.sh", arguments, {"ROBODOJO_HEADLESS": "invalid"})
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("ROBODOJO_HEADLESS", result.stdout + result.stderr)
        self.assertFalse(self.capture.exists())

    def test_public_commands_reject_malformed_mode(self):
        for command in ("client", "eval"):
            with self.subTest(command=command):
                result = self.run_shell("scripts/robodojo.sh", self.public_arguments(command, ("--gui=invalid",)))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("--gui=invalid", result.stdout + result.stderr)
                self.assertFalse(self.capture.exists())

    def test_batch_client_gui_is_rejected_before_dispatch(self):
        result = self.run_shell("scripts/robodojo.sh", [
            "client", "--policy-name", "demo_policy", "--policy-host", "127.0.0.1",
            "--policy-port", "19999", "--only", "insert_tubes", "--gui",
        ])
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--gui", result.stdout + result.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
