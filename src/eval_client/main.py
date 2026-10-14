import argparse
from datetime import datetime
import importlib
import json
import os
import sys

from isaaclab.app import AppLauncher

MAX_INPROC_RESTARTS = 3

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--env_cfg_type",
    type=str,
    required=True,
    help="config file name for evaluation",
)
parser.add_argument("--device_id", type=int, required=True, help="the device id for current process")
parser.add_argument(
    "--policy_name",
    type=str,
    required=True,
    help="XPolicyLab module name for deployment",
)
parser.add_argument("--port", type=int, required=True, help="the port for the policy WebSocket server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="IP address or hostname of the policy server. Defaults to localhost.",
)
parser.add_argument(
    "--protocol",
    choices=("ws",),
    default="ws",
    help=(
        "Env-to-policy transport. 'ws' is the default WebSocket protocol "
        "(msgpack frames over ws://host:port); also set as protocol: ws in deploy.yml."
    ),
)
parser.add_argument(
    "--policy_server_url",
    type=str,
    default="",
    help=(
        "Full WebSocket URL for the policy server (e.g. ws://127.0.0.1:9999). "
        "Built from --host and --port when omitted."
    ),
)
parser.add_argument(
    "--additional_info",
    type=str,
    required=True,
    help="additional information for the evaluation",
)


parser.add_argument("--seed", type=int, required=True, help="policy seed for eval")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Safe to import before AppLauncher: env is a namespace package (no __init__)
# and GLOBAL_CONFIGS only imports os, so this pulls in no app-dependent code.
from env.global_configs import BENCHMARK, ROOT_DIR
from src.eval_client.config import build_eval_config, eval_run_overrides

eval_run_overrides()

task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")


class PhysXBrokenError(Exception):
    pass


class PhysXFatalError(Exception):
    pass


def get_monitor():
    return None


def _physx_monitor_needed(task_name) -> bool:
    """Enable the PhysX log monitor only for tasks whose Config declares a
    non-empty `Articulation` section (those bodies trigger the PhysX
    "Invalid PhysX transform" / CUDA failures we recover from). Read with a
    lightweight yaml load before AppLauncher; any failure falls back to
    enabled (fail-safe).
    """
    cfg_path = task_registry.task_config_path(os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), task_name)
    try:
        import yaml

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return bool(cfg.get("Articulation"))
    except Exception:
        return True


# Fix the run id for this entire eval invocation (across all in-process
# os.execv self-restarts and bash-level retries). When eval_policy.sh
# launches us it exports ROBODOJO_RUN_ID up-front; if a developer runs
# main.py directly we generate one here and propagate it via the
# environment so subsequent execv calls see the same value.
if not os.environ.get("ROBODOJO_RUN_ID"):
    os.environ["ROBODOJO_RUN_ID"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

enable_monitor = _physx_monitor_needed(args_cli.task_name)
print(f"[main] PhysX monitor enabled={enable_monitor} (task={args_cli.task_name})")
if enable_monitor:
    # Start before AppLauncher so Kit inherits the redirected stdout/stderr fds.
    from src.eval_client.physx_warning_monitor import (
        PhysXBrokenError,
        PhysXFatalError,
        get_monitor,
    )

    get_monitor().start(enabled=True)

# Camera frames must not lag physics: make each Kit update wait for its frame.
from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

add_zero_delay_kit_args(args_cli)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from env.global_configs import *
from src.eval_client.eval_env import create_eval_env
from utils.cluttered_generator import UnStableError

def _resume_manifest_path(eval_cfg, run_id):
    """Mirror of EvalEnv.resume_manifest_path() so we can load BEFORE
    constructing the env. Keeping the layout aligned avoids drift between
    the writer and reader paths.
    """
    return os.path.join(
        eval_cfg.get("output_root", "eval_result"),
        BENCHMARK,
        eval_cfg["task_name"],
        eval_cfg["policy_name"],
        eval_cfg["config_name"],
        f"{eval_cfg.get('seed', 0)}_{eval_cfg.get('additional_info', '')}",
        f"_resume_{run_id}.json",
    )


def _load_resume_manifest(eval_cfg, run_id):
    """Return parsed manifest dict, or None if no resume is in progress."""
    path = _resume_manifest_path(eval_cfg, run_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as e:
        print(f"[main] failed to load resume manifest at {path}: {e}; ignoring.")
        return None
    print(
        f"[main] resuming from manifest {path} "
        f"(success={data.get('success_nums')} fail={data.get('fail_nums')} "
        f"completed={len(data.get('completed_layout_ids') or [])} "
        f"abandoned={len(data.get('abandoned_layout_ids') or [])} "
        f"restart_count={data.get('restart_count', 0)})"
    )
    return data


def _delete_resume_manifest(env):
    """Best-effort cleanup at normal completion. Failure is non-fatal."""
    try:
        path = env.resume_manifest_path()
    except Exception:
        return
    try:
        if os.path.exists(path):
            os.unlink(path)
            print(f"[main] removed resume manifest {path} (eval completed)")
    except Exception as e:
        print(f"[main] failed to unlink resume manifest {path}: {e}")


def _close_model_client(env):
    """Best-effort graceful close for policy communication."""
    try:
        model_client = getattr(env, "model_client", None)
        close = getattr(model_client, "close", None)
        if callable(close):
            close()
    except Exception as e:
        print(f"[main] failed to close model client: {e}")


def _finish_evaluation(env, simulation_app, close_application, original_error=None):
    """Close the application even when environment cleanup raises."""
    errors = [] if original_error is None else [original_error]
    if original_error is not None:
        # Kit may exit inside close(), so establish failure status first.
        try:
            simulation_app.app.post_quit(1)
        except BaseException as error:
            errors.append(error)
    if env is not None:
        try:
            _close_model_client(env)
        except BaseException as error:
            errors.append(error)
        try:
            env.close()
        except BaseException as error:
            errors.append(error)
    if errors and original_error is None:
        # A normal run whose final cleanup failed must also exit unsuccessfully.
        try:
            simulation_app.app.post_quit(1)
        except BaseException as error:
            errors.append(error)
    # Kit may terminate the process inside App.close(). Emit the failures while
    # stderr is still available, rather than relying on a later Python raise.
    if errors:
        import traceback

        for error in tuple(errors):
            try:
                traceback.print_exception(error, file=sys.stderr)
            except BaseException as reporting_error:
                errors.append(reporting_error)
    for operation in (sys.stdout.flush, sys.stderr.flush, close_application):
        try:
            operation()
        except BaseException as error:
            errors.append(error)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("Evaluation and shutdown failed", errors) from None


def _restart_or_exit(env, close_application, fatal_msg):
    """Persist progress and either os.execv-restart or sys.exit(99).

    Bounded by ROBODOJO_FATAL_RESTART_COUNT env var so a persistent hardware
    failure cannot put us in an infinite loop. The bash retry loop in
    eval_policy.sh provides a second layer of bounded restarts.
    """
    restart_count = int(os.environ.get("ROBODOJO_FATAL_RESTART_COUNT", "0")) + 1
    try:
        env.persist_resume_manifest(restart_count=restart_count)
    except Exception as e:
        print(f"[FATAL] persist_resume_manifest failed: {e}")
    print(
        f"[FATAL] PhysX kernel failure detected: {fatal_msg}; persisted manifest. "
        f"In-process restart attempt {restart_count}/{MAX_INPROC_RESTARTS}."
    )
    try:
        close_application()
    except Exception:
        pass
    if restart_count <= MAX_INPROC_RESTARTS:
        os.environ["ROBODOJO_FATAL_RESTART_COUNT"] = str(restart_count)
        print(f"[FATAL] os.execv self-restart with run_id={os.environ.get('ROBODOJO_RUN_ID')}")
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable] + sys.argv)
    print(f"[FATAL] in-process restart cap reached ({MAX_INPROC_RESTARTS}); exiting with rc=99 for bash-level retry.")
    sys.exit(99)


def _exit_for_shell_restart(env, fatal_msg):
    """Persist progress, then let eval_policy.sh restart a fresh process."""
    restart_count = int(os.environ.get("ROBODOJO_FATAL_RESTART_COUNT", "0"))
    try:
        env.persist_resume_manifest(restart_count=restart_count)
    except Exception as e:
        print(f"[FATAL] persist_resume_manifest failed: {e}")
    print(f"[FATAL] PhysX requested shell-level restart: {fatal_msg}; exiting with rc=99 for bash-level retry.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(99)


def _run_evaluation(lifecycle, close_application):
    """Assemble the env config, build the eval env, and run the eval loop with
    PhysX crash/resume recovery until the requested episode count is reached.
    """
    env_cfg, eval_num = build_eval_config(args_cli, physx_monitor_enabled=enable_monitor)
    eval_cfg = env_cfg.eval_cfg
    run_id = os.environ["ROBODOJO_RUN_ID"]
    resume_state = _load_resume_manifest(eval_cfg, run_id)
    env = create_eval_env(env_cfg, simulation_app, resume_state=resume_state)
    lifecycle["env"] = env
    eval_time = env.success_nums + env.fail_nums
    if eval_time >= eval_num:
        # Already complete on resume - nothing left to do.
        env.env_seeds = None
    else:
        env.env_seeds = env.seed_manager.get_seeds(max_count=eval_num - eval_time)
    while env.env_seeds is not None:
        retry_round = False
        if enable_monitor:
            get_monitor().reset()
        bad_envs = None
        try:
            env.reset(seed=env.env_seeds)
            env.run_eval()
            env.seed_manager.eval_step()

        except PhysXFatalError as e:
            # Unrecoverable: GPU/CUDA context is dead. Persist progress
            # and re-exec (or sys.exit(99) for bash to restart).
            if not enable_monitor:
                raise
            if get_monitor().requires_shell_restart():
                _exit_for_shell_restart(env, str(e))
            _restart_or_exit(env, close_application, str(e))

        except PhysXBrokenError as e:
            # Monitor caught the warning in time.
            bad_envs = sorted(e.broken_envs)

        except UnStableError:
            env.seed_manager.eval_step()
        except Exception as e:
            import traceback

            print(
                f"[Eval] unhandled exception during reset/run_eval: {type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            if env_cfg.tactile.get("enabled", False):
                # A broken contact world is an infrastructure error, not an
                # invalid benchmark seed. Do not silently exhaust all layouts.
                # The outer owner establishes failure status before cleanup.
                raise
            if enable_monitor and get_monitor().is_fatal():
                fatal_msg = get_monitor().get_fatal_message() or str(e)
                if get_monitor().requires_shell_restart():
                    _exit_for_shell_restart(env, fatal_msg)
                _restart_or_exit(env, close_application, fatal_msg)
            bad = {i for i in get_monitor().get_broken_envs() if i < env.num_envs} if enable_monitor else set()
            if bad:
                print(
                    f"[PhysX] downstream exception {type(e).__name__} with "
                    f"monitor broken_envs={sorted(bad)}; treating as PhysX break."
                )
                bad_envs = sorted(bad)
            else:
                env.seed_manager.eval_step()

        if bad_envs is not None:
            # Abandon broken-env seeds, refill from the seed queue, and
            # retry this round iff there is at least one real seed left.
            bad_seeds = env.get_seeds_for_envs(bad_envs)
            env.abandoned_seeds.update(bad_seeds)

            replacements = env.seed_manager.get_seeds(max_count=len(bad_envs)) or []
            bad_env_set = set(bad_envs)
            new_batch = [None] * env.num_envs
            for env_idx, seed in env.current_env_seed_map.items():
                if env_idx not in bad_env_set:
                    new_batch[env_idx] = seed
            for k, env_idx in enumerate(bad_envs):
                if k < len(replacements):
                    new_batch[env_idx] = replacements[k]
            env.env_seeds = new_batch

            real_remaining = sum(1 for s in env.env_seeds if s is not None)
            print(
                f"[PhysX] broken envs={bad_envs} -> abandon seeds={sorted(bad_seeds)}; "
                f"refill from queue={replacements}; new batch={env.env_seeds}; "
                f"real_remaining={real_remaining}"
            )
            env.close()
            if real_remaining == 0:
                print("[PhysX] no real seeds remaining in this batch, advancing.")
                retry_round = False
            else:
                retry_round = True

        if retry_round:
            continue

        print(f"Success nums: {env.success_nums}, Fail nums: {env.fail_nums}, Unstable nums: {env.unstable_nums}")
        eval_time = env.success_nums + env.fail_nums
        if eval_time >= eval_num:
            break

        env.env_seeds = env.seed_manager.get_seeds(max_count=eval_num - eval_time)
        if env.env_seeds is None:
            print("No more seeds to run, exiting.")
            break

        env.close()

    _delete_resume_manifest(env)


def main():
    """Retain restart behavior and own final application shutdown once."""
    lifecycle = {"env": None}
    close_attempted = False

    def close_application():
        nonlocal close_attempted
        if not close_attempted:
            close_attempted = True
            simulation_app.close()

    error = None
    try:
        _run_evaluation(lifecycle, close_application)
    except BaseException as caught:
        error = caught
    if close_attempted:
        # A fatal restart already closed the application before execv/exit99.
        # Do not access its former native stage or replace that exit status.
        if error is not None:
            raise error
        return
    _finish_evaluation(lifecycle["env"], simulation_app, close_application, error)


if __name__ == "__main__":
    main()
