"""Build the normal evaluation configuration without launching Isaac or Kit."""

from __future__ import annotations

import importlib
import os

from omegaconf import DictConfig, OmegaConf

from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization, resolve_random_task_num_envs


def eval_run_overrides() -> dict:
    """Validate optional output routing and episode caps before launching Kit."""
    overrides = {}
    output_root = os.environ.get("ROBODOJO_EVAL_OUTPUT_ROOT")
    if output_root is not None:
        if not output_root.strip():
            raise ValueError("ROBODOJO_EVAL_OUTPUT_ROOT must be a nonempty path")
        overrides["output_root"] = os.path.abspath(os.path.expanduser(os.path.expandvars(output_root)))
    max_actions = os.environ.get("ROBODOJO_EVAL_MAX_ACTIONS")
    if max_actions is not None:
        if not max_actions.strip().isdigit() or int(max_actions) <= 0:
            raise ValueError("ROBODOJO_EVAL_MAX_ACTIONS must be a positive integer")
        overrides["max_actions"] = int(max_actions)
    return overrides


def build_eval_config(args, *, physx_monitor_enabled: bool = False) -> tuple[DictConfig, int]:
    """Use the same task, preset, policy and environment overrides as main.

    ``args`` has the ordinary evaluation CLI fields. ``ROBODOJO_RUN_ID`` is
    supplied by the entrypoint, and optional ``EVAL_NUM`` caps the task's
    episode count (``native`` leaves the task count unchanged). This function
    reads configuration files but does not initialize simulation or policies.
    """
    overrides = eval_run_overrides()
    task_name, num_envs = args.task_name, args.num_envs
    run_id = os.environ["ROBODOJO_RUN_ID"]
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, args.env_cfg_type + ".yml"))
    policy_path = os.path.join(ROOT_DIR, "XPolicyLab", "policy", args.policy_name, "deploy.yml")
    policy_cfg = load_yaml(policy_path) if os.path.isfile(policy_path) else {}
    eval_batch = bool(policy_cfg.get("eval_batch", False))
    eval_cfg.update(task_name=task_name, num_envs=num_envs, device_id=args.device_id,
                    eval_batch=eval_batch, policy_name=args.policy_name,
                    additional_info=args.additional_info, seed=args.seed,
                    physx_monitor_enabled=physx_monitor_enabled)
    eval_cfg.update(overrides)
    # Keep credentials and model-only deployment options out of observations,
    # saved environment configuration and policy transport metadata.
    deploy_cfg = {
        "policy_name": args.policy_name, "port": args.port, "host": args.host,
        "protocol": args.protocol,
        "policy_server_url": args.policy_server_url or f"ws://{args.host}:{args.port}",
        "evaluation_id": run_id, "trial_id": f"{task_name}-{run_id}",
        "action_case_id": f"{task_name}_case", "repeat_index": None,
    }
    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    env_cfg = OmegaConf.create({
        section: load_yaml(os.path.join(ENV_CONFIG_PATH, section, eval_cfg["config"][section] + ".yml"))
        for section in ("sim", "scene", "camera", "robot")
    } | {
        "tactile": eval_cfg.get("tactile", {"enabled": False}),
        "task_env": load_yaml(task_registry.task_config_path(os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), task_name)),
        "eval_cfg": eval_cfg, "deploy_cfg": deploy_cfg,
    })
    if "output_root" in overrides and env_cfg.tactile.get("enabled", False):
        root = overrides["output_root"]
        if not env_cfg.tactile.get("uipc_workspace"):
            env_cfg.tactile.uipc_workspace = os.path.join(root, "_tactile_workspace", run_id)
        if not env_cfg.tactile.get("mesh_cache"):
            env_cfg.tactile.mesh_cache = os.path.join(root, "_cache", "native_collision")
    capped_num_envs = resolve_random_task_num_envs(task_name, num_envs, env_cfg.sim)
    if capped_num_envs != num_envs:
        print(f"[main] Random task {task_name}: num_envs capped {num_envs} -> {capped_num_envs} ")
    num_envs = capped_num_envs
    if env_cfg.tactile.get("enabled", False) and num_envs != 1:
        print(f"[main] tactile.enabled=true currently supports one simulation environment; "
              f"capping num_envs {num_envs} -> 1 (policy batch capability is unchanged)")
        num_envs = 1
    if not eval_batch and num_envs != 1:
        print(f"[main] eval_batch=false in XPolicyLab/policy/{args.policy_name}/deploy.yml; "
              f"forcing num_envs {num_envs} -> 1")
        num_envs = 1
    OmegaConf.update(env_cfg, "sim.scene.num_envs", num_envs, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", num_envs, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, eval_num = process_config(env_cfg, task_name=task_name)
    override = os.environ.get("EVAL_NUM")
    if override and override.lower() != "native":
        eval_num = min(int(override), int(eval_num))
    eval_num = int(eval_num)
    OmegaConf.update(env_cfg, "eval_cfg.eval_num", eval_num, force_add=True)
    OmegaConf.update(env_cfg, "camera.default_frequency", eval_cfg["observation"].get("collect_freq", 0), force_add=True)
    env_cfg.sim.seed = [0 for _ in range(num_envs)]
    return env_cfg, eval_num
