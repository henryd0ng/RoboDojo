"""Shared JSON trace and WebSocket service for the X5 contact reference."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import signal
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "XPolicyLab")]

from client_server.ws.model_server import PolicyServer, PolicyServerConfig


class ReferencePolicy:
    def __init__(self, output: Path, expect_tactile: str, *, reference: dict):
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.trace = (output / "policy_trace.jsonl").open("x")
        self.expect_tactile = expect_tactile
        self.reference = reference
        self.reset_count = self.observation_count = self.action_count = 0
        self.targets = None
        self._save_summary()

    def _record(self, event):
        self.trace.write(json.dumps(event, allow_nan=False) + "\n")
        self.trace.flush()

    def _save_summary(self):
        summary = {
            "policy": "x5_contact_reference",
            "uses_privileged_reference": self.reference is not None,
            "reference": self.reference,
            "expect_tactile": self.expect_tactile,
            "reset_count": self.reset_count,
            "observation_count": self.observation_count,
            "action_count": self.action_count,
            "targets": self.targets,
        }
        if self.reference is not None:
            summary.update(
                phase=self.phase,
                failed_reason=self.failed_reason,
                task_success_claimed=False,
                tactile_contact_verified=False,
            )
        path = self.output / "policy_summary.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def reset(self):
        self.reset_count += 1
        self.targets = None
        self._record({"event": "reset", "reset_count": self.reset_count})
        self._save_summary()

    def update_obs(self, observation):
        has_tactile = "tactile" in observation
        if self.expect_tactile == "disabled" and has_tactile:
            raise ValueError("Tactile-disabled observation unexpectedly contains tactile")
        if self.expect_tactile == "enabled" and not has_tactile:
            raise ValueError("Tactile-enabled observation lacks tactile")
        state = observation["state"]
        if self.targets is None:
            self.targets = {
                key: np.asarray(value, dtype=np.float32).reshape(-1).tolist()
                for key, value in state.items()
                if key.endswith("joint_state")
            }
            if not self.targets or not all(np.isfinite(value).all() for value in self.targets.values()):
                raise ValueError("Observation has no finite joint-state action targets")
        self.observation_count += 1
        vision = {}
        for name, camera in observation.get("vision", {}).items():
            rgb = np.asarray(camera["color"])
            vision[name] = {"shape": list(rgb.shape), "dtype": str(rgb.dtype)}
            if self.observation_count == 1:
                vision[name]["sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
        tactile = {}
        for name, item in observation.get("tactile", {}).items():
            rgb = np.asarray(item["rgb"])
            if not item["valid"] or rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"Invalid tactile RGB for {name}")
            tactile[name] = {"shape": list(rgb.shape), "valid": True}
        self._record(
            {
                "event": "observation",
                "index": self.observation_count,
                "env_idx": observation.get("env_idx"),
                "instruction": observation.get("instruction"),
                "metadata": observation.get("metadata", {}),
                "episode_id": observation.get("episode_id"),
                "physics_step": observation.get("physics_step"),
                "sim_time": observation.get("sim_time"),
                "state": {key: np.asarray(value).tolist() for key, value in state.items()},
                "vision": vision,
                "tactile_present": has_tactile,
                "tactile": tactile,
            }
        )

    def close(self):
        self._record({"event": "server_close"})
        self._save_summary()
        self.trace.close()


async def serve(args, policy):
    server = PolicyServer(policy, PolicyServerConfig(host="127.0.0.1", port=args.port))
    stopped = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stopped.set)
    try:
        await server.start()
        print(f"READY ws://127.0.0.1:{args.port}", flush=True)
        await stopped.wait()
    finally:
        await server.stop()
        policy.close()
