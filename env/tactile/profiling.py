"""Bounded diagnostic sampling through public libuipc Timer/Logger/World APIs.

Only for the current single-world native path. Logger and Timer are process
global, so sampling requires exclusive ownership of UIPC stepping. Normal steps
call TacEx unchanged; selected steps reproduce its advance/retrieve pair exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from uuid import uuid4


def _log_record(logger, prefix, record):
    # This binding treats the entire message as a fmt format string, even
    # without arguments. Escape JSON braces so the native output stays valid.
    message = prefix + json.dumps(record, separators=(",", ":"))
    logger.info(message.replace("{", "{{").replace("}", "}}"))


def flatten_timers(tree):
    """Keep inclusive seconds/count per full path; parent/child time must not be summed."""
    rows = []

    def visit(node, parent=""):
        path = parent + "/" + node["name"]
        if parent:
            rows.append({"path": path, "inclusive_s": node["duration"], "count": node["count"]})
        for child in node.get("children", []):
            visit(child, path)

    if tree:
        visit(tree)
    return rows


class UipcStepProfiler:
    """Sample explicit zero-based manager step ordinals, across that manager's resets."""

    def __init__(self, output, steps=(300, 301, 600, 601), fine_steps=(301, 601)):
        steps, fine_steps = tuple(steps), tuple(fine_steps)
        if not steps or any(type(step) is not int or step < 0 for step in (*steps, *fine_steps)):
            raise ValueError("Profiling steps must be explicit nonnegative integers")
        if len(set(steps)) != len(steps) or not set(fine_steps).issubset(steps):
            raise ValueError("Profiling steps must be unique and fine_steps a subset")
        self.steps, self.fine_steps = set(steps), set(fine_steps)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.session = uuid4().hex
        self.step_index = 0

    def step(self, sim, dt, *, physics_generation, reset_count):
        index = self.step_index
        self.step_index += 1
        if index not in self.steps:
            return sim.step(dt)

        from uipc import Logger, Timer

        sample_id = f"{self.session}-{index:08d}"
        fine = index in self.fine_steps
        world = sim.world
        record = {
            "sample_id": sample_id,
            "manager_step_index": index,
            "physics_generation_before": physics_generation,
            "reset_count": reset_count,
            "world_frame_before": int(world.frame()),
            "dt_argument_s": float(dt),
            "scene_configuration": sim.scene.config().to_json(),
            "fine_timers_requested": fine,
            "advance_wall_s": None,
            "retrieve_wall_s": None,
            "error": None,
            "timing_semantics": (
                "Host-call wall time; retrieve includes the backend's implicit device sync. "
                "Fine Timer adds synchronization overhead; inclusive parent/child timers overlap."
            ),
        }
        try:
            Timer.disable_all()
            Timer.report_as_json()  # Drain between completed C++ calls, never inside a Timer scope.
            Logger.set_level(Logger.Info)
            _log_record(Logger, "ROBODOJO_UIPC_PROFILE_BEGIN ", record)
            if fine:
                Timer.enable_all()
            start = time.perf_counter()
            world.advance()
            advanced = time.perf_counter()
            record["advance_wall_s"] = advanced - start
            world.retrieve()
            record["retrieve_wall_s"] = time.perf_counter() - advanced
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            # Sampling is bounded even on failure. Never retain a world or GPU buffer.
            try:
                Timer.disable_all()
                try:
                    record["timers"] = Timer.report_as_json()
                    record["timer_rows"] = flatten_timers(record["timers"])
                    record["timers_available"] = bool(record["timer_rows"])
                except Exception as error:
                    record["timers_available"] = False
                    record["timer_error"] = f"{type(error).__name__}: {error}"
                try:
                    record["world_frame_after"] = int(world.frame())
                    record["world_valid_after"] = bool(world.is_valid())
                except Exception as error:
                    record["world_state_error"] = f"{type(error).__name__}: {error}"
                _log_record(
                    Logger,
                    "ROBODOJO_UIPC_PROFILE_END ",
                    {
                        "sample_id": sample_id,
                        "world_frame_after": record.get("world_frame_after"),
                        "error": record["error"],
                    },
                )
            finally:
                Logger.set_level(Logger.Error)  # Restore the current TacEx/manager baseline.
                (self.output / f"{sample_id}.json").write_text(json.dumps(record, allow_nan=False, indent=2) + "\n")
