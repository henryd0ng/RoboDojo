"""Exercise actual EvalEnv video methods without importing Isaac or running ffmpeg."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class FakeWriter:
    """Filesystem-backed lifecycle double; its files are not encoded videos."""

    def __init__(self, out_path, height, width, channels, fps):
        self.out_path = out_path
        self.height, self.width, self.channels, self.fps = height, width, channels, fps
        self.frames = []
        self.n_frames = 0
        self.close_calls = []
        self.abort_calls = 0
        self.close_error = None
        self.closed = False
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"partial")

    def append(self, color):
        if self.closed:
            raise RuntimeError("append after close")
        if color.shape != (self.height, self.width, self.channels):
            raise ValueError("frame dimensions changed")
        self.frames.append(color.copy())
        self.n_frames += 1

    def close(self, *, announce=True):
        self.close_calls.append(announce)
        if self.close_error is not None:
            raise self.close_error
        self.closed = True
        Path(self.out_path).write_bytes(b"finalized")

    def abort(self):
        self.abort_calls += 1
        self.closed = True
        Path(self.out_path).unlink(missing_ok=True)


class FakeMosaic:
    def __init__(self):
        self.observations = []
        self.frames = []

    def compose(self, observation):
        self.observations.append(observation)
        color = np.full((720, 1280, 3), observation["physics_step"] % 255, dtype=np.uint8)
        self.frames.append(color)
        return color


class NativeTaskStub:
    def reset(self, *, seed, options):
        # The real EvalEnv reset must abort old streams before native reset.
        self.native_resets.append((list(seed), options, len(self.video_writers)))


def extracted_environment_class():
    source = ROOT / "src/eval_client/eval_env.py"
    tree = ast.parse(source.read_text())
    definition = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "EvalEnv")
    functions = {node.name: node for node in definition.body if isinstance(node, ast.FunctionDef)}
    names = ("get_obs_batch", "reset", "_stream_videos", "_append_video_frame",
             "_sweep_stream_dir", "_abort_video_writers", "save_video",
             "get_action_type", "validate_action_dict", "take_action_batch")
    # Compile only selected real methods. Neither the module imports nor the
    # simulator-heavy constructor executes; super().reset uses the CPU double.
    selected = ast.ClassDef(name="EvalEnv", bases=[ast.Name(id="NativeTaskStub", ctx=ast.Load())],
                            keywords=[], body=[functions[name] for name in names], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    namespace = {"np": np, "os": os, "deepcopy": deepcopy, "NativeTaskStub": NativeTaskStub,
                 "VideoStreamWriter": FakeWriter,
                 "format_video_saved_message": lambda path, *_: f"saved:{Path(path).name}"}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["EvalEnv"]


def observation(step):
    return {
        "episode_id": f"episode-{step}", "physics_step": step, "sim_time": step / 120,
        "vision": {name: {"color": np.full((4, 6, 3), step + index, dtype=np.uint8)}
                   for index, name in enumerate(CAMERAS)},
        "tactile": {name: {"rgb": np.full((2, 2, 3), step + index + 10, dtype=np.uint8), "valid": True}
                    for index, name in enumerate(("left", "right"))},
    }


class EvaluationVideoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="robodojo-eval-video-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.environment_class = extracted_environment_class()

    def environment(self, *, mosaic=True, num_envs=1):
        env = self.environment_class()
        env.video_writers = {}
        env.video_mosaic = FakeMosaic() if mosaic else None
        env._stream_dir = str(self.directory / "_stream")
        env.num_envs = num_envs
        env.end_flag = [False] * num_envs
        env.physx_monitor_enabled = False
        env.observations = {index: observation(20 + index) for index in range(num_envs)}
        env.native_resets = []
        env.obs_manager = SimpleNamespace(
            collect_freq=12.5, render_for_capture=lambda: None, reset=lambda: None,
            get_obs=lambda *, env_idx_list: {index: env.observations[index] for index in env_idx_list},
        )
        env.scene_manager = SimpleNamespace(layout_manager=SimpleNamespace(set_saved_layout=lambda *_: None))
        env.seed_manager = SimpleNamespace(get_seed_scene_info=lambda seed: {"seed": seed})
        env.setup_scene = lambda: None
        env.robot_manager = SimpleNamespace(set_origin_endpose=lambda: None, set_robot_init_state=lambda: None)
        env.reward_manager = SimpleNamespace(init_state=lambda: None)
        env.model_client = SimpleNamespace(call=lambda **_: None)
        return env

    def test_same_observation_records_three_cameras_and_mosaic_and_reuses_streams(self):
        env = self.environment()
        frame = env.observations[0]
        result = env.get_obs_batch()[0]
        writers = env.video_writers[0].copy()
        self.assertEqual(set(writers), {*CAMERAS, "visuotactile"})
        self.assertIs(env.video_mosaic.observations[0], frame)
        self.assertEqual(result["env_idx"], 0)
        for name in CAMERAS:
            np.testing.assert_array_equal(writers[name].frames[0], frame["vision"][name]["color"])
        mosaic = writers["visuotactile"]
        np.testing.assert_array_equal(mosaic.frames[0], env.video_mosaic.frames[0])
        self.assertEqual((mosaic.height, mosaic.width, mosaic.channels, mosaic.fps), (720, 1280, 3, 12.5))
        self.assertEqual(mosaic.frames[0].dtype, np.uint8)

        next_frame = observation(30)
        env.observations[0] = next_frame
        env.get_obs_batch()
        self.assertIs(env.video_mosaic.observations[1], next_frame)
        for name, writer in writers.items():
            self.assertIs(env.video_writers[0][name], writer)
            self.assertEqual(writer.n_frames, 2)
        np.testing.assert_array_equal(mosaic.frames[1], env.video_mosaic.frames[1])

    def test_disabled_mosaic_preserves_only_the_three_camera_streams(self):
        env = self.environment(mosaic=False)
        env.get_obs_batch()
        self.assertEqual(set(env.video_writers[0]), set(CAMERAS))
        self.assertTrue(all(writer.n_frames == 1 for writer in env.video_writers[0].values()))

    def test_success_and_failure_finalize_mosaic_with_episode_outcome(self):
        for tag in ("success", "fail"):
            with self.subTest(tag=tag):
                env = self.environment()
                env.get_obs_batch()
                writers = env.video_writers[0].copy()
                output = self.directory / tag / "episode_0000000.mp4"
                with redirect_stdout(io.StringIO()):
                    env.save_video(0, str(output), tag)
                self.assertEqual(env.video_writers, {})
                for name, writer in writers.items():
                    final = output.with_name(f"episode_0000000_{name}_{tag}.mp4")
                    self.assertEqual(final.read_bytes(), b"finalized")
                    self.assertEqual(writer.close_calls, [False])
                    self.assertEqual(writer.abort_calls, 0)
                    self.assertFalse(Path(writer.out_path).exists())

    def test_abort_isolates_environments_and_never_promotes_partial_files(self):
        env = self.environment(num_envs=2)
        env.get_obs_batch()
        first, second = env.video_writers[0].copy(), env.video_writers[1].copy()
        env._abort_video_writers([0])
        self.assertEqual(set(env.video_writers), {1})
        for writer in first.values():
            self.assertEqual(writer.abort_calls, 1)
            self.assertFalse(Path(writer.out_path).exists())
            self.assertEqual(writer.close_calls, [])
        for name, writer in second.items():
            self.assertIs(env.video_writers[1][name], writer)
            self.assertEqual(writer.abort_calls, 0)
            self.assertTrue(Path(writer.out_path).exists())
        env.observations[0] = observation(40)
        env.get_obs_batch()
        for name in first:
            self.assertIsNot(env.video_writers[0][name], first[name])
            self.assertEqual(env.video_writers[0][name].n_frames, 1)
            self.assertEqual(second[name].n_frames, 2)
        env._abort_video_writers()
        self.assertEqual(env.video_writers, {})
        self.assertFalse(list(self.directory.rglob("*.mp4")))

    def test_real_reset_aborts_all_old_streams_before_native_reset(self):
        env = self.environment(num_envs=2)
        env.get_obs_batch()
        old = {index: writers.copy() for index, writers in env.video_writers.items()}
        env.reset(seed=[0, 1], options={"reset": True})
        self.assertEqual(env.native_resets, [([0, 1], {"reset": True}, 0)])
        self.assertEqual(env.video_writers, {})
        self.assertFalse(list(self.directory.rglob("*.mp4")))
        for writers in old.values():
            for writer in writers.values():
                self.assertEqual(writer.abort_calls, 1)
                self.assertEqual(writer.close_calls, [])
        env.observations[0] = observation(50)
        env.get_obs_batch(env_idx_list=[0])
        for name, writer in env.video_writers[0].items():
            self.assertIsNot(writer, old[0][name])
            self.assertEqual(writer.n_frames, 1)
        self.assertTrue(np.all(env.video_writers[0]["visuotactile"].frames[0] == 50))

    def test_failed_mosaic_finalize_discards_partial_and_does_not_announce_saved(self):
        env = self.environment()
        env.get_obs_batch()
        mosaic = env.video_writers[0]["visuotactile"]
        mosaic.close_error = OSError("encoder rejected output")
        output = self.directory / "episode_0000000.mp4"
        log = io.StringIO()
        with redirect_stdout(log):
            env.save_video(0, str(output), "success")
        self.assertEqual(env.video_writers, {})
        self.assertEqual(mosaic.abort_calls, 1)
        self.assertFalse(Path(mosaic.out_path).exists())
        self.assertFalse((self.directory / "episode_0000000_visuotactile_success.mp4").exists())
        self.assertIn("Failed to finalize", log.getvalue())
        self.assertNotIn("saved:episode_0000000_visuotactile_success.mp4", log.getvalue())
        for name in CAMERAS:
            self.assertTrue((self.directory / f"episode_0000000_{name}_success.mp4").exists())

    def test_orphan_sweep_deletes_only_partial_video_files(self):
        env = self.environment()
        stream = Path(env._stream_dir)
        stream.mkdir()
        for name in (*CAMERAS, "visuotactile"):
            (stream / f"env0_{name}.tmp.mp4").write_bytes(b"partial")
        completed = stream / "completed.mp4"
        completed.write_bytes(b"existing finalized file")
        env._sweep_stream_dir()
        self.assertEqual(list(stream.iterdir()), [completed])

    def test_ended_environment_records_only_the_requested_final_frame(self):
        env = self.environment()
        env.end_flag[0] = True
        env.get_obs_batch()
        self.assertEqual(env.video_writers, {})
        self.assertEqual(env.video_mosaic.observations, [])
        env.get_obs_batch(last_frame=True)
        self.assertIs(env.video_mosaic.observations[0], env.observations[0])
        self.assertTrue(all(writer.n_frames == 1 for writer in env.video_writers[0].values()))




if __name__ == "__main__":
    unittest.main()
