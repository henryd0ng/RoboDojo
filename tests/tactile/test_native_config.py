"""Native layout selection must be independent of optional tactile input."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import DictConfig, OmegaConf

from env.seed_manager import seed_manager
from env_cfg.camera.template import D435, PINHOLE
from utils.load_file import load_yaml

ROOT = Path(__file__).resolve().parents[2]


def native_method(path, class_name, method_name, namespace):
    """Exercise a real native method without importing Isaac's application."""
    source = ROOT / path
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method_name]


class NativeCameraConfigTests(unittest.TestCase):

    def test_native_pinhole_uses_original_camera_template(self):
        template = deepcopy(D435)
        for override, expected in ((None, D435["clipping_range"]),):
            with self.subTest(override=override):
                camera_cfg = {"type": "d435", "mesh": "pinhole", "mount_link": "robot0/panda_hand"}
                if override is not None:
                    camera_cfg["clipping_range"] = override
                camera = Mock()
                manager = SimpleNamespace(
                    num_envs=1,
                    init_config=OmegaConf.create({"default_frequency": 25, "cam_wrist": {"camera": camera_cfg}}),
                    cameras=[], cameras_xform_path=[], cameras_xform=[], action_noise=[], camera_names=[],
                )
                method = native_method("env/camera_manager/camera_manager.py", "CameraManager", "create_cameras", {
                    "deepcopy": deepcopy, "OmegaConf": OmegaConf,
                    "REAL_MAP": {"d435": D435}, "VISUAL_MAP": {"pinhole": PINHOLE},
                    "torch": SimpleNamespace(zeros=np.zeros),
                    "find_unique_string_name": lambda name, **_: name,
                    "is_prim_path_valid": lambda _: False,
                    "get_current_stage": lambda: SimpleNamespace(DefinePrim=lambda *_: None),
                    "SingleGeometryPrim": lambda **_: SimpleNamespace(),
                    "Camera": lambda **_: camera,
                })
                method(manager)
                camera.set_clipping_range.assert_called_once_with(near_distance=expected[0], far_distance=expected[1])
                self.assertEqual(D435, template)


class NativeTactileConfigTests(unittest.TestCase):
    def setUp(self):
        self.stock = load_yaml(ROOT / "env_cfg" / "arx_x5.yml")
        self.surface = load_yaml(ROOT / "env_cfg" / "arx_x5_pad_tactile.yml")


    def test_surface_preset_preserves_native_layout_and_dual_arm_actions(self):
        self.assertEqual(self.surface["config_name"], self.stock["config_name"])
        self.assertEqual(self.surface["config"]["robot"], "dual_x5_pad")
        stock_robots = load_yaml(ROOT / "env_cfg/robot/dual_x5.yml")["robots"]
        surface_robots = load_yaml(ROOT / "env_cfg/robot/dual_x5_pad.yml")["robots"]
        self.assertEqual([robot["robot_name"] for robot in surface_robots], ["x5_pad", "x5"])
        for stock, surface in zip(stock_robots, surface_robots):
            self.assertEqual({key: value for key, value in stock.items() if key != "robot_name"},
                             {key: value for key, value in surface.items() if key != "robot_name"})
        dimensions = json.loads((ROOT / "env_cfg/robot/_robot_info.json").read_text())
        self.assertEqual(dimensions["dual_x5_pad"], dimensions["dual_x5"])

    def test_all_presets_load_the_same_native_task_layouts(self):
        tasks = ("insert_tubes", "deposit_coin", "plug_in_charger")
        with tempfile.TemporaryDirectory() as temp_dir:
            layout_dir = Path(temp_dir) / "Eval_Layout" / seed_manager.BENCHMARK / "arx_x5" / "0"
            layout_dir.mkdir(parents=True)
            for task in tasks:
                for layout_id in (2, 10):
                    (layout_dir / f"{task}_{layout_id}.json").write_text(
                        json.dumps({"task": task, "layout_id": layout_id}), encoding="utf-8"
                    )

            with patch.object(seed_manager, "ASSETS_PATH", temp_dir):
                for task in tasks:
                    for preset in (self.stock, self.surface):
                        with self.subTest(
                            task=task, robot=preset["config"]["robot"]
                        ):
                            cfg = dict(preset, task_name=task, num_envs=1, seed=0)
                            manager = seed_manager.SeedManager(cfg)
                            manager.init_eval()
                            self.assertEqual(manager.get_seeds(), [0])
                            self.assertEqual(manager.get_seeds(), [1])
                            self.assertIsNone(manager.get_seeds())
                            layouts = [manager.get_seed_scene_info(index) for index in (0, 1)]
                            self.assertEqual(layouts, [{"task": task, "layout_id": index} for index in (2, 10)])
                            self.assertEqual(
                                manager.seed_info,
                                {
                                    index: {"scene_layout": str(layout_dir / f"{task}_{layout_id}.json")}
                                    for index, layout_id in enumerate((2, 10))
                                },
                            )


if __name__ == "__main__":
    unittest.main()
