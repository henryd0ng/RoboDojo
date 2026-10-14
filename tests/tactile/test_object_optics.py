"""Object-depth optics keeps native contact physics and reset isolation."""

import ast
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from env.tactile.config import TactileConfigError, load_tactile_config
from env.tactile.gelsight import _TacExGelSight
from env.tactile.manager import RoboDojoTactileManager


def config(mode="object"):
    sensor = {
        "body_name": "finger",
        "optical_field_mm": [20.4, 15.3],
        "camera_clipping_range": [.024, .034],
    }
    return load_tactile_config({
        "enabled": True, "optical_input": mode, "resolution": [240, 320],
        **{name: {**sensor, "gelpad_prim_path": f"/robot/{name}/gel",
                  "sensor_prim_path": f"/robot/{name}/case"}
           for name in ("left", "right")},
    })


def test_optical_input_is_explicit_and_both_paths_are_supported():
    assert load_tactile_config({}).optical_input == "deformed_surface"
    assert config().optical_input == "object"
    assert "optical_input" not in config().extra
    for mode in (None, "unknown", True):
        with pytest.raises(TactileConfigError, match="optical_input"):
            load_tactile_config({"optical_input": mode})
    # A different near plane is allowed, but far must cover near + gel height.
    value = config()
    from env.tactile.config import validate_tactile_config
    validate_tactile_config(replace(value, left=replace(value.left, camera_clipping_range=(.020, .040))))
    with pytest.raises(TactileConfigError, match="nominal plane"):
        validate_tactile_config(replace(value, left=replace(value.left, camera_clipping_range=(.024, .027))))


def test_native_preset_selects_object_depth_at_univtac_resolution_and_reference():
    from utils.load_file import load_yaml

    value = load_tactile_config(load_yaml(Path(__file__).parents[2] / "env_cfg/arx_x5_pad_tactile.yml")["tactile"])
    assert value.optical_input == "object"
    assert value.resolution == (240, 320)
    assert value.left.camera_clipping_range == value.right.camera_clipping_range == (.027, .034)


@pytest.mark.parametrize("mode", ["object", "deformed_surface"])
def test_manager_selects_optical_geometry_without_creating_object_mode_proxy(mode):
    manager = RoboDojoTactileManager(config(mode))
    manager.uipc_sim = object()
    events = []
    for name in ("left", "right"):
        manager.sensors[name] = SimpleNamespace(
            _render_proxy_mesh=None,
            configure_object_rendering=lambda name=name: events.append(("object", name)),
            setup_render_proxy=lambda path: events.append(("surface", path)),
        )
    manager._ensure_optical_geometry()
    if mode == "object":
        assert events == [("object", "left"), ("object", "right")]
    else:
        assert events == [("surface", "/World/RoboDojoTactileRenderProxy/left"),
                          ("surface", "/World/RoboDojoTactileRenderProxy/right")]


def test_object_visibility_hides_usd_and_fabric_without_changing_solver_nodes(monkeypatch):
    usd_values, fabric_values = [], []
    usd_attr = SimpleNamespace(Set=usd_values.append)
    fabric_attr = SimpleNamespace(Set=fabric_values.append)
    mesh = SimpleNamespace(GetVisibilityAttr=lambda: usd_attr)
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(UsdGeom=SimpleNamespace(
        Imageable=lambda prim: prim, Tokens=SimpleNamespace(invisible="invisible"))))
    monkeypatch.setitem(sys.modules, "usdrt", SimpleNamespace())
    nodes = torch.randn(169, 3)
    original = nodes.clone()
    adapter = _TacExGelSight.__new__(_TacExGelSight)
    adapter.gelpad = SimpleNamespace(
        _usd_mesh_prim=mesh, fabric_prim=SimpleNamespace(GetAttribute=lambda _: fabric_attr),
        data=SimpleNamespace(nodal_pos_w=nodes),
    )
    adapter.configure_object_rendering()
    assert usd_values == fabric_values == ["invisible"]
    torch.testing.assert_close(nodes, original)
    assert not getattr(adapter, "_render_proxy_mesh", None)


@pytest.mark.parametrize("failure", [False, True])
def test_vision_pass_shows_cad_gel_then_restores_both_scene_graphs(monkeypatch, failure):
    class Visibility:
        def __init__(self, value):
            self.value = value

        def Get(self):
            return self.value if self.value is not None else "inherited"

        def HasAuthoredValueOpinion(self):
            return self.value is not None

        def Set(self, value):
            self.value = value

        def Clear(self):
            self.value = None

    sensor_path = "/World/envs/env_0/robot0/finger/gelsight_left"
    gel_path = "/World/envs/env_0/robot0/finger/gelpad_left/mesh"
    paths = [gel_path, f"{sensor_path}/case/mesh", f"{sensor_path}/plate/mesh"]
    usd = {path: Visibility("invisible") for path in paths}
    fabric = {path: Visibility("invisible") for path in paths}
    prims = {path: SimpleNamespace(
        GetPath=lambda path=path: path, IsValid=lambda: True,
        GetVisibilityAttr=lambda path=path: usd[path],
        IsA=lambda _: True,
    ) for path in paths}
    fabric_prims = {path: SimpleNamespace(GetAttribute=lambda _, path=path: fabric[path])
                    for path in paths}
    usd_stage = SimpleNamespace(GetPrimAtPath=prims.__getitem__)
    fabric_stage = SimpleNamespace(GetPrimAtPath=fabric_prims.__getitem__)
    context = SimpleNamespace(get_stage=lambda: usd_stage, get_stage_id=lambda: 1)
    monkeypatch.setitem(sys.modules, "omni.usd", SimpleNamespace(get_context=lambda: context))
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        UsdGeom=SimpleNamespace(Imageable=lambda prim: prim, Gprim=object),
    ))
    monkeypatch.setitem(sys.modules, "usdrt", SimpleNamespace(
        Usd=SimpleNamespace(Stage=SimpleNamespace(Attach=lambda _: fabric_stage)),
        Sdf=SimpleNamespace(Path=lambda path: path),
    ))
    nodes = torch.randn(169, 3)
    original = nodes.clone()
    value = _TacExGelSight.__new__(_TacExGelSight)
    value.sensor_cfg = SimpleNamespace(sensor_prim_path=sensor_path, name="left")
    value.gelpad = SimpleNamespace(_usd_mesh_prim=prims[gel_path],
                                 data=SimpleNamespace(nodal_pos_w=nodes))
    expected = pytest.raises(RuntimeError, match="vision failed") if failure else nullcontext()
    with expected:
        with value.show_visuals_in_vision() as changed:
            assert changed
            for graph in (usd, fabric):
                assert graph[paths[0]].Get() == "inherited"
                assert graph[paths[1]].Get() == graph[paths[2]].Get() == "invisible"
            if failure:
                raise RuntimeError("vision failed")
    assert all(attribute.Get() == "invisible" for graph in (usd, fabric) for attribute in graph.values())
    torch.testing.assert_close(nodes, original)


def adapter(depth, mode):
    result = _TacExGelSight.__new__(_TacExGelSight)
    result.optical_input = mode
    result.sensor_cfg = config().left
    result._resolution = depth.shape[-2:]
    result.ensure_initialized = lambda: None
    result.attachment = SimpleNamespace(_compute_aim_positions=lambda: None,
        obj_pose=torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]))
    result.gelpad = SimpleNamespace(data=SimpleNamespace(nodal_pos_w=torch.zeros(169, 3)))
    result.sensor = SimpleNamespace(
        update=lambda **_: None,
        camera=SimpleNamespace(update=lambda **_: None,
            data=SimpleNamespace(output={"depth": depth[..., None]})),
        data=SimpleNamespace(output={"tactile_rgb": np.zeros((*depth.shape[-2:], 3), dtype=np.uint8)}),
    )
    return result


@pytest.mark.parametrize("mode", ["object", "deformed_surface"])
def test_reference_is_recorded_but_only_surface_mode_installs_pixel_baseline(monkeypatch, mode):
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", SimpleNamespace(quat_apply_inverse=lambda q, p: p))
    depth = torch.full((1, 3, 3), float("inf") if mode == "object" else .028)
    value = adapter(depth, mode)
    value.sensor._robodojo_nominal_depth = torch.zeros_like(depth)
    value.sensor._robodojo_nominal_mask = torch.zeros_like(depth, dtype=torch.bool)
    value.calibrate_nominal()
    torch.testing.assert_close(value.reference_depth(), depth)
    if mode == "object":
        assert not hasattr(value.sensor, "_robodojo_nominal_depth")
        assert not hasattr(value.sensor, "_robodojo_nominal_mask")
    else:
        torch.testing.assert_close(value.sensor._robodojo_nominal_depth, depth)
        assert value.sensor._robodojo_nominal_mask.all()


def test_object_capture_replaces_current_validity():
    depth = torch.tensor([[[.027, .034], [0., float("inf")]]])
    value = adapter(depth, "object")
    value.capture(.04)
    torch.testing.assert_close(value.sensor._robodojo_depth_valid,
                              torch.tensor([[[True, False], [False, False]]]))
    depth.fill_(.028)
    value.capture(.04)
    assert value.sensor._robodojo_depth_valid.all()


def test_manager_records_empty_object_reference_without_a_vendor_pixel_baseline(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", SimpleNamespace(quat_apply_inverse=lambda q, p: p))
    manager = RoboDojoTactileManager(config())
    manager.initialized = True
    manager.sim = SimpleNamespace(_sim_step_counter=0, physics_dt=.004)
    manager._prepared_scene = SimpleNamespace(bindings={})
    events = []
    manager.before_render = lambda: events.append("sync")
    for name in ("left", "right"):
        manager.sensors[name] = adapter(torch.full((1, 3, 3), float("inf")), "object")
    manager.calibrate_reference(lambda: events.append("render"))
    assert events == ["sync", "render"]
    observations = manager.observe([0])[0]
    for name in ("left", "right"):
        assert observations[name]["valid"]
        assert observations[name]["reference_rgb"].shape == (3, 3, 3)
        assert len(observations[name]["reference_id"]) == 64
        assert not hasattr(manager.sensors[name].sensor, "_robodojo_nominal_depth")


def test_vendor_object_height_uses_fixed_near_plus_gel_without_reset_subtraction():
    spec = importlib.util.find_spec("tacex")
    if spec is None or spec.origin is None:
        pytest.skip("Optional TacEx source is not installed")
    source = ast.parse((Path(spec.origin).parent / "gelsight_sensor.py").read_text())
    method = next(node for node in ast.walk(source)
                  if isinstance(node, ast.FunctionDef) and node.name == "_get_height_map")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "_get_height_map", "exec"), namespace)
    depth = torch.tensor([[[.027, .028, .029, float("inf")]]])
    sensor = SimpleNamespace(
        camera=SimpleNamespace(data=SimpleNamespace(output={"depth": depth[..., None]})),
        _data=SimpleNamespace(output={"height_map": torch.zeros_like(depth)}),
        cfg=SimpleNamespace(optical_sim_cfg=SimpleNamespace(input_mode="object"),
            sensor_camera_cfg=SimpleNamespace(clipping_range=(.024, .034)),
            gelpad_dimensions=SimpleNamespace(height=.004)),
    )
    torch.testing.assert_close(namespace["_get_height_map"](sensor), torch.tensor([[[-1., 0., 0., 0.]]]),
                              atol=2e-6, rtol=0)


def test_vendor_object_wrapper_uses_native_render_direct_with_gaussian_path():
    spec = importlib.util.find_spec("tacex")
    if spec is None or spec.origin is None:
        pytest.skip("Optional TacEx source is not installed")
    source = ast.parse((Path(spec.origin).parent / "simulation_approaches/gpu_taxim/taxim_sim.py").read_text())
    method = next(node for node in ast.walk(source)
                  if isinstance(node, ast.FunctionDef) and node.name == "optical_simulation")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "optical_simulation", "exec"), namespace)
    height = torch.full((1, 3, 3), -.8)
    presses = torch.tensor([.8])
    calls = []

    def render_direct(field, **kwargs):
        calls.append((field.clone(), kwargs))
        return torch.zeros(1, 3, 3, 3)

    simulator = SimpleNamespace(
        sensor=SimpleNamespace(_data=SimpleNamespace(output={"height_map": height}), _indentation_depth=presses),
        cfg=SimpleNamespace(input_mode="object", tactile_img_res=(3, 3), with_shadow=False),
        _device="cpu", _taxim=SimpleNamespace(render_direct=render_direct),
        tactile_rgb_img=torch.empty(1, 3, 3, 3),
    )
    namespace["optical_simulation"](simulator)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], height)
    assert calls[0][1] == {"with_shadow": False, "press_depth": presses, "orig_hm_fmt": False}
