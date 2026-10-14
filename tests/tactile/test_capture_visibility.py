"""Capture tactile and camera images at one physics state with separate visibility."""

from contextlib import contextmanager
from dataclasses import replace
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.observation_manager.obs_manager import ObsManager
from env.tactile.config import TactileConfigError, load_tactile_config
from env.tactile.gelsight import _TacExGelSight
from env.tactile.manager import RoboDojoTactileManager


class Visibility:
    """USD visibility opinion with an inherited fallback."""

    def __init__(self, value=None):
        self.value = value

    def Get(self):
        return self.value or "inherited"

    def HasAuthoredValueOpinion(self):
        return self.value is not None

    def Set(self, value):
        self.value = value
        return True

    def Clear(self):
        self.value = None


def _capture_stack(monkeypatch, *, tactile=True, render_in_vision=False, object_mode=False, failure=None):
    events = []
    visibility = ([Visibility("invisible"), Visibility("invisible")]
                  if object_mode else [Visibility("inherited"), Visibility()])
    housings = [Visibility("invisible"), Visibility("invisible")] if object_mode else []
    visual_attributes = [*visibility, *housings]
    backend_rgb = np.full((2, 2, 3), 17, dtype=np.uint8)
    camera_rgba = np.full((2, 2, 4), 200, dtype=np.uint8)
    camera_depth = torch.full((1, 2, 2, 1), 0.028)
    optical_rgba = torch.full((1, 2, 2, 4), 77, dtype=torch.uint8)
    simulation = SimpleNamespace(_sim_step_counter=7, physics_dt=0.004)
    env = SimpleNamespace(sim=simulation, episode_ids=["episode-0"], capture_failure=failure)
    manager = RoboDojoTactileManager(None) if tactile else None
    if manager is not None:
        manager.enabled = True
        manager.initialized = True
        manager.config = replace(manager.config, render_in_vision=render_in_vision,
                                 optical_input="object" if object_mode else "deformed_surface")
        manager.sim = simulation
        for name, attribute in zip(("left", "right"), visibility):
            sensor = _TacExGelSight.__new__(_TacExGelSight)
            sensor._resolution = (2, 2)
            sensor._render_proxy_mesh = SimpleNamespace(GetVisibilityAttr=lambda attribute=attribute: attribute)
            sensor.sensor_cfg = SimpleNamespace(calibration_id=name, camera_clipping_range=(0.024, 0.034))
            sensor.optical_input = "object" if object_mode else "deformed_surface"
            sensor.ensure_initialized = lambda: None
            sensor._nominal_state = None
            sensor.deformation_max_mm = lambda *, measured_pose=None: 0.0
            sensor.gelpad = SimpleNamespace(data=SimpleNamespace(nodal_pos_w=torch.zeros((1, 3))))
            sensor.attachment = SimpleNamespace(
                _compute_aim_positions=lambda: None, attachment_points_idx=[0], aim_positions=np.zeros((1, 3)),
                _aim_positions=np.zeros((1, 3)), obj_pose=torch.tensor([[[0, 0, 0, 1, 0, 0, 0]]]),
            )

            def update(*, dt, force_recompute, name=name):
                optical_visibility = "invisible" if object_mode else "inherited"
                assert all(attribute.Get() == optical_visibility for attribute in visibility)
                assert not manager._vision_rendering
                events.append(("tactile", name, simulation._sim_step_counter))

            if object_mode:
                housing = housings[len(manager.sensors)]

                @contextmanager
                def show_visuals(attribute=attribute, housing=housing):
                    previous = [(item, item.value) for item in (attribute, housing)]
                    try:
                        for item, _ in previous:
                            item.Set("inherited")
                        yield True
                    finally:
                        for item, value in previous:
                            item.value = value

                sensor.show_visuals_in_vision = show_visuals

            sensor.sensor = SimpleNamespace(
                update=update,
                data=SimpleNamespace(output={"tactile_rgb": backend_rgb, "height_map": torch.full((1, 2, 2), -0.12)}),
                indentation_depth=torch.full((1, 2, 2), -0.12),
                camera=SimpleNamespace(data=SimpleNamespace(output={"depth": camera_depth, "rgb": optical_rgba})),
            )
            manager.sensors[name] = sensor
            manager._references[name] = {"reference_rgb": np.full((2, 2, 3), 3, dtype=np.uint8),
                                         "reference_id": f"reference-{name}"}
        manager.before_render = lambda: events.append(("sync", simulation._sim_step_counter))

    def render():
        hidden = any(attribute.Get() == "invisible" for attribute in visual_attributes)
        optical_pass = hidden if object_mode else not hidden
        events.append(("render", hidden, simulation._sim_step_counter))
        if not optical_pass and env.capture_failure == "render":
            raise RuntimeError("render failed")
        backend_rgb.fill(17 if optical_pass else 250)
        camera_rgba.fill(60 if hidden else 200)
        camera_depth.fill_(0.028 if optical_pass else float("nan"))
        optical_rgba.fill_(77 if optical_pass else 255)

    def readout(*, env_ids):
        hidden = any(attribute.Get() == "invisible" for attribute in visibility)
        events.append(("vision", hidden, simulation._sim_step_counter))
        if env.capture_failure == "readout":
            raise RuntimeError("readout failed")
        return [{"rgb": [{"data": camera_rgba, "info": {}}]} for _ in range(3)]

    env.render = render
    simulation.render = render
    capture = SimpleNamespace(step=readout)

    def wait_for_latest_cameras(env, *, capture_manager=None, min_passes=1, max_passes=16):
        if capture_manager is not None:
            assert capture_manager is capture
        assert (min_passes, max_passes) == (1, 16)
        env.render()

    monkeypatch.setitem(sys.modules, "env.camera_manager.capture.render_sync",
                        SimpleNamespace(wait_for_latest_cameras=wait_for_latest_cameras))
    observation = ObsManager.__new__(ObsManager)
    observation.num_envs = 1
    observation.env = env
    observation.instruction = ["plug in charger"]
    observation.collect_freq = 25
    observation.dt = 0.004
    observation.tactile_manager = manager
    observation.capture_manager = capture
    observation.camera_manager = SimpleNamespace(
        num_cams=3, camera_names=[["cam_head", "cam_left_wrist", "cam_right_wrist"]],
    )
    observation.robot_manager = None
    observation.capture_render_passes = 1
    observation.capture_render_max_passes = 16
    observation.collect_shape = True
    observation.collect_intrinsic_matrix = False
    observation.collect_extrinsic_matrix = False
    return observation, manager, visibility, events, backend_rgb, camera_rgba


def test_object_tactile_snapshot_precedes_visible_sensor_camera_pass(monkeypatch):
    obs, manager, visibility, events, backend_rgb, camera_rgba = _capture_stack(
        monkeypatch, object_mode=True, render_in_vision=True,
    )
    obs.render_for_capture()
    result = obs.get_obs()[0]
    assert events == [("sync", 7), ("render", True, 7), ("tactile", "left", 7),
                      ("tactile", "right", 7), ("render", False, 7), ("vision", False, 7)]
    assert manager.sim._sim_step_counter == result["physics_step"] == 7
    assert result["sim_time"] == result["tactile"]["left"]["capture_sim_time"] == 0.028
    assert all(item.Get() == "invisible" for item in visibility)
    assert manager._optical_render_pending
    for sensor in manager.sensors.values():
        assert torch.isnan(sensor.sensor.camera.data.output["depth"]).all()
    # Neither backend buffers nor public reference storage can change an
    # observation that was already captured before the second render.
    backend_rgb.fill(0)
    camera_rgba.fill(0)
    for name, sensor in manager.sensors.items():
        sensor._last_rgb.fill(0)
        manager._references[name]["reference_rgb"].fill(0)
        assert np.all(result["tactile"][name]["rgb"] == 17)
        assert np.all(result["tactile"][name]["reference_rgb"] == 3)
        assert result["tactile"][name]["age_steps"] == 0
    assert all(np.all(camera["color"] == 200) for camera in result["vision"].values())


@pytest.mark.parametrize("failure", ["render", "readout"])
def test_object_visibility_restores_and_next_capture_fences_after_failure(monkeypatch, failure):
    obs, manager, visibility, events, _, _ = _capture_stack(
        monkeypatch, object_mode=True, render_in_vision=True, failure=failure,
    )
    obs.render_for_capture()
    with pytest.raises(RuntimeError, match="failed"):
        obs.get_obs()
    assert all(item.Get() == "invisible" for item in visibility)
    assert not manager._vision_rendering
    assert manager._optical_render_pending
    assert manager._capture_count == 1
    obs.env.capture_failure = None
    events.clear()
    obs.env.sim._sim_step_counter = 8
    manager._physics_generation += 1
    obs.render_for_capture()
    result = obs.get_obs()[0]
    assert events[:4] == [("sync", 8), ("render", True, 8),
                         ("tactile", "left", 8), ("tactile", "right", 8)]
    assert result["physics_step"] == manager.sim._sim_step_counter == 8
    assert manager._capture_count == 2
    assert result["tactile"]["left"]["frame_id"] == 1
    assert result["tactile"]["left"]["capture_sim_time"] == 0.032
    assert np.all(result["tactile"]["left"]["rgb"] == 17)
    assert np.all(result["vision"]["cam_head"]["color"] == 200)


def test_direct_object_capture_after_visible_pass_restores_hidden_render_buffers(monkeypatch):
    obs, manager, visibility, events, _, _ = _capture_stack(
        monkeypatch, object_mode=True, render_in_vision=True,
    )
    obs.render_for_capture()
    obs.get_obs()
    assert manager._optical_render_pending
    events.clear()
    obs.env.sim._sim_step_counter = 8
    manager._physics_generation += 1
    manager.capture(dt=obs.dt)
    assert events == [("sync", 8), ("render", True, 8),
                      ("tactile", "left", 8), ("tactile", "right", 8)]
    assert all(item.Get() == "invisible" for item in visibility)
    assert not manager._optical_render_pending
    assert manager.sim._sim_step_counter == 8


def test_visible_camera_depth_is_a_snapshot_of_its_own_pass(monkeypatch):
    obs, _, _, _, _, _ = _capture_stack(monkeypatch, object_mode=True, render_in_vision=True)
    depth = np.full((2, 2, 1), 0.02, dtype=np.float32)
    readout = obs.capture_manager.step

    def read_depth(*, env_ids):
        data = readout(env_ids=env_ids)
        for camera in data:
            camera["distance_to_image_plane"] = [{"data": depth, "info": {}}]
        return data

    obs.capture_manager.step = read_depth
    obs.collect_depth = True
    obs.collect_approximate_depth = True
    obs.render_for_capture()
    result = obs.get_obs()[0]
    depth.fill(0)
    for camera in result["vision"].values():
        assert np.all(camera["depth"] == np.float32(0.02))
        assert np.all(camera["approximate_depth"] == 20)


def test_tactile_snapshot_precedes_hidden_camera_capture_at_the_same_physics_step(monkeypatch):
    obs, manager, visibility, events, backend_rgb, camera_rgba = _capture_stack(monkeypatch)
    obs.render_for_capture()
    result = obs.get_obs()[0]
    assert events == [("sync", 7), ("render", False, 7), ("tactile", "left", 7),
                      ("tactile", "right", 7), ("render", True, 7), ("vision", True, 7)]
    assert manager.sim._sim_step_counter == result["physics_step"] == 7
    assert result["sim_time"] == result["tactile"]["left"]["capture_sim_time"] == 0.028
    assert visibility[0].value == "inherited"
    assert visibility[1].value is None
    backend_rgb.fill(0)
    camera_rgba.fill(0)
    for sensor in result["tactile"].values():
        assert sensor["valid"]
        assert np.all(sensor["rgb"] == 17)
    for camera in result["vision"].values():
        assert camera["color"].shape == (2, 2, 3)
        assert np.all(camera["color"] == 60)


@pytest.mark.parametrize("failure", ["render", "readout"])
def test_proxy_visibility_restores_even_when_camera_capture_raises(monkeypatch, failure):
    obs, _, visibility, _, _, _ = _capture_stack(monkeypatch, failure=failure)
    obs.render_for_capture()
    with pytest.raises(RuntimeError, match="failed"):
        obs.get_obs()
    assert visibility[0].value == "inherited"
    assert visibility[1].value is None


def test_debug_visibility_keeps_one_render_and_original_camera_image(monkeypatch):
    obs, _, visibility, events, _, _ = _capture_stack(monkeypatch, render_in_vision=True)
    obs.render_for_capture()
    result = obs.get_obs()[0]
    assert [event for event in events if event[0] == "render"] == [("render", False, 7)]
    assert ("vision", False, 7) in events
    assert all(attribute.Get() == "inherited" for attribute in visibility)
    assert np.all(result["vision"]["cam_head"]["color"] == 200)


@pytest.mark.parametrize("lifecycle_method", ["reset", "invalidate", "close"])
def test_lifecycle_invalidates_rgb_instead_of_reusing_old_episode(monkeypatch, lifecycle_method):
    obs, manager, _, _, _, _ = _capture_stack(monkeypatch)
    obs.render_for_capture()
    manager.capture(dt=obs.dt)
    sensor = manager.sensors["left"]
    getattr(sensor, lifecycle_method)()
    assert sensor.observe() == {"rgb": None, "valid": False}


def test_ordinary_camera_capture_needs_no_tactile_backend_or_second_render(monkeypatch):
    obs, _, _, events, _, _ = _capture_stack(monkeypatch, tactile=False)
    obs.render_for_capture()
    result = obs.get_obs()[0]
    assert events == [("render", False, 7), ("vision", False, 7)]
    assert "tactile" not in result


@pytest.mark.parametrize("inactive", ["disabled", "uninitialized", "suspended"])
def test_inactive_tactile_manager_does_not_hide_or_request_a_render(monkeypatch, inactive):
    obs, manager, visibility, events, _, _ = _capture_stack(monkeypatch)
    if inactive == "disabled":
        manager.enabled = False
    elif inactive == "uninitialized":
        manager.initialized = False
    else:
        manager._suspended = True
    obs.render_for_capture()
    result = obs.get_obs()[0]
    assert [event for event in events if event[0] == "render"] == [("render", False, 7)]
    assert "tactile" not in result
    assert all(attribute.Get() == "inherited" for attribute in visibility)


def test_render_in_vision_is_an_explicit_boolean_option():
    assert load_tactile_config({}).render_in_vision is False
    configured = load_tactile_config({"render_in_vision": True})
    assert configured.render_in_vision is True
    assert "render_in_vision" not in configured.extra
    for value in ("false", 0, None):
        with pytest.raises(TactileConfigError, match="render_in_vision"):
            load_tactile_config({"render_in_vision": value})
