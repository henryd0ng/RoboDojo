"""CPU checks for the five-view native observation video."""

from copy import deepcopy
import json
import shutil
import subprocess

import numpy as np
from PIL import ImageDraw
import pytest

from utils.save_file import VideoStreamWriter
from utils.tactile_video import TactileVideoMosaic


def _image(color, shape=(48, 64)):
    return np.full((*shape, 3), color, dtype=np.uint8)


def _observation():
    return {
        "vision": {
            "cam_head": {"color": _image((255, 0, 0))},
            "cam_left_wrist": {"color": _image((0, 255, 0))},
            "cam_right_wrist": {"color": _image((0, 0, 255))},
        },
        "tactile": {
            "left": {"rgb": _image((255, 255, 0), (128, 128)), "valid": True,
                     "frame_id": 8, "capture_sim_time": 1.234, "age_steps": 0},
            "right": {"rgb": _image((255, 0, 255), (128, 128)), "valid": np.bool_(True),
                      "frame_id": 8, "capture_sim_time": 1.234, "age_steps": 0},
        },
        "sim_time": 1.234,
        "physics_step": 309,
    }


def test_five_views_use_rgb_in_the_documented_order_without_mutating_observation():
    observation = _observation()
    before = deepcopy(observation)
    frame = TactileVideoMosaic().compose(observation)
    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8
    for row, column, color in [(0, 0, (255, 0, 0)), (0, 1, (0, 255, 0)),
                               (0, 2, (0, 0, 255)), (1, 0, (255, 255, 0)), (1, 1, (255, 0, 255))]:
        np.testing.assert_array_equal(frame[200 + row * 353, 218 + column * 422], color)
    for name, camera in observation["vision"].items():
        np.testing.assert_array_equal(camera["color"], before["vision"][name]["color"])
    for name, sensor in observation["tactile"].items():
        np.testing.assert_array_equal(sensor["rgb"], before["tactile"][name]["rgb"])
        assert sensor["valid"] == before["tactile"][name]["valid"]
    assert observation["sim_time"] == before["sim_time"]
    assert not np.shares_memory(frame, observation["vision"]["cam_head"]["color"])


def test_letterboxing_preserves_image_aspect_ratio():
    observation = {"vision": {"cam_head": {"color": _image((255, 0, 0), (40, 160))}}}
    frame = TactileVideoMosaic().compose(observation)
    rows, columns = np.nonzero(np.all(frame == (255, 0, 0), axis=-1))
    width = columns.max() - columns.min() + 1
    height = rows.max() - rows.min() + 1
    assert width / height == pytest.approx(4, abs=0.02)
    assert np.all(frame[70, 218] == (18, 23, 30))


def test_invalid_or_missing_capture_never_reuses_previous_rgb(monkeypatch):
    drawn = []
    original_text = ImageDraw.ImageDraw.text

    def record_text(self, position, text, *args, **kwargs):
        drawn.append(text)
        return original_text(self, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    mosaic = TactileVideoMosaic()
    observation = _observation()
    mosaic.compose(observation)
    drawn.clear()
    observation["tactile"]["left"]["valid"] = False
    observation["tactile"]["right"]["rgb"] = None
    del observation["vision"]["cam_right_wrist"]
    frame = mosaic.compose(observation)
    assert "Valid views: 2 / 5" in drawn
    assert drawn.count("Unavailable") == 3
    assert "left: Unavailable | frame 8" in drawn
    assert not np.any(np.all(frame == (255, 255, 0), axis=-1))
    assert not np.any(np.all(frame == (255, 0, 255), axis=-1))
    assert not np.any(np.all(frame == (0, 0, 255), axis=-1))


def test_labels_and_time_identify_current_observation(monkeypatch):
    drawn = []
    original_text = ImageDraw.ImageDraw.text

    def record_text(self, position, text, *args, **kwargs):
        drawn.append(text)
        return original_text(self, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    TactileVideoMosaic().compose(_observation())
    assert {"cam_head", "cam_left_wrist", "cam_right_wrist",
            "Tactile left - current RGB", "Tactile right - current RGB"} <= set(drawn)
    assert "Sim time: 1.234 s" in drawn
    assert "Physics step: 309" in drawn
    assert "Valid views: 5 / 5" in drawn
    assert "  Age: 0 physics steps" in drawn
    assert not any("force" in label.lower() for label in drawn)


@pytest.mark.parametrize("bad_rgb", [np.zeros((5, 5), dtype=np.uint8),
                                     np.zeros((5, 5, 3), dtype=np.float32),
                                     np.zeros((0, 5, 3), dtype=np.uint8)])
def test_malformed_images_are_unavailable_instead_of_silently_converted(bad_rgb):
    frame = TactileVideoMosaic().compose({"tactile": {"left": {"rgb": bad_rgb, "valid": True}}})
    assert frame.shape == (720, 1280, 3)
    assert np.array_equal(frame, TactileVideoMosaic().compose({}))


def test_five_view_frames_encode_to_an_mp4(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required for the video encoding check")
    output = tmp_path / "five_views.mp4"
    mosaic = TactileVideoMosaic(width=640, height=360)
    writer = VideoStreamWriter(str(output), mosaic.height, mosaic.width, 3, fps=5)
    for step in range(3):
        observation = _observation()
        observation["physics_step"] = step
        writer.append(mosaic.compose(observation))
    writer.close(announce=False)
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,pix_fmt", "-of", "json", str(output),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    assert (stream["width"], stream["height"], stream["nb_frames"]) == (640, 360, "3")
    assert stream["pix_fmt"] == "yuv420p"
