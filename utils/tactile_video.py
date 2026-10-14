"""Compose native camera and tactile observations for episode video recording."""

from collections.abc import Mapping
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
SENSOR_NAMES = ("left", "right")


class TactileVideoMosaic:
    """Render three cameras and two current tactile RGB images without caching frames.

    ``compose`` accepts a single environment's normal observation and returns an
    RGB uint8 HWC frame suitable for ``VideoStreamWriter``. Missing images and
    invalid tactile captures are shown as unavailable, never as an earlier frame.
    """

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        if width < 640 or height < 360 or width % 2 or height % 2:
            raise ValueError("Mosaic dimensions must be even and at least 640 x 360.")
        self.width = width
        self.height = height
        self._scale = min(width / 1280, height / 720)
        self._margin = max(6, round(12 * self._scale))
        self._gap = max(5, round(10 * self._scale))
        self._header = max(20, round(36 * self._scale))
        self._font = self._load_font(max(10, round(18 * self._scale)))
        self._title_font = self._load_font(max(11, round(20 * self._scale)))
        self._boxes = self._layout()

    @staticmethod
    def _load_font(size: int):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def _layout(self) -> list[tuple[int, int, int, int]]:
        width = (self.width - 2 * self._margin - 2 * self._gap) // 3
        height = (self.height - 2 * self._margin - self._gap) // 2
        boxes = []
        for row in range(2):
            for column in range(3):
                x = self._margin + column * (width + self._gap)
                y = self._margin + row * (height + self._gap)
                boxes.append((x, y, x + width, y + height))
        return boxes

    @staticmethod
    def _rgb(value) -> np.ndarray | None:
        if value is None:
            return None
        try:
            image = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
            return None
        if image.shape[0] == 0 or image.shape[1] == 0:
            return None
        return image[..., :3]

    @staticmethod
    def _mapping(value) -> Mapping:
        return value if isinstance(value, Mapping) else {}

    def _panel_header(self, canvas: Image.Image, box, label: str) -> None:
        draw = ImageDraw.Draw(canvas)
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(18, 23, 30))
        draw.text((x0 + 8, y0 + 6), label, font=self._title_font, fill=(231, 237, 245))

    def _panel(self, canvas: Image.Image, box, label: str, rgb: np.ndarray | None) -> None:
        self._panel_header(canvas, box, label)
        draw = ImageDraw.Draw(canvas)
        x0, y0, x1, y1 = box
        image_box = (x0 + 4, y0 + self._header, x1 - 4, y1 - 4)
        if rgb is None:
            draw.text(
                ((x0 + x1) // 2, (image_box[1] + image_box[3]) // 2),
                "Unavailable", font=self._font, fill=(245, 164, 115), anchor="mm",
            )
            return
        available_width = image_box[2] - image_box[0]
        available_height = image_box[3] - image_box[1]
        scale = min(available_width / rgb.shape[1], available_height / rgb.shape[0])
        size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
        image = Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
        position = (image_box[0] + (available_width - size[0]) // 2,
                    image_box[1] + (available_height - size[1]) // 2)
        canvas.paste(image, position)

    @staticmethod
    def _number(value, *, seconds: bool = False) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "Unavailable"
        if not math.isfinite(number):
            return "Unavailable"
        return f"{number:.3f} s" if seconds else str(int(number))

    def compose(self, observation: Mapping) -> np.ndarray:
        """Return a new frame; observation arrays and metadata remain unchanged."""
        vision = self._mapping(observation.get("vision"))
        tactile = self._mapping(observation.get("tactile"))
        canvas = Image.new("RGB", (self.width, self.height), color=(9, 13, 19))
        valid_views = 0
        # Preserve the stock three-camera order while allowing other native
        # camera configurations with different available view names.
        camera_names = [name for name in CAMERA_NAMES if name in vision]
        camera_names.extend(name for name in vision if name not in camera_names)
        camera_names.extend(name for name in CAMERA_NAMES if name not in camera_names)
        for index, name in enumerate(camera_names[:3]):
            camera = self._mapping(vision.get(name))
            rgb = self._rgb(camera.get("color"))
            valid_views += rgb is not None
            self._panel(canvas, self._boxes[index], name, rgb)

        sensor_status = {}
        for index, name in enumerate(SENSOR_NAMES, start=3):
            sensor = self._mapping(tactile.get(name))
            valid = sensor.get("valid", False)
            rgb = self._rgb(sensor.get("rgb")) if isinstance(valid, (bool, np.bool_)) and valid else None
            sensor_status[name] = rgb is not None
            valid_views += rgb is not None
            self._panel(canvas, self._boxes[index], f"Tactile {name} - current RGB", rgb)

        box = self._boxes[5]
        self._panel_header(canvas, box, "Observation")
        draw = ImageDraw.Draw(canvas)
        x0, y0, _, _ = box
        lines = [
            f"Sim time: {self._number(observation.get('sim_time'), seconds=True)}",
            f"Physics step: {self._number(observation.get('physics_step'))}",
            f"Valid views: {valid_views} / 5",
        ]
        for name in SENSOR_NAMES:
            sensor = self._mapping(tactile.get(name))
            state = "valid" if sensor_status[name] else "Unavailable"
            lines.extend([
                f"{name}: {state} | frame {self._number(sensor.get('frame_id'))}",
                f"  Capture: {self._number(sensor.get('capture_sim_time'), seconds=True)}",
                f"  Age: {self._number(sensor.get('age_steps'))} physics steps",
            ])
        line_height = max(14, round(28 * self._scale))
        for index, line in enumerate(lines):
            draw.text((x0 + 8, y0 + self._header + index * line_height),
                      line, font=self._font, fill=(202, 211, 224))
        return np.asarray(canvas).copy()
