"""Load the X5A assembly with the supplied tactile finger pads."""

from env.global_configs import ROBOTS_PATH

from .x5 import get_robot_config as get_x5_config


def get_robot_config():
    config = get_x5_config()
    return config.replace(
        spawn=config.spawn.replace(usd_path=f"{ROBOTS_PATH}/x5_pad/ARX.usda")
    )
