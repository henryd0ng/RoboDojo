"""Optional visuo-tactile support for RoboDojo.

The package is deliberately import-light: importing RoboDojo with tactile
disabled must not import TacEx, UIPC, Warp, or any Isaac sensor extension.
"""

from .config import (
    TactileConfig,
    load_tactile_config,
    validate_calibration_dir,
    validate_tactile_config,
)
from .manager import RoboDojoTactileManager, TactileDependencyError

__all__ = [
    "RoboDojoTactileManager",
    "TactileConfig",
    "TactileDependencyError",
    "load_tactile_config",
    "validate_calibration_dir",
    "validate_tactile_config",
]
