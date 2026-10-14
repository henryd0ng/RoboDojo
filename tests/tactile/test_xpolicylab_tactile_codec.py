"""Verify that XPolicyLab's websocket codec preserves tactile RGB arrays."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


_xpolicylab = Path(__file__).resolve().parents[2] / "XPolicyLab"
# XPolicyLab uses absolute client_server imports; preserve RoboDojo's
# existing import precedence and restore the search path after loading them.
with patch.object(sys, "path", [*sys.path, str(_xpolicylab)]):
    from client_server.ws.protocol.codec import decode_frame, encode_frame


def test_nested_tactile_rgb_msgpack_round_trip():
    left = np.zeros((128, 128, 3), dtype=np.uint8)
    right = np.ones((128, 128, 3), dtype=np.uint8) * 237
    payload = {
        "observation": {
            "tactile": {
                "left": {"rgb": left, "valid": True},
                "right": {"rgb": right, "valid": False},
            }
        }
    }

    decoded = decode_frame(encode_frame(payload))
    tactile = decoded["observation"]["tactile"]
    assert tactile["left"]["valid"] is True
    assert tactile["right"]["valid"] is False
    assert tactile["left"]["rgb"].shape == (128, 128, 3)
    assert tactile["left"]["rgb"].dtype == np.uint8
    assert tactile["right"]["rgb"].dtype == np.uint8
    np.testing.assert_array_equal(tactile["left"]["rgb"], left)
    np.testing.assert_array_equal(tactile["right"]["rgb"], right)
