# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth Map Visualization Plugin for LichtFeld Studio.

Provides Jet colormap and Grayscale depth visualization for Gaussian Splats.
"""

import lichtfeld as lf

from .panels.depthmap_panel import DepthmapPanel, _unregister_frame_handler, _depth_log
from .operators.point_picker import DEPTHMAP_OT_pick_point
from .core.colormaps import jet_colormap, grayscale_colormap
from .core.depthmap import apply_depthmap_colors

_classes = [DepthmapPanel, DEPTHMAP_OT_pick_point]


def on_load():
    """Called when plugin loads."""
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("Depth Map Visualization plugin loaded")
    _depth_log("=" * 60)
    _depth_log("SESSION START — Depth Map plugin loaded")


def on_unload():
    """Called when plugin unloads."""
    _depth_log("SESSION END — Depth Map plugin unloaded")
    _unregister_frame_handler()
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("Depth Map Visualization plugin unloaded")


__all__ = [
    "DepthmapPanel",
    "jet_colormap",
    "grayscale_colormap",
    "apply_depthmap_colors",
]
