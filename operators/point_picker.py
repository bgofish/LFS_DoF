# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Point picker operator - modal operator for picking points on the model."""

import numpy as np
from typing import Optional, Tuple

import lichtfeld as lf
import lichtfeld.selection as sel
from lfs_plugins.types import Operator, Event


# Module-level state for the _pending_pick pattern.
# The operator deposits a result here; on_update() in the panel consumes it.
_on_point_picked_callback = None  # callable(world_pos, point_num) set by the panel
_pick_point_num = 0
_pick_cancelled = False           # Set to True when operator is ESC/right-click cancelled
_pending_pick: Optional[Tuple] = None  # (world_position, point_num) awaiting on_update


def set_pick_callback(callback, point_num: int):
    """Set the module-level callback and arm a fresh pick session."""
    global _on_point_picked_callback, _pick_point_num, _pick_cancelled, _pending_pick
    _on_point_picked_callback = callback
    _pick_point_num = point_num
    _pick_cancelled = False
    _pending_pick = None


def clear_pick_callback():
    """Clear the pick callback and signal cancellation."""
    global _on_point_picked_callback, _pick_point_num, _pick_cancelled, _pending_pick
    _on_point_picked_callback = None
    _pick_point_num = 0
    _pick_cancelled = True
    _pending_pick = None


def was_pick_cancelled():
    """Check if pick was cancelled and clear the flag."""
    global _pick_cancelled
    if _pick_cancelled:
        _pick_cancelled = False
        return True
    return False


def consume_pending_pick() -> Optional[Tuple]:
    """Return and clear the pending pick result, or None if nothing is pending."""
    global _pending_pick
    result, _pending_pick = _pending_pick, None
    return result


class DEPTHMAP_OT_pick_point(Operator):
    """Modal operator for picking a point on the gaussian splat model."""

    # Full dotted ID required by the host: package.module.ClassName
    id          = "lfs_plugins.DoF.operators.point_picker.DEPTHMAP_OT_pick_point"
    label       = "Pick Depth Point"
    description = "Click on the model to pick a point for depth range"
    options     = {'BLOCKING'}

    def invoke(self, context, event: Event) -> set:
        """Start modal mode."""
        return {'RUNNING_MODAL'}

    def modal(self, context, event: Event) -> set:
        """Deposit pick results into _pending_pick; panel's on_update() consumes them."""
        global _pending_pick, _pick_point_num

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            result = sel.pick_at_screen(event.mouse_region_x, event.mouse_region_y)
            if result is not None and _on_point_picked_callback is not None:
                # Deposit the result for on_update() to consume — do NOT call the
                # panel method directly from the operator (wrong thread / re-entrancy).
                _pending_pick = (result.world_position, _pick_point_num)
            return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            clear_pick_callback()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def cancel(self, context):
        """Clean up on cancel."""
        clear_pick_callback()


def pick_point_at_screen(screen_x: float, screen_y: float) -> Optional[Tuple[float, float, float]]:
    """
    Find the frontmost gaussian splat at the given screen position.
    
    Uses splat scales to compute approximate screen-space radius for each splat,
    then picks the closest-to-camera splat that contains the click point.
    
    Args:
        screen_x: X pixel coordinate
        screen_y: Y pixel coordinate
        
    Returns:
        3D position (x, y, z) of the picked splat, or None if no splat found
    """
    # Get viewport render with screen positions
    vp_render = lf.get_viewport_render()
    if vp_render is None:
        return None
    
    screen_positions = vp_render.screen_positions
    if screen_positions is None:
        return None
    
    # Get scene and view
    scene = lf.get_scene()
    if not scene:
        return None
    
    view = lf.get_current_view()
    if not view:
        return None
    
    combined = scene.combined_model()
    if not combined:
        return None
    
    means = combined.get_means().numpy()  # [N, 3]
    screen_pos = screen_positions.numpy()  # [N, 2]
    
    if len(means) != len(screen_pos):
        return None
    
    # Get splat scales to compute screen-space radius
    scales = combined.get_scaling().numpy()  # [N, 3] - world-space scales
    
    # Camera position for depth calculation
    cam_pos = np.array(view.translation.numpy()).flatten()
    
    # Compute distance from camera to each splat
    to_splats = means - cam_pos
    depths = np.linalg.norm(to_splats, axis=1)
    
    # Compute approximate screen-space radius for each splat
    # Use the max scale dimension as the splat "radius" in world space
    # Then project to screen space: screen_radius ≈ world_radius * focal / depth
    world_radii = np.max(scales, axis=1)  # [N]
    
    # Approximate focal length from FOV and viewport size
    fov_rad = np.radians(view.fov_y)
    focal = view.height / (2.0 * np.tan(fov_rad / 2.0))
    
    # Screen-space radius (with minimum to ensure pickability)
    screen_radii = np.maximum(world_radii * focal / np.maximum(depths, 0.01), 5.0)
    
    # Distance from click to each splat center in screen space
    dx = screen_pos[:, 0] - screen_x
    dy = screen_pos[:, 1] - screen_y
    screen_distances = np.sqrt(dx*dx + dy*dy)
    
    # Find splats where click is within their screen radius
    # Also filter out points behind camera (negative depth conceptually, but we use distance)
    # and points with invalid screen positions
    valid_x = (screen_pos[:, 0] >= 0) & (screen_pos[:, 0] < view.width)
    valid_y = (screen_pos[:, 1] >= 0) & (screen_pos[:, 1] < view.height)
    within_radius = screen_distances < screen_radii
    
    mask = valid_x & valid_y & within_radius
    
    if not np.any(mask):
        # Fallback: use a larger fixed radius
        mask = screen_distances < 30.0
        if not np.any(mask):
            return None
    
    # Among valid splats, pick the one closest to camera (frontmost)
    valid_indices = np.where(mask)[0]
    valid_depths = depths[mask]
    closest_idx = valid_indices[np.argmin(valid_depths)]
    
    return tuple(means[closest_idx])


