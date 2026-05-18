# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth map computation and color application for Gaussian Splats."""

import numpy as np
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
import datetime

import lichtfeld as lf

from .colormaps import get_colormap, jet_colormap, grayscale_colormap

DEPTH_LOG_PATH = r"u:\temp\DEPTH.TXT"

def _depth_log(msg: str):
    try:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(DEPTH_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


@dataclass
class BoundingBox:
    """3D axis-aligned bounding box for region-of-interest selection."""
    min_x: float = -float('inf')
    max_x: float = float('inf')
    min_y: float = -float('inf')
    max_y: float = float('inf')
    min_z: float = -float('inf')
    max_z: float = float('inf')
    
    def contains(self, positions: np.ndarray) -> np.ndarray:
        inside = (
            (positions[:, 0] >= self.min_x) & (positions[:, 0] <= self.max_x) &
            (positions[:, 1] >= self.min_y) & (positions[:, 1] <= self.max_y) &
            (positions[:, 2] >= self.min_z) & (positions[:, 2] <= self.max_z)
        )
        return inside
    
    def is_valid(self) -> bool:
        return (
            self.min_x < self.max_x and
            self.min_y < self.max_y and
            self.min_z < self.max_z and
            np.isfinite(self.min_x) and np.isfinite(self.max_x) and
            np.isfinite(self.min_y) and np.isfinite(self.max_y) and
            np.isfinite(self.min_z) and np.isfinite(self.max_z)
        )


def compute_depth_values(
    positions: np.ndarray,
    axis: Literal["x", "y", "z", "camera"] = "z",
    camera_pos: Optional[np.ndarray] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    bbox: Optional[BoundingBox] = None,
    camera_fwd: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute normalized depth values from positions.
    
    For axis="camera": if camera_fwd is supplied, computes projective depth
    (dot product along forward vector — flat bands perpendicular to view).
    If camera_fwd is None, falls back to radial Euclidean distance.
    """
    positions = np.asarray(positions)
    n_points = positions.shape[0]
    
    if bbox is not None and bbox.is_valid():
        mask = bbox.contains(positions)
    else:
        mask = np.ones(n_points, dtype=bool)
    
    # Normalise axis aliases
    if axis in ("camera_parallel", "camera_radial"):
        axis = "camera"

    if axis == "x":
        depths = positions[:, 0]
    elif axis == "y":
        depths = positions[:, 1]
    elif axis == "z":
        depths = positions[:, 2]
    elif axis == "camera":
        if camera_pos is None:
            camera_pos = np.array([0.0, 0.0, 0.0])
        if camera_fwd is not None:
            # Projective depth: signed distance along look direction.
            # Produces flat bands parallel to the image plane.
            fwd = camera_fwd / (np.linalg.norm(camera_fwd) + 1e-12)
            depths = (positions - camera_pos).dot(fwd)
        else:
            # Radial fallback
            depths = np.linalg.norm(positions - camera_pos, axis=1)
    else:
        raise ValueError(f"Unknown axis: {axis}. Use 'x', 'y', 'z', or 'camera'")
    
    masked_depths = depths[mask] if mask.any() else depths
    actual_min = min_depth if min_depth is not None else float(np.min(masked_depths))
    actual_max = max_depth if max_depth is not None else float(np.max(masked_depths))
    
    if actual_min > actual_max:
        actual_min, actual_max = actual_max, actual_min
    
    depth_range = actual_max - actual_min
    if depth_range < 1e-6:
        depth_range = 1.0
    
    normalized = (depths - actual_min) / depth_range
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized, mask, actual_min, actual_max


def _get_keyframe_transforms():
    """Read full 4x4 world_transform from each keyframe node.
    Returns list of np.ndarray shape (4,4), one per keyframe in order."""
    try:
        rs = lf.get_render_scene()
        nodes = list(rs.get_nodes())
        kf_container = next(
            (n for n in nodes if getattr(n, 'name', '') == 'Keyframes'), None)
        if kf_container is None:
            return []
        transforms = []
        for child_id in kf_container.children:
            child = rs.get_node_by_id(child_id)
            t = np.array(child.world_transform, dtype=np.float64)  # (4,4)
            transforms.append(t)
        return transforms
    except Exception as e:
        _depth_log(f"KEYFRAMES | read error: {e}")
        return []


def _mat3_to_quat(m):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    m = np.array(m, dtype=np.float64)
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
        w = (m[2,1] - m[1,2]) / s
        x = 0.25 * s
        y = (m[0,1] + m[1,0]) / s
        z = (m[0,2] + m[2,0]) / s
    elif m[1,1] > m[2,2]:
        s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
        w = (m[0,2] - m[2,0]) / s
        x = (m[0,1] + m[1,0]) / s
        y = 0.25 * s
        z = (m[1,2] + m[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
        w = (m[1,0] - m[0,1]) / s
        x = (m[0,2] + m[2,0]) / s
        y = (m[1,2] + m[2,1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    # Ensure shortest path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        # Nearly identical — linear interpolate
        return (q1 + t * (q2 - q1)) / np.linalg.norm(q1 + t * (q2 - q1))
    theta0 = np.arccos(dot)
    theta  = theta0 * t
    sin0   = np.sin(theta0)
    return (np.sin(theta0 - theta) / sin0) * q1 + (np.sin(theta) / sin0) * q2


def _quat_to_mat3(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _interpolate_keyframe_transform(transforms, current_frame, total_frames):
    """Interpolate between keyframe transforms using SLERP for rotation
    and linear interpolation for translation — prevents orientation flipping."""
    n = len(transforms)
    if n == 0:
        return None
    if n == 1:
        return transforms[0]

    t = (current_frame / max(total_frames - 1, 1)) * (n - 1)
    i = int(t)
    i = max(0, min(i, n - 2))
    frac = t - i

    m0, m1 = transforms[i], transforms[i + 1]

    # Linear interpolation for translation
    pos = m0[:3, 3] * (1.0 - frac) + m1[:3, 3] * frac

    # SLERP for rotation
    q0 = _mat3_to_quat(m0[:3, :3])
    q1 = _mat3_to_quat(m1[:3, :3])
    q  = _quat_slerp(q0, q1, frac)
    rot = _quat_to_mat3(q)

    # Rebuild 4x4
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rot
    result[:3,  3] = pos
    return result


def _camera_state_from_transform(m, fov=None):
    """Extract eye, target, up from a 4x4 camera world_transform.
    Column 3 = eye, Column 2 = -forward (camera looks down -Z), Column 1 = up.
    """
    eye     = m[:3, 3]
    forward = -m[:3, 2]   # camera looks down -Z
    norm = np.linalg.norm(forward)
    if norm > 1e-6:
        forward = forward / norm
    up = m[:3, 1]
    up_norm = np.linalg.norm(up)
    if up_norm > 1e-6:
        up = up / up_norm
    target = eye + forward * 10.0
    return eye, target, up


def _get_export_camera_pos(current_frame=None, total_frames=None):
    """Get camera position and forward for the current frame.
    Returns (position, forward) as np.ndarray[3], or (None, None) on failure.
    When keyframes exist the full transform is interpolated so both position
    and orientation follow the sequencer path exactly."""
    camera_pos = None
    camera_fwd = None

    # Live viewport fallback
    try:
        view = lf.get_current_view()
        if view is not None:
            camera_pos = np.array(view.position, dtype=np.float32)
            if hasattr(view, 'rotation'):
                rot = np.array(view.rotation.numpy())
                fwd = np.array([-rot[0][2], rot[1][2], rot[2][2]], dtype=np.float32)
                norm = np.linalg.norm(fwd)
                if norm > 1e-6:
                    camera_fwd = fwd / norm
    except Exception as e:
        _depth_log(f"CAM_SRC | get_current_view error: {e}")

    # Keyframe path override
    if current_frame is not None and total_frames is not None and total_frames > 0:
        transforms = _get_keyframe_transforms()
        _depth_log(f"CAM_KF | frame={current_frame}/{total_frames} keyframes={len(transforms)}")
        if transforms:
            m = _interpolate_keyframe_transform(transforms, current_frame, total_frames)
            if m is not None:
                eye, target, up = _camera_state_from_transform(m)
                camera_pos = eye.astype(np.float32)
                # Use live-viewport forward convention for depth shading
                # (only X negated) — separate from render_at target direction
                camera_fwd = np.array([-m[0,2], m[1,2], m[2,2]], dtype=np.float32)
                norm = np.linalg.norm(camera_fwd)
                if norm > 1e-6:
                    camera_fwd = camera_fwd / norm

    return camera_pos, camera_fwd


def apply_depthmap_colors(
    node_name: str,
    colormap: str = "jet",
    axis: Literal["x", "y", "z", "camera"] = "z",
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    range_only: bool = False,
    invert: bool = False,
    original_sh0: Optional[np.ndarray] = None,
    current_frame: Optional[int] = None,
    total_frames: Optional[int] = None,
) -> Tuple[bool, str]:
    """Apply depth-based colors to a splat node."""
    # Normalise axis aliases
    if axis in ("camera_parallel", "camera_radial"):
        axis = "camera"

    scene = lf.get_scene()
    if scene is None:
        return False, "No scene loaded"
    
    node = scene.get_node(node_name)
    if node is None:
        return False, f"Node '{node_name}' not found"
    
    splat = node.splat_data()
    if splat is None:
        pc = node.point_cloud()
        if pc is None:
            return False, f"Node '{node_name}' is not a splat or point cloud"
        
        positions = pc.means.numpy()
        camera_pos, camera_fwd = None, None
        if axis == "camera":
            camera_pos, camera_fwd = _get_export_camera_pos(current_frame, total_frames)
        
        normalized, mask, d_min, d_max = compute_depth_values(
            positions, axis, camera_pos, min_depth, max_depth, camera_fwd=camera_fwd
        )
        if invert:
            normalized = 1.0 - normalized
        cmap_fn = get_colormap(colormap)
        colors = cmap_fn(normalized)
        colors_tensor = lf.Tensor.from_numpy(colors.astype(np.float32))
        positions_tensor = lf.Tensor.from_numpy(positions.astype(np.float32))
        pc.set_data(positions_tensor, colors_tensor)
        return True, f"Applied {colormap} depth map (depth range: {d_min:.2f} - {d_max:.2f})"
    
    # --- Splat path ---
    sh0_raw = splat.sh0_raw
    if sh0_raw is None or sh0_raw.ndim == 0:
        return False, f"Node '{node_name}' has degenerate SH0 data (0-d tensor)"

    # Use combined_model positions (world space, matches pick_at_screen).
    # Fall back to splat positions if point count doesn't match sh0.
    combined = scene.combined_model()
    if combined is not None:
        positions = combined.get_means().numpy()
    else:
        positions = splat.get_means().numpy()
    
    if positions.shape[0] != sh0_raw.shape[0]:
        positions = splat.get_means().numpy()

    # Get camera position + forward vector
    camera_pos, camera_fwd = None, None
    if axis == "camera":
        camera_pos, camera_fwd = _get_export_camera_pos(current_frame, total_frames)
    
    # Compute depths
    normalized, mask, d_min, d_max = compute_depth_values(
        positions, axis, camera_pos, min_depth, max_depth, camera_fwd=camera_fwd
    )
    
    if invert:
        normalized = 1.0 - normalized
    
    cmap_fn = get_colormap(colormap)
    if colormap == "grayscale":
        colors = grayscale_colormap(normalized, invert=False)
    else:
        colors = cmap_fn(normalized)
    
    C0 = 0.28209479177387814
    
    if range_only and min_depth is not None and max_depth is not None:
        # Re-compute raw depths for range masking
        if axis == "x":
            depths = positions[:, 0]
        elif axis == "y":
            depths = positions[:, 1]
        elif axis == "z":
            depths = positions[:, 2]
        else:  # camera
            if camera_pos is None:
                camera_pos = np.array([0.0, 0.0, 0.0])
            if camera_fwd is not None:
                fwd = camera_fwd / (np.linalg.norm(camera_fwd) + 1e-12)
                depths = (positions - camera_pos).dot(fwd)
            else:
                depths = np.linalg.norm(positions - camera_pos, axis=1)
        
        range_lo = min(min_depth, max_depth)
        range_hi = max(min_depth, max_depth)
        in_range = (depths >= range_lo) & (depths <= range_hi)
        
        sh0_tensor = splat.sh0_raw
        base_sh0 = original_sh0.copy() if original_sh0 is not None else sh0_tensor.numpy().copy()
        new_sh0_colors = (colors - 0.5) / C0
        new_sh0_colors = new_sh0_colors.reshape(-1, 1, 3).astype(np.float32)
        base_sh0[in_range] = new_sh0_colors[in_range]
        sh0_tensor[:] = lf.Tensor.from_numpy(base_sh0).cuda()
    else:
        in_range = None
        sh0_colors = (colors - 0.5) / C0
        sh0_colors = sh0_colors.reshape(-1, 1, 3).astype(np.float32)
        sh0_tensor = splat.sh0_raw
        sh0_tensor[:] = lf.Tensor.from_numpy(sh0_colors).cuda()
    
    # For grayscale: zero out higher-order SH to remove view-dependent colour
    if colormap == "grayscale":
        shN_tensor = splat.shN_raw
        if shN_tensor is not None and shN_tensor.ndim > 0 and shN_tensor.shape[0] > 0:
            shN_np = shN_tensor.numpy().copy()
            if in_range is not None:
                shN_np[in_range] = 0.0
            else:
                shN_np[:] = 0.0
            shN_tensor[:] = lf.Tensor.from_numpy(shN_np.astype(np.float32)).cuda()
    
    scene = lf.get_scene()
    if scene:
        scene.notify_changed()
    lf.ui.request_redraw()
    
    return True, f"Applied {colormap} depth map (depth range: {d_min:.2f} - {d_max:.2f})"


def get_scene_bounds(node_name: Optional[str] = None) -> Optional[BoundingBox]:
    """Get the bounding box of a node or the entire scene."""
    scene = lf.get_scene()
    if scene is None:
        return None
    
    all_positions = []
    if node_name:
        node = scene.get_node(node_name)
        if node is None:
            return None
        nodes = [node]
    else:
        nodes = list(scene.get_nodes())
    
    for node in nodes:
        splat = node.splat_data()
        if splat is not None:
            all_positions.append(splat.get_means().numpy())
            continue
        pc = node.point_cloud()
        if pc is not None:
            all_positions.append(pc.means.numpy())
    
    if not all_positions:
        return None
    
    positions = np.concatenate(all_positions, axis=0)
    return BoundingBox(
        min_x=float(np.min(positions[:, 0])),
        max_x=float(np.max(positions[:, 0])),
        min_y=float(np.min(positions[:, 1])),
        max_y=float(np.max(positions[:, 1])),
        min_z=float(np.min(positions[:, 2])),
        max_z=float(np.max(positions[:, 2])),
    )
