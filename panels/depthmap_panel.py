# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth Map Visualization Panel — RML-driven."""

from __future__ import annotations
from pathlib import Path
import threading
import datetime
import os
import subprocess
import sys
import numpy as np
import lichtfeld as lf

from ..core.depthmap import (apply_depthmap_colors, _get_export_camera_pos,
                             _get_keyframe_transforms, _interpolate_keyframe_transform,
                             _camera_state_from_transform)

# ── Viewport export helpers ───────────────────────────────────────────────────
import re as _re

def _vp_parse_version(v: str) -> tuple:
    parts = v.lstrip("v").split(".")[:3]
    return tuple(int(_re.match(r"\d+", x).group()) for x in parts)

_VP_Y_UP = _vp_parse_version(lf.__version__) >= (0, 5, 1)

VP_RESOLUTIONS = [
    ("Viewport",  None),
    ("1080p HD",  1080),
    ("1440p 2K",  1440),
    ("4K",        2160),
    ("8K",        4320),
]

def _vp_get_view_params():
    view = lf.get_current_view()
    if view is None:
        return None
    cam_pos = np.array(view.translation.numpy()).flatten()
    rot = np.array(view.rotation.numpy())
    forward = -rot[:, 2] if rot.shape == (3, 3) else np.array([0, 0, -1])
    up_vec  =  rot[:, 1] if rot.shape == (3, 3) else np.array([0, 1,  0])
    eye    = tuple(cam_pos.tolist())
    target = tuple((cam_pos + forward * 10).tolist())
    up     = tuple(up_vec.tolist())
    fov    = float(view.fov_y)
    vp_w   = int(view.width)
    vp_h   = int(view.height)
    return eye, target, up, fov, vp_w, vp_h

def _vp_fix_orientation(arr, from_render_at):
    if from_render_at:
        arr = np.flip(arr, axis=0).copy()
    else:
        if _VP_Y_UP:
            arr = np.flip(arr, axis=0).copy()
    arr = np.flip(arr, axis=0).copy()
    return arr

def _vp_render_at_explicit(eye, target, up, fov, width, height):
    """render_at with explicit camera params — reshapes and orients correctly."""
    try:
        tensor = lf.render_at(eye, target, width, height, fov, up)
        if tensor is None:
            _depth_log(f"render_at_explicit: tensor is None (w={width} h={height})")
            return None
        arr = tensor.numpy().astype(np.float32)
        _depth_log(f"render_at_explicit: raw shape={arr.shape} w={width} h={height}")
        # render_at may return [H*W, 1, C] — reshape to [H, W, C]
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.reshape(height, width, arr.shape[2])
        elif arr.ndim == 2:
            arr = arr.reshape(height, width, -1)
        _depth_log(f"render_at_explicit: after reshape={arr.shape}")
        arr = _vp_fix_orientation(arr, from_render_at=True)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr
    except Exception as e:
        _depth_log(f"render_at_explicit error: {e}")
        return None
    params = _vp_get_view_params()
    if params is None:
        return None
    eye, target, up, fov, _, _ = params
    try:
        bg_tensor = None
        if bg_color is not None:
            for shape_fn in [
                lambda: np.full((height, width, 3), bg_color, dtype=np.float32),
                lambda: np.array(bg_color, dtype=np.float32),
                lambda: np.array([[bg_color]], dtype=np.float32),
            ]:
                try:
                    bg_np = shape_fn()
                    try:
                        bg_tensor = lf.Tensor.from_numpy(bg_np).cuda()
                    except Exception:
                        bg_tensor = lf.Tensor.from_numpy(bg_np)
                    break
                except Exception:
                    bg_tensor = None
        tensor = lf.render_at(eye, target, width, height, fov, up, bg_tensor)
        if tensor is None:
            return None
        arr = tensor.numpy().astype(np.float32)
        # render_at may return [H*W, 1, C] — reshape to [H, W, C]
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.reshape(height, width, arr.shape[2])
        elif arr.ndim == 2:
            arr = arr.reshape(height, width, -1)
        arr = _vp_fix_orientation(arr, from_render_at=True)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr
    except Exception as e:
        lf.log.error(f"[VPExport] render_at failed: {e}")
        return None

def _vp_capture_viewport_arr():
    vp = lf.capture_viewport()
    if vp is None or vp.image is None:
        return None
    arr = np.asarray(vp.image.cpu().contiguous(), dtype=np.float32)
    arr = _vp_fix_orientation(arr, from_render_at=False)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr

def _vp_capture_arr(target_h, bg_color=None):
    if target_h is None:
        return _vp_capture_viewport_arr()
    params = _vp_get_view_params()
    if params:
        _, _, _, _, vp_w, vp_h = params
        target_w = max(1, round(vp_w * target_h / max(vp_h, 1)))
    else:
        target_w = round(target_h * 16 / 9)
    arr = _vp_render_at_size(target_w, target_h, bg_color=bg_color)
    if arr is not None:
        return arr
    arr = _vp_capture_viewport_arr()
    if arr is None:
        return None
    from PIL import Image as _Image
    img = _Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), "RGB")
    img = img.resize((target_w, target_h), _Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0

def _vp_arr_to_image(arr):
    from PIL import Image as _Image
    rgb = (arr[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
    return _Image.fromarray(rgb, "RGB")

def _vp_bw2a(black_arr, white_arr):
    from PIL import Image as _Image
    b = (black_arr[..., :3] * 255.0).clip(0, 255)
    w = (white_arr[..., :3] * 255.0).clip(0, 255)
    diff = w - b
    alpha = 1.0 - (np.mean(diff, axis=2) / 255.0)
    alpha = np.clip(alpha, 0, 1)
    recovered = b / (alpha[:, :, np.newaxis] + 1e-10)
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    alpha_u8 = (alpha * 255).astype(np.uint8)
    rgba = np.dstack([recovered, alpha_u8])
    return _Image.fromarray(rgba, "RGBA")

# ── Depth log ─────────────────────────────────────────────────────────────────

DEPTH_LOG_PATH = r"u:\temp\DEPTH.TXT"

def _depth_log(msg: str):
    try:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(DEPTH_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

# ── Draw handler ──────────────────────────────────────────────────────────────

_draw_handler_registered = False
panel: "DepthmapPanel | None" = None

_last_cam_pos = None

def _depthmap_draw_handler(ctx):
    global panel, _last_cam_pos
    if panel is None:
        return

    # For camera-axis mode, detect viewport movement and mark dirty so live
    # preview re-applies depth from the new camera position every frame.
    if panel._enabled and panel._live_preview:
        axis = panel.AXIS_ITEMS[panel._axis_idx][0]
        if axis == "camera":
            params = _vp_get_view_params()
            if params:
                cam_pos = params[0]  # eye position tuple
                if _last_cam_pos is None or cam_pos != _last_cam_pos:
                    _last_cam_pos = cam_pos
                    panel._settings_dirty = True

    # Live preview — fires every frame exactly like the original draw() method.
    if panel._settings_dirty:
        if panel._enabled and panel._live_preview:
            panel._settings_dirty = False
            panel._apply_depthmap(silent=True)
        else:
            panel._settings_dirty = False

    # Render progress polling
    if panel._render_active:
        lf.ui.request_redraw()

    # Point markers
    if panel._point1_pos:
        ctx.draw_point_3d(panel._point1_pos, (0.0, 1.0, 0.0, 1.0), 16.0)
        screen = ctx.world_to_screen(panel._point1_pos)
        if screen:
            ctx.draw_circle_2d(screen, 12.0, (0.0, 1.0, 0.0, 1.0), 2.0)
            ctx.draw_text_2d(
                (screen[0] + 14, screen[1] - 8),
                f"P1: {panel._min_depth:.2f}" if panel._min_depth is not None else "P1",
                (0.0, 1.0, 0.0, 1.0),
            )
    if panel._point2_pos:
        ctx.draw_point_3d(panel._point2_pos, (1.0, 0.5, 0.0, 1.0), 16.0)
        screen = ctx.world_to_screen(panel._point2_pos)
        if screen:
            ctx.draw_circle_2d(screen, 12.0, (1.0, 0.5, 0.0, 1.0), 2.0)
            ctx.draw_text_2d(
                (screen[0] + 14, screen[1] - 8),
                f"P2: {panel._max_depth:.2f}" if panel._max_depth is not None else "P2",
                (1.0, 0.5, 0.0, 1.0),
            )
    if panel._picking_point > 0:
        ctx.draw_text_2d(
            (20, 50),
            f"PICK POINT {panel._picking_point}: Click on model  (ESC / Right-click to cancel)",
            (0.0, 1.0, 0.5, 0.95),
        )

def _ensure_draw_handler():
    global _draw_handler_registered
    if not _draw_handler_registered:
        try:
            lf.remove_draw_handler("depthmap_viz_overlay")
        except Exception:
            pass
        lf.add_draw_handler("depthmap_viz_overlay", _depthmap_draw_handler, "POST_VIEW")
        _draw_handler_registered = True

_frame_handler_registered = False

def _register_frame_handler():
    pass  # live preview handled via draw handler

def _unregister_frame_handler():
    _depth_log("UNREGISTER | (draw handler remains active)")


# ── Panel ─────────────────────────────────────────────────────────────────────

class DepthmapPanel(lf.ui.Panel):
    id                 = "depthmap_viz.panel"
    label              = "DoF"
    space              = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order              = 30
    template           = str(Path(__file__).resolve().with_name("depthmap_panel.rml"))
    height_mode        = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 150

    COLORMAP_ITEMS = [
        ("grayscale", "Grayscale"),
        ("jet",       "Jet (Rainbow)"),
        ("turbo",     "Turbo"),
        ("viridis",   "Viridis"),
    ]
    AXIS_ITEMS = [
        ("z",      "Z-Axis (Depth)"),
        ("y",      "Y-Axis (Height)"),
        ("x",      "X-Axis (Side)"),
        ("camera", "Camera - Parallel (Projective)"),
        ("camera", "Camera - Radial (Spherical)"),
    ]

    def __init__(self):
        global panel
        panel = self

        # Core state
        self._enabled          = False
        self._live_preview     = True
        self._colormap_idx     = 0
        self._axis_idx         = 0
        self._invert           = False
        self._saved_colors     = {}
        self._saved_shN        = {}
        self._current_target   = None
        self._depth_map_active = False

        # Range
        self._use_custom_range     = False
        self._use_selection_method = False
        self._min_depth            = None
        self._max_depth            = None
        self._range_only           = False
        self._point1_pos           = None
        self._point2_pos           = None
        self._captured_pos         = None
        self._captured_depth       = None
        self._picking_point        = 0

        # Status
        self._status_msg      = ""
        self._status_is_error = False

        # Render video
        self._render_output_dir   = "c:\\temp"
        self._render_total_frames = 300
        self._render_fps          = 30
        self._render_active       = False
        self._render_cancel       = False
        self._render_progress     = 0.0
        self._render_status       = ""
        # Frame-pump state (driven from on_update on the main thread)
        self._render_frame_idx    = 0
        self._render_frame_paths  = []
        self._render_node_name    = ""
        self._render_rgb_pass     = False  # True when doing the RGB second pass
        self._render_rgb_paths    = []

        # Viewport export
        self._vp_export_resolution_idx = 0
        self._vp_export_compress       = 3
        self._vp_export_transparency   = False
        self._vp_export_status         = ""
        self._vp_export_status_ok      = True
        self._vp_bw2a_state            = {"step": 0}

        self._settings_dirty   = False  # set by any handler; on_update applies if live
        self._video_expanded   = True
        self._export_expanded  = True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def on_bind_model(self, ctx):
        _ensure_draw_handler()
        model = ctx.create_data_model("depthmap_panel")

        # ── Target / enable ──────────────────────────────────────────────
        model.bind_func("target_name",    lambda: self._get_selected_splat_name() or "")
        model.bind_func("no_splat",       lambda: self._get_selected_splat_name() is None)
        model.bind_func("has_splat",      lambda: self._get_selected_splat_name() is not None)
        model.bind_func("enabled",        lambda: self._enabled)
        model.bind_func("disabled",       lambda: not self._enabled)
        model.bind_event("toggle_enable", self._on_toggle_enable)
        model.bind_func("live_preview",     lambda: self._live_preview)
        model.bind_func("not_live_preview", lambda: not self._live_preview)
        model.bind_event("toggle_live",   self._on_toggle_live)

        # ── Colormap ─────────────────────────────────────────────────────
        model.bind_func("colormap_label", lambda: self.COLORMAP_ITEMS[self._colormap_idx][1])
        for i, (key, label) in enumerate(self.COLORMAP_ITEMS):
            idx = i
            model.bind_event(f"set_cmap_{i}", lambda h, e, a, i=idx: self._on_set_colormap(i))
            model.bind_func(f"cmap_{i}_active", lambda i=idx: self._colormap_idx == i)
        model.bind_func("invert",          lambda: self._invert)
        model.bind_event("toggle_invert",  self._on_toggle_invert)

        # ── Axis ─────────────────────────────────────────────────────────
        model.bind_func("axis_label", lambda: self.AXIS_ITEMS[self._axis_idx][1])
        for i, (key, label) in enumerate(self.AXIS_ITEMS):
            idx = i
            model.bind_event(f"set_axis_{i}", lambda h, e, a, i=idx: self._on_set_axis(i))
            model.bind_func(f"axis_{i}_active", lambda i=idx: self._axis_idx == i)

        # ── Depth Range ───────────────────────────────────────────────────
        model.bind_func("point1_set",   lambda: self._point1_pos is not None)
        model.bind_func("point1_unset", lambda: self._point1_pos is None)
        model.bind_func("point1_label", lambda: f"Point 1 (Min): {self._min_depth:.2f}" if self._point1_pos else "Point 1 (Min): Not set")
        model.bind_func("point2_set",   lambda: self._point2_pos is not None)
        model.bind_func("point2_unset", lambda: self._point2_pos is None)
        model.bind_func("point2_label", lambda: f"Point 2 (Max): {self._max_depth:.2f}" if self._point2_pos else "Point 2 (Max): Not set")

        model.bind_func("picking_p1",     lambda: self._picking_point == 1)
        model.bind_func("not_picking_p1", lambda: self._picking_point != 1)
        model.bind_func("picking_p2",     lambda: self._picking_point == 2)
        model.bind_func("not_picking_p2", lambda: self._picking_point != 2)
        model.bind_event("pick_p1",       self._on_pick_p1)
        model.bind_event("pick_p2",       self._on_pick_p2)
        model.bind_event("stop_pick",     self._on_stop_pick)

        model.bind_func("has_points",     lambda: self._point1_pos is not None or self._point2_pos is not None)
        model.bind_event("clear_points",  self._on_clear_points)
        model.bind_func("has_range",      lambda: self._use_custom_range and self._min_depth is not None)

        model.bind_func("min_depth_str",  lambda: f"{self._min_depth:.2f}" if self._min_depth is not None else "—")
        model.bind_func("max_depth_str",  lambda: f"{self._max_depth:.2f}" if self._max_depth is not None else "—")
        model.bind("min_depth_val",
                   lambda: f"{self._min_depth:.2f}" if self._min_depth is not None else "0.00",
                   self._on_set_min_depth)
        model.bind("max_depth_val",
                   lambda: f"{self._max_depth:.2f}" if self._max_depth is not None else "0.00",
                   self._on_set_max_depth)

        model.bind_event("min_sub1",   lambda h, e, a: self._nudge_depth("min", -1.0))
        model.bind_event("min_add1",   lambda h, e, a: self._nudge_depth("min", +1.0))
        model.bind_event("min_sub01",  lambda h, e, a: self._nudge_depth("min", -0.1))
        model.bind_event("min_add01",  lambda h, e, a: self._nudge_depth("min", +0.1))
        model.bind_event("min_sub001", lambda h, e, a: self._nudge_depth("min", -0.01))
        model.bind_event("min_add001", lambda h, e, a: self._nudge_depth("min", +0.01))
        model.bind_event("max_sub1",   lambda h, e, a: self._nudge_depth("max", -1.0))
        model.bind_event("max_add1",   lambda h, e, a: self._nudge_depth("max", +1.0))
        model.bind_event("max_sub01",  lambda h, e, a: self._nudge_depth("max", -0.1))
        model.bind_event("max_add01",  lambda h, e, a: self._nudge_depth("max", +0.1))
        model.bind_event("max_sub001", lambda h, e, a: self._nudge_depth("max", -0.01))
        model.bind_event("max_add001", lambda h, e, a: self._nudge_depth("max", +0.01))
        model.bind_event("swap_range", self._on_swap_range)
        model.bind_func("min_gt_max",  lambda: (self._min_depth or 0) > (self._max_depth or 0))
        model.bind_func("range_only",  lambda: self._range_only)
        model.bind_event("toggle_range_only", self._on_toggle_range_only)

        # ── Apply / Restore ───────────────────────────────────────────────
        model.bind_event("apply_depthmap",   self._on_apply)
        model.bind_event("update_depthmap",  self._on_update)
        model.bind_event("restore_original", self._on_restore)

        # ── Presets ───────────────────────────────────────────────────────
        model.bind_event("preset_gray_z",   lambda h, e, a: self._apply_preset(0, 0))
        model.bind_event("preset_jet_z",  lambda h, e, a: self._apply_preset(1, 0))
        model.bind_event("preset_turbo_z", lambda h, e, a: self._apply_preset(2, 0))
        model.bind_event("preset_cam_p",   lambda h, e, a: self._apply_preset(0, 3))
        model.bind_event("preset_cam_o",   lambda h, e, a: self._apply_preset(0, 4))

        # ── Status ────────────────────────────────────────────────────────
        model.bind_func("status_msg",     lambda: self._status_msg)
        model.bind_func("status_ok",      lambda: not self._status_is_error and bool(self._status_msg))
        model.bind_func("status_err",     lambda: self._status_is_error and bool(self._status_msg))
        model.bind_func("has_status",     lambda: bool(self._status_msg))

        # ── Render Video ──────────────────────────────────────────────────
        model.bind("render_output_dir",
                   lambda: self._render_output_dir,
                   lambda v: setattr(self, "_render_output_dir", v))
        model.bind("render_frames_str",
                   lambda: str(self._render_total_frames),
                   self._on_set_render_frames)
        model.bind("render_fps_str",
                   lambda: str(self._render_fps),
                   self._on_set_render_fps)
        model.bind_func("render_duration_str", lambda: f"{self._render_total_frames / max(self._render_fps,1):.1f}s  ({self._render_total_frames} frames @ {self._render_fps}fps)")
        model.bind_func("render_active",       lambda: self._render_active)
        model.bind_func("render_idle",         lambda: not self._render_active)
        model.bind_func("render_progress",     lambda: f"{self._render_progress:.2f}")
        model.bind_func("render_progress_pct", lambda: f"{self._render_progress * 100:.0f}%")
        model.bind_func("render_status",       lambda: self._render_status)
        model.bind_func("render_status_ok",    lambda: bool(self._render_status) and "ERROR" not in self._render_status)
        model.bind_func("render_status_err",   lambda: "ERROR" in self._render_status)
        model.bind_event("do_render_video",    self._on_render_video)
        model.bind_event("cancel_render",      self._on_cancel_render)

        # ── Export PNG ────────────────────────────────────────────────────
        res_labels_str = "|".join(r[0] for r in VP_RESOLUTIONS)
        model.bind_func("vp_res_labels",    lambda: res_labels_str)
        model.bind_func("vp_res_idx",       lambda: str(self._vp_export_resolution_idx))
        for i, (label, _) in enumerate(VP_RESOLUTIONS):
            idx = i
            model.bind_event(f"set_vp_res_{i}", lambda h, e, a, i=idx: self._on_set_vp_res(i))
            model.bind_func(f"vp_res_{i}_active", lambda i=idx: self._vp_export_resolution_idx == i)
        model.bind_func("vp_size_hint",     self._vp_size_hint)
        model.bind("vp_compress_str",
                   lambda: str(self._vp_export_compress),
                   self._on_set_vp_compress)
        model.bind_func("vp_transparency",  lambda: self._vp_export_transparency)
        model.bind_event("toggle_vp_transp", self._on_toggle_vp_transp)
        model.bind_func("vp_export_label",  lambda: f"Export {VP_RESOLUTIONS[self._vp_export_resolution_idx][0]} PNG")
        model.bind_event("do_vp_export",    self._on_vp_export)
        model.bind_event("open_dof",              self._on_open_dof)
        model.bind_event("open_dof_still",        self._on_open_dof_still)
        model.bind_event("open_dof_export_still",   self._on_vp_export)
        model.bind_event("open_dof_video",        self._on_open_dof_video)
        model.bind_event("browse_output",         self._on_browse_output)
        model.bind_event("run_still_bat",         self._on_run_still_bat)
        model.bind_event("open_output",           self._on_open_output)
        model.bind_event("toggle_video_section",  self._on_toggle_video_section)
        model.bind_event("toggle_export_section", self._on_toggle_export_section)
        model.bind_func("video_collapsed",  lambda: not self._video_expanded)
        model.bind_func("video_expanded",   lambda: self._video_expanded)
        model.bind_func("export_collapsed", lambda: not self._export_expanded)
        model.bind_func("export_expanded",  lambda: self._export_expanded)
        model.bind_func("vp_status",        lambda: self._vp_export_status)
        model.bind_func("vp_status_ok",     lambda: bool(self._vp_export_status) and self._vp_export_status_ok)
        model.bind_func("vp_status_err",    lambda: bool(self._vp_export_status) and not self._vp_export_status_ok)

        self._handle = model.get_handle()
        self._dirty_all()

    def on_update(self, doc):
        from ..operators.point_picker import was_pick_cancelled, consume_pending_pick

        # Consume any pick result deposited by the modal operator this frame.
        pending = consume_pending_pick()
        if pending is not None:
            world_pos, point_num = pending
            self._on_pick_result(world_pos, point_num)

        # Poll for ESC/cancel from the modal operator.
        if was_pick_cancelled() and self._picking_point > 0:
            self._picking_point = 0
            self._status_msg = "Picking cancelled"
            self._status_is_error = False

        # Drive the video render pump — one frame per update tick.
        if self._render_active:
            self._render_pump_frame()

        self._dirty("target_name", "has_splat", "no_splat",
                    "point1_label", "point2_label", "point1_set", "point1_unset",
                    "point2_set", "point2_unset",
                    "picking_p1", "not_picking_p1", "picking_p2", "not_picking_p2",
                    "status_msg", "has_status", "status_ok", "status_err",
                    "render_active", "render_idle", "render_progress", "render_progress_pct", "render_status",
                    "render_status_ok", "render_status_err",
                    "vp_status", "vp_status_ok", "vp_status_err",
                    "min_gt_max",
                    "live_preview", "not_live_preview",
                    "video_collapsed", "video_expanded",
                    "export_collapsed", "export_expanded",
                    )
        return True

    def on_unmount(self, doc):
        doc.remove_data_model("depthmap_panel")

    def _dirty(self, *keys):
        if self._handle:
            for k in keys:
                self._handle.dirty(k)

    def _dirty_all(self):
        if self._handle:
            self._handle.dirty_all()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_selected_splat_name(self):
        scene = lf.get_scene()
        if not scene:
            return None
        selected = lf.get_selected_node_names()
        for name in selected:
            node = scene.get_node(name)
            if node and node.splat_data() is not None:
                return name
        for node in scene.get_nodes():
            if node.splat_data() is not None:
                return node.name
        return None

    def _save_original_colors(self, node_name, force=False):
        if not force and node_name in self._saved_colors:
            return True
        scene = lf.get_scene()
        if not scene:
            return False
        node = scene.get_node(node_name)
        if not node:
            return False
        splat = node.splat_data()
        if splat:
            self._saved_colors[node_name] = splat.sh0_raw.clone().cpu()
            shN = splat.shN_raw
            if shN is not None and shN.shape[0] > 0:
                self._saved_shN[node_name] = shN.clone().cpu()
            self._current_target = node_name
            return True
        return False

    def _restore_original_colors(self, silent=False):
        node_name = self._current_target or self._get_selected_splat_name()
        if not node_name or node_name not in self._saved_colors:
            if not silent:
                self._status_msg = "No saved colors to restore"
                self._status_is_error = True
            return False
        scene = lf.get_scene()
        if not scene:
            return False
        node = scene.get_node(node_name)
        if not node:
            return False
        splat = node.splat_data()
        if splat:
            try:
                saved = self._saved_colors[node_name].cuda()
                sh0_tensor = splat.sh0_raw
                sh0_tensor[:] = saved
                if node_name in self._saved_shN:
                    saved_shN = self._saved_shN[node_name].cuda()
                    shN_tensor = splat.shN_raw
                    if shN_tensor is not None:
                        shN_tensor[:] = saved_shN
                scene.notify_changed()
                lf.ui.request_redraw()
                self._depth_map_active = False
                if not silent:
                    self._status_msg = "Restored original colors"
                    self._status_is_error = False
                return True
            except Exception as e:
                if not silent:
                    self._status_msg = f"Restore error: {e}"
                    self._status_is_error = True
        return False

    def _get_depth_from_position(self, pos):
        axis = self.AXIS_ITEMS[self._axis_idx][0]
        if axis == "x":
            return pos[0]
        elif axis == "y":
            return pos[1]
        elif axis == "z":
            return pos[2]
        elif axis == "camera":
            view = lf.get_current_view()
            if view:
                cam_pos = np.array(view.translation.numpy()).flatten()
                return float(np.linalg.norm(np.array(pos) - cam_pos))
        return pos[2]

    def _apply_depthmap(self, silent=False):
        node_name = self._get_selected_splat_name()
        if not node_name:
            if not silent:
                self._status_msg = "No splat selected"
                self._status_is_error = True
            return False
        axis = self.AXIS_ITEMS[self._axis_idx][0]
        colormap = self.COLORMAP_ITEMS[self._colormap_idx][0]
        min_d = self._min_depth if self._use_custom_range else None
        max_d = self._max_depth if self._use_custom_range else None
        original_sh0 = self._saved_colors.get(node_name)
        original_sh0_np = original_sh0.numpy() if original_sh0 is not None else None
        try:
            ok, msg = apply_depthmap_colors(
                node_name=node_name,
                colormap=colormap,
                axis=axis,
                min_depth=min_d,
                max_depth=max_d,
                range_only=self._range_only,
                invert=self._invert,
                original_sh0=original_sh0_np,
            )
            self._depth_map_active = ok
            if not silent:
                self._status_msg = msg
                self._status_is_error = not ok
            return ok
        except Exception as e:
            if not silent:
                self._status_msg = f"Error: {e}"
                self._status_is_error = True
            _depth_log(f"APPLY ERROR: {e}")
            return False

    def _apply_preset(self, cmap_idx, axis_idx):
        self._colormap_idx = cmap_idx
        self._axis_idx = axis_idx
        node_name = self._get_selected_splat_name()
        if node_name and not self._enabled:
            self._save_original_colors(node_name, force=True)
            self._enabled = True
        self._apply_depthmap()
        self._dirty_all()

    def _start_picking(self, point_num):
        self._picking_point = point_num
        self._status_msg = f"Click on model to pick Point {point_num}..."
        self._status_is_error = False
        from ..operators.point_picker import set_pick_callback
        set_pick_callback(self._on_pick_result, point_num)
        try:
            lf.ui.ops.invoke(
                "lfs_plugins.DoF.operators.point_picker.DEPTHMAP_OT_pick_point"
            )
        except Exception as e:
            _depth_log(f"PICK start error: {e}")

    def _cancel_picking(self):
        self._picking_point = 0
        from ..operators.point_picker import clear_pick_callback
        clear_pick_callback()
        lf.ui.ops.cancel_modal()
        self._status_msg = "Picking cancelled"
        self._status_is_error = False

    def _on_pick_result(self, world_pos, point_num):
        """Called directly by the modal operator on each click."""
        self._captured_pos = world_pos
        self._captured_depth = self._get_depth_from_position(world_pos)
        if point_num == 1:
            self._point1_pos = world_pos
            self._min_depth = self._captured_depth
            self._status_msg = f"Point 1: {self._min_depth:.2f}"
        else:
            self._point2_pos = world_pos
            self._max_depth = self._captured_depth
            self._status_msg = f"Point 2: {self._max_depth:.2f}"
        self._use_custom_range = True
        self._status_is_error = False
        node_name = self._get_selected_splat_name()
        if node_name:
            if not self._enabled:
                self._save_original_colors(node_name, force=True)
                self._enabled = True
            self._apply_depthmap(silent=False)
        self._dirty_all()

    def _nudge_depth(self, which, delta):
        if which == "min":
            self._min_depth = (self._min_depth or 0.0) + delta
        else:
            self._max_depth = (self._max_depth or 0.0) + delta
        self._use_custom_range = True
        self._settings_dirty = True
        self._dirty("min_depth_val", "max_depth_val", "min_depth_str", "max_depth_str", "min_gt_max")

    def _vp_size_hint(self):
        _, target_h = VP_RESOLUTIONS[self._vp_export_resolution_idx]
        if target_h is None:
            params = _vp_get_view_params()
            if params:
                _, _, _, _, w, h = params
                return f"Viewport: {w} × {h} px"
            return "Native viewport"
        params = _vp_get_view_params()
        if params:
            _, _, _, _, vp_w, vp_h = params
            out_w = max(1, round(vp_w * target_h / max(vp_h, 1)))
            return f"Output: {out_w} × {target_h} px"
        return f"Height: {target_h} px"

    def _vp_out_size(self, target_h):
        params = _vp_get_view_params()
        if params:
            _, _, _, _, vp_w, vp_h = params
            return max(1, round(vp_w * target_h / max(vp_h, 1))), target_h
        return round(target_h * 16 / 9), target_h

    def _vp_set_status(self, msg, *, success=False, warning=False, error=False):
        self._vp_export_status = msg
        self._vp_export_status_ok = not error
        self._dirty("vp_status", "vp_status_ok", "vp_status_err")

    def _do_viewport_export_png(self):
        from pathlib import Path as _Path
        _, target_h = VP_RESOLUTIONS[self._vp_export_resolution_idx]
        res_label = VP_RESOLUTIONS[self._vp_export_resolution_idx][0]
        default_name = f"depth_{res_label.lower().replace(' ', '_')}.png"
        path = None
        if sys.platform == "win32":
            ps_script = f'''
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.SaveFileDialog
$d.Title = "Save Viewport Depth as PNG"
$d.Filter = "PNG Image (*.png)|*.png"
$d.FileName = "{default_name}"
$d.DefaultExt = "png"
if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ Write-Output $d.FileName }}
'''
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                path = result.stdout.strip() or None
            except Exception:
                pass
        if not path:
            path = str(_Path(os.getcwd()) / default_name)
        if not path.lower().endswith(".png"):
            path += ".png"

        if self._vp_export_transparency:
            if target_h is not None:
                try:
                    self._vp_set_status("Rendering black bg...", warning=True)
                    black = _vp_render_at_size(*self._vp_out_size(target_h), bg_color=(0.0, 0.0, 0.0))
                    self._vp_set_status("Rendering white bg...", warning=True)
                    white = _vp_render_at_size(*self._vp_out_size(target_h), bg_color=(1.0, 1.0, 1.0))
                    if black is None or white is None:
                        self._vp_set_status("BW2A render failed.", error=True)
                        return
                    rgba_img = _vp_bw2a(black, white)
                    rgba_img.save(path, "PNG", compress_level=self._vp_export_compress)
                    h, w = black.shape[:2]
                    self._vp_set_status(f"Saved {w}×{h} RGBA: {path}", success=True)
                except Exception as e:
                    self._vp_set_status(f"Error: {e}", error=True)
            else:
                self._vp_bw2a_state.update({
                    "step": 1, "black": None, "white": None,
                    "orig_bg": None, "out_path": path,
                })
                self._vp_set_status("Starting BW2A capture...", warning=True)
                lf.add_draw_handler("depthmap.vp_bw2a", self._vp_bw2a_draw_handler)
        else:
            try:
                self._vp_set_status(f"Rendering {res_label}...", warning=True)
                arr = _vp_capture_arr(target_h)
                if arr is None:
                    self._vp_set_status("Capture failed.", error=True)
                    return
                img = _vp_arr_to_image(arr)
                img.save(path, "PNG", compress_level=self._vp_export_compress)
                h, w = arr.shape[:2]
                self._vp_set_status(f"Saved {w}×{h}: {path}", success=True)
            except Exception as e:
                self._vp_set_status(f"Error: {e}", error=True)

    def _vp_bw2a_draw_handler(self, context):
        rs = lf.get_render_settings()
        s = self._vp_bw2a_state
        if s["step"] == 1:
            s["orig_bg"] = rs.background_color
            rs.background_color = (0.0, 0.0, 0.0)
            s["step"] = 2
        elif s["step"] == 2:
            arr = _vp_capture_viewport_arr()
            if arr is None:
                self._vp_set_status("Capture failed (black bg).", error=True)
                rs.background_color = s["orig_bg"]
                s["step"] = 0
                lf.remove_draw_handler("depthmap.vp_bw2a")
                return
            s["black"] = arr
            rs.background_color = (1.0, 1.0, 1.0)
            s["step"] = 3
        elif s["step"] == 3:
            arr = _vp_capture_viewport_arr()
            if arr is None:
                self._vp_set_status("Capture failed (white bg).", error=True)
                rs.background_color = s["orig_bg"]
                s["step"] = 0
                lf.remove_draw_handler("depthmap.vp_bw2a")
                return
            s["white"] = arr
            rs.background_color = s["orig_bg"]
            s["step"] = 0
            lf.remove_draw_handler("depthmap.vp_bw2a")
            try:
                rgba_img = _vp_bw2a(s["black"], s["white"])
                rgba_img.save(s["out_path"], "PNG", compress_level=self._vp_export_compress)
                h, w = s["black"].shape[:2]
                self._vp_set_status(f"Saved {w}×{h} RGBA: {s['out_path']}", success=True)
            except Exception as e:
                self._vp_set_status(f"Error: {e}", error=True)

    def _render_depth_video(self, output_dir, total_frames, fps):
        """Arm the per-frame render pump.  Actual work is driven one frame per
        on_update() tick on the main thread, so apply_depthmap_colors +
        notify_changed are fully flushed before render_at is called."""
        node_name = self._get_selected_splat_name()
        if not node_name:
            self._render_status = "ERROR: No splat selected"
            self._dirty("render_status", "render_status_err")
            return

        os.makedirs(output_dir, exist_ok=True)

        self._render_active      = True
        self._render_cancel      = False
        self._render_progress    = 0.0
        self._render_status      = "Starting..."
        self._render_frame_idx   = 0
        self._render_frame_paths = []
        self._render_rgb_paths   = []
        self._render_rgb_pass    = False
        self._render_node_name   = node_name

    def _render_pump_frame(self):
        """Called by on_update() once per tick while a render is active.
        Applies depth, notifies scene, then immediately renders — all on the
        main thread so the GPU flush between apply and render_at is guaranteed."""
        i            = self._render_frame_idx
        total_frames = self._render_total_frames
        output_dir   = self._render_output_dir
        node_name    = self._render_node_name

        if self._render_cancel:
            self._render_status  = "Cancelled"
            self._render_active  = False
            return

        # ── 1. Update depth colors for this frame ────────────────────────
        axis  = self.AXIS_ITEMS[self._axis_idx][0]
        cmap  = self.COLORMAP_ITEMS[self._colormap_idx][0]
        min_d = self._min_depth if self._use_custom_range else None
        max_d = self._max_depth if self._use_custom_range else None

        if self._render_rgb_pass:
            # RGB pass — restore original colours (camera still moves for depth calc)
            try:
                self._restore_original_colors(silent=True)
                scene = lf.get_scene()
                if scene: scene.notify_changed()
            except Exception as e:
                _depth_log(f"RENDER PUMP RGB restore ERROR frame {i}: {e}")
                self._render_status = f"ERROR rgb frame {i}: {e}"
                self._render_active = False
                return
        else:
            # Depth pass — apply depthmap colours
            try:
                apply_depthmap_colors(
                    node_name=node_name, colormap=cmap, axis=axis,
                    min_depth=min_d, max_depth=max_d,
                    invert=self._invert,
                    current_frame=i, total_frames=total_frames,
                )
                scene = lf.get_scene()
                if scene: scene.notify_changed()
            except Exception as e:
                _depth_log(f"RENDER PUMP ERROR frame {i}: {e}")
                self._render_status = f"ERROR frame {i}: {e}"
                self._render_active = False
                return

        # ── 2. Render this frame from the keyframe camera position ────────
        params = _vp_get_view_params()
        if params:
            vp_eye, vp_target, vp_up, fov, vp_w, vp_h = params

            # Get full interpolated transform — eye, target AND up from keyframe
            transforms = _get_keyframe_transforms()
            if transforms:
                m = _interpolate_keyframe_transform(transforms, i, total_frames)
                if m is not None:
                    eye, target, up = _camera_state_from_transform(m)
                    eye    = tuple(eye.tolist())
                    target = tuple(target.tolist())
                    up     = tuple(up.tolist())
                else:
                    eye, target, up = vp_eye, vp_target, vp_up
            else:
                eye, target, up = vp_eye, vp_target, vp_up

            # Use explicit camera params — _vp_render_at_size uses live viewport
            arr = _vp_render_at_explicit(eye, target, up, fov, vp_w, vp_h)
            if arr is not None:
                        rgb_dir = os.path.join(output_dir, "RGB")
                        os.makedirs(rgb_dir, exist_ok=True)
            if arr is not None:
                from PIL import Image as _Image
                if self._render_rgb_pass:
                    sub_dir = os.path.join(output_dir, "RGB")
                else:
                    sub_dir = os.path.join(output_dir, "DGS")
                os.makedirs(sub_dir, exist_ok=True)
                frame_path = os.path.join(sub_dir, f"frame_{i:04d}.png")
                _Image.fromarray(
                    (arr[..., :3] * 255).clip(0, 255).astype(np.uint8)
                ).save(frame_path)
                if self._render_rgb_pass:
                    self._render_rgb_paths.append(frame_path)
                else:
                    self._render_frame_paths.append(frame_path)

        # ── 3. Advance or finish ──────────────────────────────────────────
        self._render_frame_idx  = i + 1
        self._render_progress   = (i + 1) / total_frames
        if self._render_rgb_pass:
            self._render_status = f"RGB frame {i+1}/{total_frames}"
        else:
            self._render_status = f"Frame {i+1}/{total_frames}"

        if self._render_frame_idx >= total_frames:
            if self._render_rgb_pass:
                # Both passes done — encode both videos
                self._render_active    = False
                self._render_progress  = 1.0
                self._render_rgb_pass  = False
                # Re-apply depthmap so viewport is restored to depth view
                self._apply_depthmap(silent=True)
                self._finish_render_video()
            else:
                # Depth pass done — start RGB pass
                self._render_frame_idx  = 0
                self._render_rgb_paths  = []
                self._render_rgb_pass   = True
                self._render_status     = "Starting RGB pass..."

    def _finish_render_video(self):
        """Encode depth + RGB frame sequences to mp4 via PyAV."""
        import threading as _threading

        depth_paths = self._render_frame_paths
        rgb_paths   = self._render_rgb_paths
        output_dir  = self._render_output_dir
        fps         = self._render_fps

        if not depth_paths:
            self._render_status = "No frames rendered"
            return

        def _encode_pass(frame_paths, video_path, label):
            try:
                import av
                from PIL import Image as _Image
                first = _Image.open(frame_paths[0]).convert("RGB")
                w, h  = first.size

                # libx264 DLL fails at runtime on some builds — try in order
                codecs = ["h264_nvenc", "h264_amf", "h264_qsv",
                          "libopenh264", "libx264", "mpeg4"]
                stream    = None
                container = None
                used      = None
                for codec in codecs:
                    try:
                        if container:
                            try: container.close()
                            except: pass
                        container = av.open(video_path, mode="w")
                        stream = container.add_stream(codec, rate=fps)
                        stream.width   = w
                        stream.height  = h
                        stream.pix_fmt = "yuv420p"
                        if codec not in ("libopenh264", "h264_nvenc",
                                         "h264_amf", "h264_qsv"):
                            stream.bit_rate = 8_000_000
                        # Probe: encode one blank frame
                        probe = av.VideoFrame(w, h, "yuv420p")
                        probe.pts = 0
                        list(stream.encode(probe))
                        used = codec
                        _depth_log(f"ENCODE | codec={codec} OK")
                        break
                    except Exception as ce:
                        _depth_log(f"ENCODE | codec={codec} failed: {ce}")
                        stream = None

                if used is None:
                    return "ERROR: no working video codec found"

                for idx, path in enumerate(frame_paths):
                    img   = _Image.open(path).convert("RGB")
                    frame = av.VideoFrame.from_image(img)
                    frame.pts = idx + 1  # +1 because probe used 0
                    for pkt in stream.encode(frame):
                        container.mux(pkt)
                    if idx % 10 == 0:
                        self._render_status = f"Encoding {label} {idx+1}/{len(frame_paths)}..."
                for pkt in stream.encode():
                    container.mux(pkt)
                container.close()
                return video_path
            except ImportError:
                return "ERROR: PyAV not installed"
            except Exception as e:
                return f"ERROR: {e}"

        def _encode():
            depth_out = os.path.join(output_dir, "depth_video.mp4")
            rgb_out   = os.path.join(output_dir, "rgb_video.mp4")

            result_d = _encode_pass(depth_paths, depth_out, "Depth")
            if rgb_paths:
                result_r = _encode_pass(rgb_paths, rgb_out, "RGB")
                if "ERROR" in str(result_d) or "ERROR" in str(result_r):
                    self._render_status = f"{result_d} | {result_r}"
                else:
                    self._render_status = f"Done: depth_video.mp4 + rgb_video.mp4"
            else:
                self._render_status = f"Saved: {depth_out}" if "ERROR" not in str(result_d) else result_d

        _threading.Thread(target=_encode, daemon=True).start()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_toggle_enable(self, h, e, a):
        node_name = self._get_selected_splat_name()
        if not node_name:
            return
        self._enabled = not self._enabled
        if self._enabled:
            self._save_original_colors(node_name, force=True)
            self._apply_depthmap(silent=True)
        else:
            self._restore_original_colors(silent=True)
        self._dirty("enabled", "disabled", "live_preview", "not_live_preview")

    def _on_toggle_live(self, h, e, a):
        self._live_preview = not self._live_preview
        self._dirty("live_preview", "not_live_preview")
        lf.ui.request_redraw()

    def _on_set_colormap(self, idx):
        self._colormap_idx = idx
        self._settings_dirty = True
        self._dirty("colormap_label", *[f"cmap_{i}_active" for i in range(len(self.COLORMAP_ITEMS))])

    def _on_toggle_invert(self, h, e, a):
        self._invert = not self._invert
        self._settings_dirty = True
        self._dirty("invert")

    def _on_set_axis(self, idx):
        self._axis_idx = idx
        # Cam P (idx 3) and Cam O (idx 4) both default to greyscale
        if idx in (3, 4):
            self._colormap_idx = 1  # grayscale
            self._dirty("colormap_label", *[f"cmap_{i}_active" for i in range(len(self.COLORMAP_ITEMS))])
        self._settings_dirty = True
        self._dirty("axis_label", *[f"axis_{i}_active" for i in range(len(self.AXIS_ITEMS))])

    def _on_pick_p1(self, h, e, a):
        self._start_picking(1)

    def _on_pick_p2(self, h, e, a):
        self._start_picking(2)

    def _on_stop_pick(self, h, e, a):
        self._picking_point = 0
        from ..operators.point_picker import clear_pick_callback
        clear_pick_callback()
        lf.ui.ops.cancel_modal()
        self._status_msg = "Picking cancelled"
        self._status_is_error = False
        self._dirty("picking_p1", "not_picking_p1", "picking_p2", "not_picking_p2", "status_msg", "has_status")

    def _on_clear_points(self, h, e, a):
        self._point1_pos = None
        self._point2_pos = None
        self._use_custom_range = False
        self._status_msg = "Points cleared"
        self._status_is_error = False
        self._picking_point = 0
        self._settings_dirty = True
        self._dirty_all()

    def _on_set_min_depth(self, v):
        try:
            self._min_depth = float(v)
            self._use_custom_range = True
            if self._enabled and self._live_preview:
                self._apply_depthmap(silent=True)
            self._dirty("min_depth_str", "min_gt_max")
        except ValueError:
            pass

    def _on_set_max_depth(self, v):
        try:
            self._max_depth = float(v)
            self._use_custom_range = True
            if self._enabled and self._live_preview:
                self._apply_depthmap(silent=True)
            self._dirty("max_depth_str", "min_gt_max")
        except ValueError:
            pass

    def _on_swap_range(self, h, e, a):
        self._min_depth, self._max_depth = self._max_depth, self._min_depth
        self._settings_dirty = True
        self._dirty("min_depth_val", "max_depth_val", "min_depth_str", "max_depth_str", "min_gt_max")

    def _on_toggle_range_only(self, h, e, a):
        self._range_only = not self._range_only
        self._settings_dirty = True
        self._dirty("range_only")

    def _on_apply(self, h, e, a):
        node_name = self._get_selected_splat_name()
        if node_name:
            self._save_original_colors(node_name, force=True)
            self._apply_depthmap()
            self._enabled = True
        self._dirty_all()

    def _on_update(self, h, e, a):
        self._apply_depthmap()

    def _on_restore(self, h, e, a):
        self._restore_original_colors()
        self._enabled = False
        self._dirty_all()

    def _on_render_video(self, h, e, a):
        node_name = self._get_selected_splat_name()
        if not node_name:
            self._render_status = "ERROR: Select a splat first"
            self._dirty("render_status", "render_status_err")
            return
        self._render_depth_video(
            self._render_output_dir,
            self._render_total_frames,
            self._render_fps,
        )
        self._dirty("render_active", "render_idle")

    def _on_cancel_render(self, h, e, a):
        self._render_cancel = True

    def _on_set_render_frames(self, v):
        try:
            self._render_total_frames = max(1, int(v))
            self._dirty("render_duration_str")
        except ValueError:
            pass

    def _on_set_render_fps(self, v):
        try:
            self._render_fps = max(1, int(v))
            self._dirty("render_duration_str")
        except ValueError:
            pass

    def _on_set_vp_res(self, idx):
        self._vp_export_resolution_idx = idx
        self._dirty("vp_size_hint", "vp_export_label",
                    *[f"vp_res_{i}_active" for i in range(len(VP_RESOLUTIONS))])

    def _on_set_vp_compress(self, v):
        try:
            self._vp_export_compress = max(0, min(9, int(v)))
        except ValueError:
            pass

    def _on_toggle_vp_transp(self, h, e, a):
        self._vp_export_transparency = not self._vp_export_transparency
        self._dirty("vp_transparency")

    def _on_toggle_video_section(self, h, e, a):
        self._video_expanded = not self._video_expanded
        self._dirty("video_collapsed", "video_expanded")
        lf.ui.request_redraw()

    def _on_toggle_export_section(self, h, e, a):
        self._export_expanded = not self._export_expanded
        self._dirty("export_collapsed", "export_expanded")
        lf.ui.request_redraw()

    def _on_run_still_bat(self, h, e, a):
        try:
            userprofile = os.environ.get("USERPROFILE", "C:\\Users\\default")
            rgb_path    = os.path.join(self._render_output_dir, "VIEWPORT_DRGB.png")
            gsc_path    = os.path.join(self._render_output_dir, "VIEWPORT_DGSC.png")
            script_path = os.path.join(userprofile, ".lichtfeld", "plugins", "DoF", "python", "Still-DoF_Bokeh.py")
            activate    = os.path.join(userprofile, "anaconda3", "Scripts", "activate.bat")

            missing = [p for p in (rgb_path, gsc_path) if not os.path.exists(p)]
            if missing:
                self._vp_set_status(f"File not found: {os.path.basename(missing[0])} — use Export first", error=True)
                return
            if not os.path.exists(script_path):
                self._vp_set_status(f"Script not found: {script_path}", error=True)
                return

            # Use conda's python.exe directly — activate doesn't reliably
            # switch PATH when launched from an embedded host like LichtFeld.
            import tempfile
            python_exe = os.path.join(userprofile, "anaconda3", "python.exe")
            if not os.path.exists(python_exe):
                self._vp_set_status(f"Anaconda python.exe not found: {python_exe}", error=True)
                return
            bat_lines = [
                "@echo off",
                f'"{python_exe}" "{script_path}" "{rgb_path}" "{gsc_path}"',
            ]
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".bat", delete=False, encoding="utf-8"
            )
            tmp.write("\r\n".join(bat_lines))
            tmp.close()
            subprocess.Popen(["cmd", "/c", tmp.name], shell=False)
            self._vp_set_status("Launched: Still-DoF_Bokeh.py via Anaconda", success=True)
        except Exception as e:
            self._vp_set_status(f"Launch error: {e}", error=True)

    def _on_browse_output(self, h, e, a):
        # QFileDialog crashes LichtFeld when called off the main thread.
        # tkinter isn't available in the embedded Python environment.
        # PowerShell's FolderBrowserDialog runs in its own process — no Qt
        # thread issues, works from any callback thread.
        # -Sta is required: FolderBrowserDialog needs a COM STA thread.
        # -NonInteractive must NOT be set — it suppresses the dialog UI.
        try:
            initialdir = self._render_output_dir if os.path.isdir(self._render_output_dir) else "c:\\"
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
                f"$d.SelectedPath = '{initialdir}';"
                "$d.Description = 'Select output folder';"
                "$d.ShowNewFolderButton = $true;"
                "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Sta", "-Command", ps_script],
                capture_output=True, text=True, timeout=120
            )
            folder = result.stdout.strip()
            if folder and os.path.isdir(folder):
                self._render_output_dir = folder
                self._dirty("render_output_dir")
        except Exception as e:
            self._vp_set_status(f"Browse error: {e}", error=True)

    def _on_open_output(self, h, e, a):
        try:
            import subprocess as _sp
            folder = self._render_output_dir
            if os.path.isdir(folder):
                _sp.Popen(["explorer", folder])
            else:
                self._vp_set_status(f"Folder not found: {folder}", error=True)
        except Exception as e:
            self._vp_set_status(f"Open error: {e}", error=True)

    def _on_open_dof_video(self, h, e, a):
        """Launch Video-DoF_Bokeh.py with the last rendered video pair."""
        depth_path = os.path.join(self._render_output_dir, "depth_video.mp4")
        rgb_path   = os.path.join(self._render_output_dir, "rgb_video.mp4")
        missing = [p for p in (depth_path, rgb_path) if not os.path.exists(p)]
        if missing:
            self._vp_set_status(
                f"File not found: {os.path.basename(missing[0])}", error=True)
            return
        script = str(Path(__file__).parent.parent / "python" / "Video-DoF_Bokeh.py")
        if not os.path.exists(script):
            self._vp_set_status("Video-DoF_Bokeh.py not found in python/ folder", error=True)
            return
        try:
            import threading as _t, sys as _sys

            def _launch(dep=depth_path, rgb=rgb_path, scr=script):
                _sys.argv = [scr, dep, rgb]
                code = Path(scr).read_text(encoding="utf-8")
                try:
                    exec(compile(code, scr, "exec"),
                         {"__name__": "__main__", "__file__": scr})
                except SystemExit:
                    pass

            _t.Thread(target=_launch, daemon=True).start()
            self._vp_set_status("Opening DoF Video compositor...", success=True)
        except Exception as e:
            self._vp_set_status(f"Launch error: {e}", error=True)

    def _on_open_dof_still(self, h, e, a):
        """Open Still-DoF_Bokeh.py with the last exported PNGs — no re-export."""
        rgb_path = os.path.join(self._render_output_dir, "VIEWPORT_DRGB.png")
        gsc_path = os.path.join(self._render_output_dir, "VIEWPORT_DGSC.png")
        missing = [p for p in (rgb_path, gsc_path) if not os.path.exists(p)]
        if missing:
            self._vp_set_status(
                f"File not found: {os.path.basename(missing[0])} — use Export first", error=True)
            return
        script = str(Path(__file__).parent.parent / "python" / "Still-DoF_Bokeh.py")
        if not os.path.exists(script):
            self._vp_set_status("Still-DoF_Bokeh.py not found in python/ folder", error=True)
            return
        try:
            import threading as _t, sys as _sys

            def _launch(rgb=rgb_path, gsc=gsc_path, scr=script):
                _sys.argv = [scr, rgb, gsc]
                code = Path(scr).read_text(encoding="utf-8")
                try:
                    exec(compile(code, scr, "exec"),
                         {"__name__": "__main__", "__file__": scr})
                except SystemExit:
                    pass

            _t.Thread(target=_launch, daemon=True).start()
            self._vp_set_status(
                f"Opening DoF Still: {os.path.basename(rgb_path)} + "
                f"{os.path.basename(gsc_path)}", success=True)
        except Exception as e:
            self._vp_set_status(f"Launch error: {e}", error=True)

    def _on_open_dof(self, h, e, a):
        """Launch the DoF compositor with the last written file pair."""
        rgb_path = os.path.join(self._render_output_dir, "VIEWPORT_DRGB.png")
        gsc_path = os.path.join(self._render_output_dir, "VIEWPORT_DGSC.png")
        missing = [p for p in (rgb_path, gsc_path) if not os.path.exists(p)]
        if missing:
            self._vp_set_status(
                f"File not found: {os.path.basename(missing[0])}", error=True)
            return
        script = str(Path(__file__).parent.parent / "python" / "Still-DoF_Bokeh.py")
        try:
            import threading as _t, sys as _sys

            def _launch(rgb=rgb_path, gsc=gsc_path, scr=script):
                _sys.argv = [scr, rgb, gsc]
                code = Path(scr).read_text(encoding="utf-8")
                try:
                    exec(compile(code, scr, "exec"),
                         {"__name__": "__main__", "__file__": scr})
                except SystemExit:
                    pass

            _t.Thread(target=_launch, daemon=True).start()
            self._vp_set_status(
                f"Opening DoF: {os.path.basename(rgb_path)} + "
                f"{os.path.basename(gsc_path)}", success=True)
        except Exception as e:
            self._vp_set_status(f"Launch error: {e}", error=True)

    def _on_vp_export(self, h, e, a):
        """Dual-capture export — saves two PNGs for Still-DoF_Bokeh.py:
        1. Restore original colours  → capture → VIEWPORT_DRGB.png  (colour, depth OFF)
        2. Re-apply depthmap         → capture → VIEWPORT_DGSC.png  (greyscale depth, depth ON)
        Use "Open DoF Still" button to launch the compositor with these files.
        All GPU state changes are sequenced through the draw handler so each
        capture sees the correct fully-rendered frame."""
        node_name = self._get_selected_splat_name()
        if not node_name:
            self._vp_set_status("No splat selected.", error=True)
            return

        out_dir   = self._render_output_dir
        rgb_path  = os.path.join(out_dir, "VIEWPORT_DRGB.png")
        gsc_path  = os.path.join(out_dir, "VIEWPORT_DGSC.png")
        script    = str(Path(__file__).parent.parent / "python" / "Still-DoF_Bokeh.py")

        os.makedirs(out_dir, exist_ok=True)

        # State machine driven from draw handler:
        #  step 1 → restore colours, mark dirty, wait frame
        #  step 2 → capture RGB, re-apply depthmap, wait frame
        #  step 3 → capture GSC, restore state, launch script
        state = {"step": 1, "was_enabled": self._enabled, "target_h": VP_RESOLUTIONS[self._vp_export_resolution_idx][1]}

        def _seq(ctx):
            s = state["step"]

            if s == 1:
                # Restore to plain colours for RGB capture
                self._restore_original_colors(silent=True)
                self._enabled = False
                scene = lf.get_scene()
                if scene: scene.notify_changed()
                state["step"] = 2

            elif s == 2:
                # Capture colour viewport → VIEWPORT_DRGB.png
                target_h = state["target_h"]
                try:
                    arr = _vp_capture_arr(target_h)
                    if arr is None:
                        self._vp_set_status("RGB capture failed.", error=True)
                        lf.remove_draw_handler("depthmap.dof_seq")
                        return
                    img = _vp_arr_to_image(arr)
                    with open(rgb_path, "wb") as fh:
                        img.save(fh, "PNG", compress_level=self._vp_export_compress)
                        fh.flush()
                        os.fsync(fh.fileno())
                    self._vp_set_status("RGB saved — applying depthmap...", warning=True)
                except Exception as e:
                    self._vp_set_status(f"RGB save error: {e}", error=True)
                    lf.remove_draw_handler("depthmap.dof_seq")
                    return

                # Re-apply depthmap for greyscale capture
                self._enabled = True
                self._apply_depthmap(silent=True)
                scene = lf.get_scene()
                if scene: scene.notify_changed()
                state["step"] = 3

            elif s == 3:
                # Capture depthmap viewport → VIEWPORT_DGSC.png
                target_h = state["target_h"]
                saved_ok = False
                try:
                    arr = _vp_capture_arr(target_h)
                    if arr is None:
                        self._vp_set_status("Depth capture failed.", error=True)
                        lf.remove_draw_handler("depthmap.dof_seq")
                        return
                    # Explicit flush+close via context manager to guarantee OS write
                    img = _vp_arr_to_image(arr)
                    with open(gsc_path, "wb") as fh:
                        img.save(fh, "PNG", compress_level=self._vp_export_compress)
                        fh.flush()
                        os.fsync(fh.fileno())
                    self._vp_set_status(f"Saved: {rgb_path} + {gsc_path}", success=True)
                    saved_ok = True
                except Exception as e:
                    self._vp_set_status(f"Depth save error: {e}", error=True)
                    lf.remove_draw_handler("depthmap.dof_seq")
                    return
                finally:
                    lf.remove_draw_handler("depthmap.dof_seq")
                    # Restore original enabled state
                    if not state["was_enabled"]:
                        self._restore_original_colors(silent=True)
                        self._enabled = False
                    self._dirty("enabled", "disabled")

                if not saved_ok:
                    return
                # Files saved — user clicks "Open DoF Still" to launch compositor.

        lf.add_draw_handler("depthmap.dof_seq", _seq)
        self._vp_set_status("Capturing...", warning=True)

