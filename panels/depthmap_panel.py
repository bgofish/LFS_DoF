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

from ..core.depthmap import (_get_export_camera_pos,
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

# (label, container_format, extension, codec_priority_list)
# codec_priority_list: tried in order until one works in the current PyAV build
VIDEO_FORMATS = [
    ("MP4",  "mp4",  ".mp4", ["h264_nvenc", "h264_amf", "h264_qsv", "libopenh264", "libx264", "mpeg4"]),
    ("MKV",  "matroska", ".mkv", ["h264_nvenc", "h264_amf", "h264_qsv", "libopenh264", "libx264", "mpeg4"]),
    ("MOV",  "mov",  ".mov", ["h264_nvenc", "h264_amf", "h264_qsv", "libopenh264", "libx264", "mpeg4"]),
    ("AVI",  "avi",  ".avi", ["h264_nvenc", "h264_amf", "h264_qsv", "libx264", "mpeg4", "msmpeg4"]),
    ("WebM", "webm", ".webm", ["libvpx-vp9", "libvpx"]),
]

# (label, crf_h264, crf_vp9, bpp)
# Bitrate is calculated per-encode from BPP * width * height * fps so quality
# scales automatically with resolution and frame rate.
# CRF values are no longer used (pure VBR bitrate mode for all codecs).
VIDEO_QUALITIES = [
    ("Very High", 16, 24, 0.50),   # ~50 Mbps @ 1080p30 — pristine master quality
    ("High",      22, 31, 0.20),   # ~20 Mbps @ 1080p30 — crisp, no macroblocking
    ("Medium",    28, 38, 0.08),   # ~ 8 Mbps @ 1080p30 — balanced web standard
    ("Low",       36, 48, 0.03),   # ~ 3 Mbps @ 1080p30 — compressed preview
]

VIDEO_FORMAT_DESCS = [
    "Widest compatibility",
    "Flexible container, any codec",
    "Apple / DaVinci Resolve",
    "Legacy universal format",
    "Web streaming, VP9 codec",
]

def _equirect_composite(ra, hdr, eye, target, up_vec_in, fov_deg, exposure=0.0):
    """Composite HDR equirectangular onto render_at output.
    ra         : [H, W, 3] float32  — render_at output (black where no splat)
    hdr        : [Hm, Wm, 3] float32 — equirectangular image
    eye/target : camera position and look-at point for this specific frame
    up_vec_in  : up vector tuple for this frame
    Returns [H, W, 3] float32 composited image.
    """
    h, w   = ra.shape[:2]
    hm, wm = hdr.shape[:2]

    # Derive camera basis from eye/target/up — correct per-frame, not live viewport
    fwd   = np.array(target) - np.array(eye)
    fwd   = fwd / np.linalg.norm(fwd)
    up    = np.array(up_vec_in, dtype=np.float32)
    right = np.cross(fwd, up);  right = right / np.linalg.norm(right)
    up    = np.cross(right, fwd)  # reorthogonalise

    # Per-pixel ray directions
    fov_rad = np.radians(fov_deg)
    tan_h   = np.tan(fov_rad / 2)
    tan_w   = tan_h * (w / h)
    ys = np.linspace( tan_h, -tan_h, h)
    xs = np.linspace(-tan_w,  tan_w, w)
    xs, ys = np.meshgrid(xs, ys)

    ray_world = (xs[..., np.newaxis] * right +
                 ys[..., np.newaxis] * up +
                 np.ones_like(xs)[..., np.newaxis] * fwd)
    ray_norm  = ray_world / np.linalg.norm(ray_world, axis=-1, keepdims=True)

    # Equirect UV
    rx, ry, rz = ray_norm[...,0], ray_norm[...,1], ray_norm[...,2]
    u = 0.5 + np.arctan2(rz, rx) / (2 * np.pi)
    v = 0.5 - np.arcsin(np.clip(ry, -1, 1)) / np.pi

    # Sample HDR via nearest-neighbour
    px = np.clip((u * wm).astype(np.int32), 0, wm - 1)
    py = np.clip((v * hm).astype(np.int32), 0, hm - 1)
    bg = hdr[py, px].astype(np.float32)

    # Exposure + Reinhard tonemap
    if exposure != 0.0:
        bg = bg * (2.0 ** exposure)
    bg = bg / (1.0 + bg)
    bg = np.clip(bg, 0.0, 1.0)

    # Derive splat mask from render_at alpha channel directly:
    # render_at returns RGBA where alpha=0 means no splat (background).
    # If only 3 channels, use luminance threshold on black background pixels.
    if ra.shape[2] == 4:
        splat_mask = (ra[:, :, 3] > 0.05).astype(np.float32)[..., np.newaxis]
        ra = ra[:, :, :3]
    else:
        # Black pixels in render_at are background — splat has colour
        lum = ra.max(axis=2)
        splat_mask = (lum > 0.01).astype(np.float32)[..., np.newaxis]

    # Splat in front, HDR behind
    return (ra * splat_mask + bg * (1.0 - splat_mask)).astype(np.float32)


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

def _vp_render_at_size(width, height, bg_color=None):
    """render_at sized for export (1080p/1440p/4K/8K) — uses the live
    viewport's current eye/target/up/fov, optionally with a solid
    background colour (used by the BW2A transparency export path)."""
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

    def __init__(self):
        global panel
        panel = self

        # Core state
        self._current_target   = None

        # Range — these mirror Lichtfeld's native depth_view_range and are
        # kept locally only so Pick Point labels can show a value the moment
        # a point is clicked, before the next on_update() tick reads it back.
        self._min_depth            = None
        self._max_depth            = None
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
        self._video_use_custom_res = False
        self._video_quality        = 1        # 0=Very High  1=High  2=Medium  3=Low
        self._video_format         = 0        # index into VIDEO_FORMATS
        self._video_custom_w      = 1920
        self._video_custom_h      = 1080
        self._render_active       = False
        self._render_cancel       = False
        self._render_progress     = 0.0
        self._render_status       = ""
        # Frame-pump state (driven from on_update on the main thread)
        self._render_frame_idx    = 0
        self._render_frame_paths  = []
        self._render_rgb_pass     = False  # True when doing the RGB second pass
        self._render_rgb_only     = False  # True when skipping depth pass entirely
        self._render_use_equirect  = False
        self._render_equirect_path = ""     # full path to user's HDR/EXR/panorama
        self._render_equirect_exposure = 0.0  # EV stops (+/- float)
        self._equirect_cache       = None   # (path, numpy_array) cache last loaded
        self._render_rgb_paths    = []
        # Native depth_view snapshot — set when a render starts, restored
        # when it finishes/cancels (replaces the old per-frame colour bake).
        self._render_was_depth_view = False
        self._render_was_depth_mode = "gray"
        self._render_depth_mode     = "gray"

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

        # ── Target ───────────────────────────────────────────────────────
        model.bind_func("target_name",    lambda: self._get_selected_splat_name() or "")
        model.bind_func("no_splat",       lambda: self._get_selected_splat_name() is None)
        model.bind_func("has_splat",      lambda: self._get_selected_splat_name() is not None)

        # ── Depth Map (native lf.depth_view toggle) ─────────────────────────
        model.bind_func("depth_mode_gray_active",
                         lambda: lf.get_depth_view() and lf.get_depth_view_mode() == "gray")
        model.bind_func("depth_mode_color_active",
                         lambda: lf.get_depth_view() and lf.get_depth_view_mode() == "palette")
        model.bind_func("depth_mode_off_active", lambda: not lf.get_depth_view())
        model.bind_func("depth_mode_label",
                         lambda: (f"{lf.get_depth_view_mode().capitalize()} (ON)"
                                  if lf.get_depth_view() else "OFF"))
        model.bind_event("set_depth_mode_gray",  self._on_set_depth_mode_gray)
        model.bind_event("set_depth_mode_color", self._on_set_depth_mode_color)
        model.bind_event("set_depth_mode_off",   self._on_set_depth_mode_off)

        # ── Depth Range (drives lf.set_depth_view_range) ───────────────────
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
        model.bind_func("has_range",      lambda: self._min_depth is not None)

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
        model.bind("video_custom_w_str",
                   lambda: str(self._video_custom_w),
                   self._on_set_video_custom_w)
        model.bind("video_custom_h_str",
                   lambda: str(self._video_custom_h),
                   self._on_set_video_custom_h)
        model.bind_func("video_res_viewport",    lambda: not self._video_use_custom_res)
        model.bind_func("video_res_custom",      lambda: self._video_use_custom_res)
        model.bind_func("video_res_report",      self._video_res_report)
        model.bind_event("set_video_res_viewport", self._on_set_video_res_viewport)
        model.bind_event("set_video_res_custom",   self._on_set_video_res_custom)
        # Video quality + format
        model.bind_func("video_quality_label", lambda: VIDEO_QUALITIES[self._video_quality][0])
        model.bind_func("video_quality_mbps",  lambda: f"~{VIDEO_QUALITIES[self._video_quality][3] * 1920 * 1080 * 30 / 1_000_000:.0f} Mbps")
        model.bind_func("video_format_label",  lambda: VIDEO_FORMATS[self._video_format][0])
        model.bind_func("video_format_desc",   lambda: VIDEO_FORMAT_DESCS[self._video_format])
        model.bind("equirect_path",
                   lambda: self._render_equirect_path,
                   self._on_set_equirect_path)
        model.bind("equirect_exposure_str",
                   lambda: f"{self._render_equirect_exposure:+.1f}",
                   self._on_set_equirect_exposure)
        model.bind_func("video_use_equirect",  lambda: self._render_use_equirect)
        model.bind_func("video_no_equirect",         lambda: not self._render_use_equirect)
        model.bind_event("toggle_video_equirect",    self._on_toggle_video_equirect)
        model.bind_event("browse_equirect",          self._on_browse_equirect)
        model.bind_event("cycle_video_quality", self._on_cycle_video_quality)
        model.bind_event("cycle_video_format",  self._on_cycle_video_format)
        model.bind_func("render_duration_str", lambda: f"{self._render_total_frames / max(self._render_fps,1):.1f}s  ({self._render_total_frames} frames @ {self._render_fps}fps)")
        model.bind_func("render_active",       lambda: self._render_active)
        model.bind_func("render_idle",         lambda: not self._render_active)
        model.bind_func("render_progress",     lambda: f"{self._render_progress:.2f}")
        model.bind_func("render_progress_pct", lambda: f"{self._render_progress * 100:.0f}%")
        model.bind_func("render_status",       lambda: self._render_status)
        model.bind_func("render_status_ok",    lambda: bool(self._render_status) and "ERROR" not in self._render_status)
        model.bind_func("render_status_err",   lambda: "ERROR" in self._render_status)
        model.bind_event("do_render_video",         self._on_render_video)
        model.bind_event("do_render_rgb_only",      self._on_render_rgb_only_video)
        model.bind_event("run_video_bat",           self._on_run_video_bat)
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
        model.bind_func("vp_transparency",    lambda: self._vp_export_transparency)
        model.bind_func("vp_no_transparency", lambda: not self._vp_export_transparency)
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

        # Mirror Lichtfeld's native depth_view_range into our local labels,
        # so editing Near/Far on the native overlay (top-left of viewport)
        # stays in sync with what the panel shows.
        try:
            near, far = lf.get_depth_view_range()
            self._min_depth, self._max_depth = near, far
        except Exception:
            pass

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
                    "vp_transparency", "vp_no_transparency",
                    "min_gt_max", "min_depth_str", "max_depth_str", "has_range",
                    "depth_mode_gray_active", "depth_mode_color_active",
                    "depth_mode_off_active", "depth_mode_label",
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

    def _camera_distance(self, pos):
        """Distance from the current viewport camera to a 3D world position.
        Native depth_view always visualizes true camera-space depth, so this
        is the only depth measure Pick Point needs now."""
        view = lf.get_current_view()
        if view:
            cam_pos = np.array(view.translation.numpy()).flatten()
            return float(np.linalg.norm(np.array(pos) - cam_pos))
        return float(pos[2])

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
        self._captured_depth = self._camera_distance(world_pos)
        if point_num == 1:
            self._point1_pos = world_pos
            self._min_depth = self._captured_depth
            self._status_msg = f"Point 1: {self._min_depth:.2f}"
        else:
            self._point2_pos = world_pos
            self._max_depth = self._captured_depth
            self._status_msg = f"Point 2: {self._max_depth:.2f}"
        self._status_is_error = False
        self._push_depth_range()
        self._dirty_all()

    def _push_depth_range(self):
        """Push our locally-tracked min/max straight into Lichtfeld's native
        depth_view_range so the (already-active or about-to-be-activated)
        native Depth Map overlay reflects it immediately."""
        if self._min_depth is None or self._max_depth is None:
            return
        try:
            lf.set_depth_view_range(self._min_depth, self._max_depth)
            lf.ui.request_redraw()
        except Exception as e:
            _depth_log(f"set_depth_view_range error: {e}")

    def _nudge_depth(self, which, delta):
        if which == "min":
            self._min_depth = (self._min_depth or 0.0) + delta
        else:
            self._max_depth = (self._max_depth or 0.0) + delta
        self._push_depth_range()
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

    def _render_frame_equirect(self, eye, target, up, fov, w, h):
        """Render frame and composite equirectangular HDR background if enabled."""
        # Always get render_at output — 3-channel, no bg
        ra = _vp_render_at_explicit(eye, target, up, fov, w, h)
        if not self._render_use_equirect or not self._render_equirect_path:
            return ra
        if ra is None:
            return ra

        # Load HDR (cached — only re-read if path changed)
        hdr = None
        if (self._equirect_cache is not None and
                self._equirect_cache[0] == self._render_equirect_path):
            hdr = self._equirect_cache[1]
        else:
            try:
                import cv2 as _cv2
                img = _cv2.imread(self._render_equirect_path,
                                  _cv2.IMREAD_ANYDEPTH | _cv2.IMREAD_COLOR)
                if img is not None:
                    hdr = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB).astype(np.float32)
                    self._equirect_cache = (self._render_equirect_path, hdr)
                else:
                    self._vp_set_status(f"Could not load equirect: {self._render_equirect_path}", error=True)
                    return ra
            except Exception as e:
                self._vp_set_status(f"Equirect load error: {e}", error=True)
                return ra

        # Pass eye/target/up directly — composite derives rotation per-frame
        return _equirect_composite(ra, hdr, eye, target, up,
                                   fov, exposure=self._render_equirect_exposure)

    def _render_depth_video(self, output_dir, total_frames, fps):
        """Arm the per-frame render pump. Uses Lichtfeld's native depth_view
        toggle for the depth pass — render_at() respects it directly, so
        there is no per-frame colour baking, no GPU flush to wait on, and
        no splat data is ever touched."""
        if not lf.has_scene():
            self._render_status = "ERROR: No scene loaded"
            self._dirty("render_status", "render_status_err")
            return

        os.makedirs(output_dir, exist_ok=True)

        self._render_active      = True
        self._render_cancel      = False
        self._render_progress    = 0.0
        self._render_frame_idx   = 0
        self._render_frame_paths = []
        self._render_rgb_paths   = []
        self._render_was_depth_view = lf.get_depth_view()
        self._render_was_depth_mode = lf.get_depth_view_mode()
        self._render_depth_mode     = lf.get_depth_view_mode() or "gray"

        if self._render_rgb_only:
            # Skip depth pass — start directly on the RGB pass
            self._render_rgb_pass = True
            self._render_status   = "Starting RGB pass..."
        else:
            self._render_rgb_pass = False
            self._render_status   = "Starting..." 

    def _restore_render_depth_view(self):
        try:
            lf.set_depth_view(self._render_was_depth_view)
            lf.set_depth_view_mode(self._render_was_depth_mode)
            lf.ui.request_redraw()
        except Exception as e:
            _depth_log(f"restore depth_view error: {e}")

    def _render_pump_frame(self):
        """Called by on_update() once per tick while a render is active.
        Toggles the native depth_view state for the current pass, then
        renders directly — render_at() honours depth_view, so each frame
        is just a state flip + a render call."""
        i            = self._render_frame_idx
        total_frames = self._render_total_frames
        output_dir   = self._render_output_dir

        if self._render_cancel:
            self._render_status  = "Cancelled"
            self._render_active  = False
            self._restore_render_depth_view()
            return

        # ── 1. Ensure correct native depth_view state for this pass ───────
        try:
            if self._render_rgb_pass:
                lf.set_depth_view(False)
            else:
                lf.set_depth_view_mode(self._render_depth_mode)
                lf.set_depth_view(True)
        except Exception as e:
            _depth_log(f"RENDER PUMP depth_view ERROR frame {i}: {e}")
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
            if self._video_use_custom_res:
                vp_w, vp_h = self._video_custom_w, self._video_custom_h
            arr = self._render_frame_equirect(eye, target, up, fov, vp_w, vp_h)
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
                # RGB pass complete — restore viewport to its pre-render state
                self._render_active    = False
                self._render_progress  = 1.0
                self._render_rgb_pass  = False
                self._restore_render_depth_view()
                self._finish_render_video()
            else:
                # Depth pass done — start RGB pass
                self._render_frame_idx  = 0
                self._render_rgb_paths  = []
                self._render_rgb_pass   = True
                self._render_status     = "Starting RGB pass..."

    def _finish_render_video(self):
        """Encode depth + RGB frame sequences to video via PyAV."""
        import threading as _threading

        depth_paths = self._render_frame_paths
        rgb_paths   = self._render_rgb_paths
        output_dir  = self._render_output_dir
        fps         = self._render_fps
        fmt_label, fmt_container, fmt_ext, fmt_codecs = VIDEO_FORMATS[self._video_format]
        q_label, q_crf_h264, q_crf_vp9, q_bpp = VIDEO_QUALITIES[self._video_quality]

        if not depth_paths and not rgb_paths:
            self._render_status = "No frames rendered"
            return

        def _encode_pass(frame_paths, stem, label):
            video_path = os.path.join(output_dir, f"{stem}{fmt_ext}")
            try:
                import av
                from PIL import Image as _Image
                first = _Image.open(frame_paths[0]).convert("RGB")
                w, h  = first.size

                # Calculate bitrate from BPP scaled to actual resolution + fps
                bitrate_bps = int(w * h * fps * q_bpp)
                maxrate     = int(bitrate_bps * 1.25)
                bufsize     = int(bitrate_bps * 2)

                stream    = None
                container = None
                used      = None

                for codec in fmt_codecs:
                    try:
                        if container:
                            try: container.close()
                            except: pass

                        # Codec-appropriate options — must be passed before add_stream
                        # opens the context; assigning to codec_context.options after is a no-op
                        if any(x in codec for x in ("nvenc", "amf", "qsv")):
                            codec_opts = {"rc": "vbr", "maxrate": str(maxrate), "bufsize": str(bufsize)}
                        elif codec in ("libx264", "libopenh264"):
                            codec_opts = {"b:v": str(bitrate_bps), "maxrate": str(maxrate),
                                          "bufsize": str(bufsize), "preset": "medium"}
                        else:
                            # mpeg4: b:v via options + codec_context.bit_rate for belt-and-braces
                            codec_opts = {"b:v": str(bitrate_bps), "maxrate": str(maxrate),
                                          "bufsize": str(bufsize)}

                        container = av.open(video_path, mode="w", format=fmt_container)
                        stream = container.add_stream(codec, rate=fps, options=codec_opts)
                        stream.width    = w
                        stream.height   = h
                        stream.pix_fmt  = "yuv420p"
                        stream.bit_rate = bitrate_bps
                        stream.codec_context.bit_rate = bitrate_bps

                        # Test codec by encoding the actual first frame —
                        # no blank probe frame so no green screen at start.
                        img0  = _Image.open(frame_paths[0]).convert("RGB")
                        frm0  = av.VideoFrame.from_image(img0)
                        frm0.pts = 0
                        list(stream.encode(frm0))
                        used = codec
                        _depth_log(f"ENCODE | codec={codec} quality={q_label} "
                                   f"{bitrate_bps/1_000_000:.1f}Mbps format={fmt_label} OK")
                        break
                    except Exception as ce:
                        _depth_log(f"ENCODE | codec={codec} failed: {ce}")
                        stream = None

                if used is None:
                    return f"ERROR: no working codec for {fmt_label}"

                for idx, path in enumerate(frame_paths):
                    img   = _Image.open(path).convert("RGB")
                    frame = av.VideoFrame.from_image(img)
                    frame.pts = idx
                    if idx == 0:
                        continue   # already encoded above during codec probe
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

        rgb_only = self._render_rgb_only

        def _encode():
            if rgb_only:
                result_r = _encode_pass(rgb_paths, "rgb_video", "RGB")
                self._render_status = (f"Saved: rgb_video{fmt_ext}  [{q_label}]"
                                       if "ERROR" not in str(result_r) else result_r)
            else:
                result_d = _encode_pass(depth_paths, "depth_video", "Depth")
                if rgb_paths:
                    result_r = _encode_pass(rgb_paths, "rgb_video", "RGB")
                    if "ERROR" in str(result_d) or "ERROR" in str(result_r):
                        self._render_status = f"{result_d} | {result_r}"
                    else:
                        self._render_status = (f"Done: depth_video{fmt_ext} + "
                                               f"rgb_video{fmt_ext}  [{q_label}]")
                else:
                    self._render_status = (f"Saved: depth_video{fmt_ext}  [{q_label}]"
                                           if "ERROR" not in str(result_d) else result_d)

        _threading.Thread(target=_encode, daemon=True).start()

    # ── Event handlers ────────────────────────────────────────────────────────

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
        self._status_msg = "Points cleared"
        self._status_is_error = False
        self._picking_point = 0
        self._dirty_all()

    def _on_set_min_depth(self, v):
        try:
            self._min_depth = float(v)
            self._push_depth_range()
            self._dirty("min_depth_str", "min_gt_max")
        except ValueError:
            pass

    def _on_set_max_depth(self, v):
        try:
            self._max_depth = float(v)
            self._push_depth_range()
            self._dirty("max_depth_str", "min_gt_max")
        except ValueError:
            pass

    def _on_swap_range(self, h, e, a):
        self._min_depth, self._max_depth = self._max_depth, self._min_depth
        self._push_depth_range()
        self._dirty("min_depth_val", "max_depth_val", "min_depth_str", "max_depth_str", "min_gt_max")

    def _on_set_depth_mode_gray(self, h, e, a):
        lf.set_depth_view_mode("gray")
        lf.set_depth_view(True)
        lf.ui.request_redraw()
        self._dirty("depth_mode_gray_active", "depth_mode_color_active",
                    "depth_mode_off_active", "depth_mode_label")

    def _on_set_depth_mode_color(self, h, e, a):
        lf.set_depth_view_mode("palette")
        lf.set_depth_view(True)
        lf.ui.request_redraw()
        self._dirty("depth_mode_gray_active", "depth_mode_color_active",
                    "depth_mode_off_active", "depth_mode_label")

    def _on_set_depth_mode_off(self, h, e, a):
        lf.set_depth_view(False)
        lf.ui.request_redraw()
        self._dirty("depth_mode_gray_active", "depth_mode_color_active",
                    "depth_mode_off_active", "depth_mode_label")

    def _on_render_video(self, h, e, a):
        if not lf.has_scene():
            self._render_status = "ERROR: No scene loaded"
            self._dirty("render_status", "render_status_err")
            return
        self._render_rgb_only = False
        self._render_depth_video(
            self._render_output_dir,
            self._render_total_frames,
            self._render_fps,
        )
        self._dirty("render_active", "render_idle")

    def _on_render_rgb_only_video(self, h, e, a):
        if not lf.has_scene():
            self._render_status = "ERROR: No scene loaded"
            self._dirty("render_status", "render_status_err")
            return
        self._render_rgb_only = True
        self._render_depth_video(
            self._render_output_dir,
            self._render_total_frames,
            self._render_fps,
        )
        self._dirty("render_active", "render_idle")

    def _on_run_video_bat(self, h, e, a):
        try:
            userprofile = os.environ.get("USERPROFILE", "C:\\Users\\default")
            fmt_ext     = VIDEO_FORMATS[self._video_format][2]
            depth_path  = os.path.join(self._render_output_dir, f"depth_video{fmt_ext}")
            rgb_path    = os.path.join(self._render_output_dir, f"rgb_video{fmt_ext}")
            python_exe  = os.path.join(userprofile, "anaconda3", "python.exe")
            script_path = os.path.join(userprofile, ".lichtfeld", "plugins", "DoF", "python", "Video-DoF_Bokeh.py")

            missing = [p for p in (depth_path, rgb_path) if not os.path.exists(p)]
            if missing:
                self._vp_set_status(f"File not found: {os.path.basename(missing[0])} — render first", error=True)
                return
            if not os.path.exists(python_exe):
                self._vp_set_status(f"Anaconda python.exe not found: {python_exe}", error=True)
                return
            if not os.path.exists(script_path):
                self._vp_set_status(f"Video-DoF_Bokeh.py not found: {script_path}", error=True)
                return

            import tempfile
            bat_lines = [
                "@echo off",
                f'"{python_exe}" "{script_path}" "{rgb_path}" "{depth_path}"',
            ]
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".bat", delete=False, encoding="utf-8"
            )
            tmp.write("\r\n".join(bat_lines))
            tmp.close()
            subprocess.Popen(["cmd", "/c", tmp.name], shell=False)
            self._vp_set_status("Launched: Video-DoF_Bokeh.py via Anaconda", success=True)
        except Exception as e:
            self._vp_set_status(f"Launch error: {e}", error=True)

    def _on_cancel_render(self, h, e, a):
        self._render_cancel = True

    def _on_set_equirect_path(self, v):
        self._render_equirect_path = str(v).strip()
        self._equirect_cache = None   # invalidate cache
        self._dirty("equirect_path")

    def _on_set_equirect_exposure(self, v):
        try:
            self._render_equirect_exposure = float(v)
        except ValueError:
            pass

    def _on_browse_equirect(self, h, e, a):
        try:
            userprofile = os.environ.get("USERPROFILE", "C:\\Users\\default")
            initialdir  = os.path.dirname(self._render_equirect_path) if self._render_equirect_path else userprofile
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$d = New-Object System.Windows.Forms.OpenFileDialog;"
                f"$d.InitialDirectory = '{initialdir}';"
                "$d.Filter = 'HDR/EXR images (*.hdr;*.exr;*.png;*.jpg)|*.hdr;*.exr;*.png;*.jpg|All files (*.*)|*.*';"
                "$d.Title = 'Select equirectangular image';"
                "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.FileName }"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Sta", "-Command", ps_script],
                capture_output=True, text=True, timeout=120
            )
            path = result.stdout.strip()
            if path and os.path.exists(path):
                self._render_equirect_path = path
                self._equirect_cache = None
                self._dirty("equirect_path")
        except Exception as e:
            self._vp_set_status(f"Browse error: {e}", error=True)

    def _on_toggle_video_equirect(self, h, e, a):
        self._render_use_equirect = not self._render_use_equirect
        self._dirty("video_use_equirect", "video_no_equirect")

    def _on_cycle_video_quality(self, h, e, a):
        self._video_quality = (self._video_quality + 1) % len(VIDEO_QUALITIES)
        self._dirty("video_quality_label", "video_quality_mbps")

    def _on_cycle_video_format(self, h, e, a):
        self._video_format = (self._video_format + 1) % len(VIDEO_FORMATS)
        self._dirty("video_format_label", "video_format_desc")

    def _on_set_video_custom_w(self, v):
        try:
            self._video_custom_w = max(2, int(v) & ~1)   # ensure even
        except ValueError:
            pass

    def _on_set_video_custom_h(self, v):
        try:
            self._video_custom_h = max(2, int(v) & ~1)   # ensure even
        except ValueError:
            pass

    def _on_set_video_res_viewport(self, h, e, a):
        self._video_use_custom_res = False
        self._video_quality        = 1        # 0=Very High  1=High  2=Medium  3=Low
        self._video_format         = 0        # index into VIDEO_FORMATS
        self._dirty("video_res_viewport", "video_res_custom", "video_res_report")

    def _on_set_video_res_custom(self, h, e, a):
        self._video_use_custom_res = True
        self._dirty("video_res_viewport", "video_res_custom", "video_res_report")

    def _video_res_report(self):
        if self._video_use_custom_res:
            return f"Custom: {self._video_custom_w} × {self._video_custom_h} px"
        params = _vp_get_view_params()
        if params:
            _, _, _, _, vp_w, vp_h = params
            return f"Viewport: {vp_w} × {vp_h} px"
        return "Viewport resolution"

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
        fmt_ext    = VIDEO_FORMATS[self._video_format][2]
        rgb_path   = os.path.join(self._render_output_dir, f"rgb_video{fmt_ext}")
        depth_path = os.path.join(self._render_output_dir, f"depth_video{fmt_ext}")
        missing = [p for p in (rgb_path, depth_path) if not os.path.exists(p)]
        if missing:
            self._vp_set_status(
                f"File not found: {os.path.basename(missing[0])} — render first", error=True)
            return
        script = str(Path(__file__).parent.parent / "python" / "Video-DoF_Bokeh.py")
        if not os.path.exists(script):
            self._vp_set_status("Video-DoF_Bokeh.py not found in python/ folder", error=True)
            return
        try:
            import threading as _t, sys as _sys

            def _launch(rgb=rgb_path, dep=depth_path, scr=script):
                # argv[1] = colour/RGB video, argv[2] = depth/greyscale video
                _sys.argv = [scr, rgb, dep]
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
        1. Native depth_view OFF → capture → VIEWPORT_DRGB.png  (colour)
        2. Native depth_view ON, mode='gray' → capture → VIEWPORT_DGSC.png  (greyscale depth)
        Use "Open DoF Still" button to launch the compositor with these files.
        Uses Lichtfeld's built-in depth-view toggle (lf.set_depth_view /
        lf.set_depth_view_mode) instead of baking colours into splat SH0 —
        no splat data is touched, so there is nothing to corrupt or restore
        on the splat side. Only the depth_view render setting itself is
        snapshotted and restored once the export completes."""
        out_dir   = self._render_output_dir
        rgb_path  = os.path.join(out_dir, "VIEWPORT_DRGB.png")
        gsc_path  = os.path.join(out_dir, "VIEWPORT_DGSC.png")
        script    = str(Path(__file__).parent.parent / "python" / "Still-DoF_Bokeh.py")

        os.makedirs(out_dir, exist_ok=True)

        # Snapshot current depth-view state so it can be restored exactly.
        was_depth_view  = lf.get_depth_view()
        was_depth_mode  = lf.get_depth_view_mode()
        was_depth_range = lf.get_depth_view_range()

        # State machine driven from draw handler:
        #  step 1 → ensure depth_view OFF, wait frame
        #  step 2 → capture RGB, switch depth_view ON (gray), wait frame
        #  step 3 → capture GSC, restore depth_view state
        state = {"step": 1, "target_h": VP_RESOLUTIONS[self._vp_export_resolution_idx][1]}

        def _seq(ctx):
            s = state["step"]

            if s == 1:
                lf.set_depth_view(False)
                lf.ui.request_redraw()
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
                    self._vp_set_status("RGB saved — switching to depth view...", warning=True)
                except Exception as e:
                    self._vp_set_status(f"RGB save error: {e}", error=True)
                    lf.remove_draw_handler("depthmap.dof_seq")
                    return

                # Switch to native greyscale depth view for the second capture
                lf.set_depth_view_mode("gray")
                lf.set_depth_view(True)
                lf.ui.request_redraw()
                state["step"] = 3

            elif s == 3:
                # Capture native depth-view viewport → VIEWPORT_DGSC.png
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
                finally:
                    lf.remove_draw_handler("depthmap.dof_seq")
                    # Restore native depth-view state exactly as it was
                    lf.set_depth_view(was_depth_view)
                    lf.set_depth_view_mode(was_depth_mode)
                    lf.set_depth_view_range(*was_depth_range)
                    lf.ui.request_redraw()

                if not saved_ok:
                    return
                # Files saved — user clicks "Open DoF Still" to launch compositor.

        lf.add_draw_handler("depthmap.dof_seq", _seq)
        self._vp_set_status("Capturing...", warning=True)

