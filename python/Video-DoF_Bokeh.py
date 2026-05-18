import sys
import os
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QCheckBox, 
                               QComboBox, QPushButton, QFileDialog, QMessageBox)

class ProDepthBlurVideoApp(QMainWindow):
    def __init__(self, init_color_path=None, init_depth_path=None):
        super().__init__()
        self.setWindowTitle("Pro Depth Map Video Sync & Focus Engine - PySide6")
        self.setGeometry(100, 100, 1400, 1000)

        # Core Video Capture Trackers
        self.color_path = None
        self.depth_path = None
        self.cap_color = None
        self.cap_depth = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.depth_frame_offset = 0  

        # Playback Loops
        self.is_playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.advance_loop_frame)
        
        # Navigation Coordinate Space (Pan & Zoom state)
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_start = None
        
        # Optical configuration parameters
        self.max_blur_kernel = 21 
        self.focal_value = 255  
        self.dof_thickness = 20  
        self.selected_shape = "Circle"

        self.setup_ui()

        if init_color_path and init_depth_path:
            self.load_initial_videos(init_color_path, init_depth_path)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Drag and Drop Panels ---
        drop_layout = QHBoxLayout()
        self.lbl_color = QLabel("Drag & Drop COLOR Video Here")
        self.lbl_color.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_color.setStyleSheet("background-color: #2b2b2b; color: #aaaaaa; border: 2px dashed #555555; font-size: 14px;")
        self.lbl_color.setFixedHeight(60)
        
        self.lbl_depth = QLabel("Drag & Drop DEPTH Video Here")
        self.lbl_depth.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_depth.setStyleSheet("background-color: #2b2b2b; color: #aaaaaa; border: 2px dashed #555555; font-size: 14px;")
        self.lbl_depth.setFixedHeight(60)

        self.lbl_color.setAcceptDrops(True)
        self.lbl_depth.setAcceptDrops(True)
        self.lbl_color.dragEnterEvent = lambda e: e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
        self.lbl_depth.dragEnterEvent = lambda e: e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
        self.lbl_color.dropEvent = lambda e: self.handle_drop(e, "color")
        self.lbl_depth.dropEvent = lambda e: self.handle_drop(e, "depth")

        drop_layout.addWidget(self.lbl_color)
        drop_layout.addWidget(self.lbl_depth)
        main_layout.addLayout(drop_layout)

        # --- Interactive Canvas Viewport ---
        self.canvas = QLabel()
        self.canvas.setStyleSheet("background-color: #111111;")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.canvas, stretch=1)

        self.canvas.mousePressEvent = self.start_pan
        self.canvas.mouseMoveEvent = self.execute_pan
        self.canvas.wheelEvent = self.execute_zoom

        # --- NEW: Timeline Scrubbing & Control Panel ---
        timeline_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play Loop")
        self.btn_play.setEnabled(False)
        self.btn_play.setFixedWidth(100)
        self.btn_play.setStyleSheet("background-color: #29b6f6; color: black; font-weight: bold;")
        self.btn_play.clicked.connect(self.toggle_playback)

        # The Timeline Scrub Slider
        self.slider_timeline = QSlider(Qt.Orientation.Horizontal)
        self.slider_timeline.setRange(0, 0)
        self.slider_timeline.setEnabled(False)
        self.slider_timeline.sliderMoved.connect(self.on_timeline_scrub) # Triggers on click-drag scrubbing
        self.slider_timeline.sliderPressed.connect(self.on_scrub_start)
        self.slider_timeline.sliderReleased.connect(self.on_scrub_stop)

        self.lbl_time_status = QLabel("Frame: 0 / 0")
        self.lbl_time_status.setFixedWidth(130)

        timeline_layout.addWidget(self.btn_play)
        timeline_layout.addWidget(self.slider_timeline, stretch=1)
        timeline_layout.addWidget(self.lbl_time_status)
        main_layout.addLayout(timeline_layout)

        # --- Sync and Modes Options Bar ---
        toggle_layout = QHBoxLayout()
        self.chk_split = QCheckBox("Split View Mode")
        self.chk_split.setChecked(True)
        self.chk_split.stateChanged.connect(self.update_preview)
        
        self.chk_invert = QCheckBox("Invert Depth Mask")
        self.chk_invert.stateChanged.connect(self.update_preview)
        
        self.chk_grey = QCheckBox("View Focus Mask Only")
        self.chk_grey.stateChanged.connect(self.update_preview)

        self.shape_menu = QComboBox()
        self.shape_menu.addItems([
            "Circle", "Triangle (3 Blade)", "Diamond (4 Blade)", "Pentagon (5 Blade)", 
            "Hexagon (6 Blade)", "Heptagon (7 Blade)", "Octagon (8 Blade)", 
            "Nonagon (9 Blade)", "Decagon (10 Blade)", "Hendecagon (11 Blade)"
        ])
        self.shape_menu.currentTextChanged.connect(self.on_shape_change)

        # Timeline Frame Offset Buttons
        sync_box = QHBoxLayout()
        self.btn_sync_minus = QPushButton("-1 Fr")
        self.btn_sync_minus.setFixedWidth(45)
        self.btn_sync_minus.setEnabled(False)
        self.btn_sync_minus.clicked.connect(lambda: self.adjust_sync(-1))
        
        self.lbl_sync_val = QLabel("Sync: +0")
        self.lbl_sync_val.setFixedWidth(65)
        self.lbl_sync_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_sync_plus = QPushButton("+1 Fr")
        self.btn_sync_plus.setFixedWidth(45)
        self.btn_sync_plus.setEnabled(False)
        self.btn_sync_plus.clicked.connect(lambda: self.adjust_sync(1))
        
        sync_box.addWidget(self.btn_sync_minus)
        sync_box.addWidget(self.lbl_sync_val)
        sync_box.addWidget(self.btn_sync_plus)

        toggle_layout.addWidget(self.chk_split)
        toggle_layout.addWidget(self.chk_invert)
        toggle_layout.addWidget(self.chk_grey)
        toggle_layout.addWidget(QLabel("Bokeh Shape:"))
        toggle_layout.addWidget(self.shape_menu)
        toggle_layout.addWidget(QLabel("      Timeline Sync:"))
        toggle_layout.addLayout(sync_box)
        toggle_layout.addStretch()
        main_layout.addLayout(toggle_layout)

        # --- Optical Settings Sliders ---
        sliders_layout = QVBoxLayout()
        
        blur_box = QHBoxLayout()
        blur_box.addWidget(QLabel("Max Bokeh Diameter:"), stretch=1)
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(3, 101)
        self.slider_blur.setSingleStep(2)
        self.slider_blur.setValue(self.max_blur_kernel)
        self.slider_blur.valueChanged.connect(self.on_blur_change)
        blur_box.addWidget(self.slider_blur, stretch=5)
        sliders_layout.addLayout(blur_box)

        focal_box = QHBoxLayout()
        focal_box.addWidget(QLabel("Focal Distance (0-255):"), stretch=1)
        self.slider_focal = QSlider(Qt.Orientation.Horizontal)
        self.slider_focal.setRange(0, 255)
        self.slider_focal.setValue(self.focal_value)
        self.slider_focal.valueChanged.connect(self.on_focal_change)
        focal_box.addWidget(self.slider_focal, stretch=5)
        sliders_layout.addLayout(focal_box)

        dof_box = QHBoxLayout()
        dof_box.addWidget(QLabel("Focus Thickness (DoF):"), stretch=1)
        self.slider_dof = QSlider(Qt.Orientation.Horizontal)
        self.slider_dof.setRange(0, 100)
        self.slider_dof.setValue(self.dof_thickness)
        self.slider_dof.valueChanged.connect(self.on_dof_change)
        dof_box.addWidget(self.slider_dof, stretch=5)
        sliders_layout.addLayout(dof_box)

        main_layout.addLayout(sliders_layout)

        # --- Export Button ---
        self.btn_export = QPushButton("Export Full Composite Video Sequence")
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        self.btn_export.clicked.connect(self.export_video)
        main_layout.addWidget(self.btn_export)

    # --- Loading & Drag/Drop Mechanics ---
    def handle_drop(self, event, target):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if target == "color":
                self.color_path = file_path
                self.lbl_color.setText(f"Color Video Loaded:\n{os.path.basename(file_path)}")
                self.lbl_color.setStyleSheet("background-color: #1e3d1e; color: white; border: 2px solid green;")
            else:
                self.depth_path = file_path
                self.lbl_depth.setText(f"Depth Video Loaded:\n{os.path.basename(file_path)}")
                self.lbl_depth.setStyleSheet("background-color: #1e3d1e; color: white; border: 2px solid green;")
            self.init_video_captures()

    def load_initial_videos(self, c_path, d_path):
        if os.path.exists(c_path) and os.path.exists(d_path):
            self.color_path, self.depth_path = c_path, d_path
            self.lbl_color.setText(f"Color Loaded:\n{os.path.basename(c_path)}")
            self.lbl_color.setStyleSheet("background-color: #1e3d1e; color: white; border: 2px solid green;")
            self.lbl_depth.setText(f"Depth Loaded:\n{os.path.basename(d_path)}")
            self.lbl_depth.setStyleSheet("background-color: #1e3d1e; color: white; border: 2px solid green;")
            self.init_video_captures()

    def init_video_captures(self):
        if not self.color_path or not self.depth_path: return
        if self.cap_color: self.cap_color.release()
        if self.cap_depth: self.cap_depth.release()

        self.cap_color = cv2.VideoCapture(self.color_path)
        self.cap_depth = cv2.VideoCapture(self.depth_path)

        f_color = int(self.cap_color.get(cv2.CAP_PROP_FRAME_COUNT))
        f_depth = int(self.cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = min(f_color, f_depth)
        self.current_frame_idx = 0

        if self.total_frames > 0:
            self.btn_play.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.btn_sync_minus.setEnabled(True)
            self.btn_sync_plus.setEnabled(True)
            
            self.slider_timeline.setEnabled(True)
            self.slider_timeline.setRange(0, self.total_frames - 1)
            self.slider_timeline.setValue(0)
            
            # Match project refresh cadence dynamically to the source frame rate
            fps = self.cap_color.get(cv2.CAP_PROP_FPS) or 30.0
            self.play_timer.setInterval(int(1000 / fps))
            
            self.update_preview()

    # --- Timeline Scrubbing Interceptors ---
    def on_timeline_scrub(self, position):
        self.current_frame_idx = position
        self.update_preview()

    def on_scrub_start(self):
        # Pause playback loop while user actively drags the scrubber
        self.was_playing_before_scrub = self.is_playing
        if self.is_playing:
            self.play_timer.stop()

    def on_scrub_stop(self):
        # Restore playback loop once scrubbing ends
        if getattr(self, 'was_playing_before_scrub', False):
            self.play_timer.start()

    def toggle_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.play_timer.stop()
            self.btn_play.setText("▶ Play Loop")
            self.btn_play.setStyleSheet("background-color: #29b6f6; color: black; font-weight: bold;")
        else:
            self.is_playing = True
            self.play_timer.start()
            self.btn_play.setText("⏸ Pause")
            self.btn_play.setStyleSheet("background-color: orange; color: white; font-weight: bold;")

    def advance_loop_frame(self):
        if self.current_frame_idx >= self.total_frames - 1:
            self.current_frame_idx = 0
        else:
            self.current_frame_idx += 1
            
        self.slider_timeline.setValue(self.current_frame_idx)
        self.update_preview()

    def adjust_sync(self, delta):
        self.depth_frame_offset += delta
        self.lbl_sync_val.setText(f"Sync: {self.depth_frame_offset:+}")
        self.update_preview()

    # --- Coordinate Systems (Pan & Zoom) ---
    def start_pan(self, event):
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.drag_start = event.position()

    def execute_pan(self, event):
        if self.cap_color is None or self.drag_start is None: return
        delta = event.position() - self.drag_start
        self.pan_x += delta.x()
        self.pan_y += delta.y()
        self.drag_start = event.position()
        self.update_preview()

    def execute_zoom(self, event):
        if self.cap_color is None: return
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        new_zoom = max(0.1, min(self.zoom_level * factor, 30.0))
        
        mouse_x = event.position().x()
        mouse_y = event.position().y()
        c_w = max(self.canvas.width(), 100)
        
        virtual_x = mouse_x - (c_w // 2) if (self.chk_split.isChecked() and mouse_x > (c_w // 2)) else mouse_x

        self.pan_x = virtual_x - (virtual_x - self.pan_x) * (new_zoom / self.zoom_level)
        self.pan_y = mouse_y - (mouse_y - self.pan_y) * (new_zoom / self.zoom_level)
        self.zoom_level = new_zoom
        self.update_preview()

    # --- Polygonal Optical Filter Core ---
    def create_polygonal_kernel(self, size, sides):
        if sides < 3:
            kernel = np.zeros((size, size), dtype=np.float32)
            cv2.circle(kernel, (size // 2, size // 2), size // 2, 1, -1)
            return kernel
        kernel = np.zeros((size, size), dtype=np.uint8)
        center, radius = size // 2, size // 2
        points = []
        start_angle_offset = np.pi if sides == 3 else -np.pi / 2
        for i in range(sides):
            angle = start_angle_offset + (i * 2 * np.pi / sides)
            points.append([int(center + radius * np.cos(angle)), int(center + radius * np.sin(angle))])
        cv2.fillPoly(kernel, [np.array(points, dtype=np.int32)], 255)
        return kernel.astype(np.float32)

    def generate_bokeh_blur(self, img, size):
        if size <= 3: return img.copy()
        shape_blade_mapping = {
            "Circle": 0, "Triangle (3 Blade)": 3, "Diamond (4 Blade)": 4, "Pentagon (5 Blade)": 5,
            "Hexagon (6 Blade)": 6, "Heptagon (7 Blade)": 7, "Octagon (8 Blade)": 8,
            "Nonagon (9 Blade)": 9, "Decagon (10 Blade)": 10, "Hendecagon (11 Blade)": 11
        }
        sides = shape_blade_mapping.get(self.selected_shape, 0)
        kernel = self.create_polygonal_kernel(size, sides)
        kernel /= np.sum(kernel)
        return cv2.filter2D(img, -1, kernel)

    def process_depth_blur(self, color_img, depth_img):
        h, w = color_img.shape[:2]
        depth_img = cv2.resize(depth_img, (w, h))
        if len(depth_img.shape) == 3: depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        if self.chk_invert.isChecked(): depth_img = cv2.bitwise_not(depth_img)

        focal_target_matrix = np.full(depth_img.shape, self.focal_value, dtype=np.uint8)
        absolute_distance = cv2.absdiff(depth_img, focal_target_matrix).astype(np.float32)
        absolute_distance = np.maximum(0.0, absolute_distance - self.dof_thickness)
        
        sharpness_mask = np.clip(255.0 - (absolute_distance * 3.0), 0, 255).astype(np.uint8)
        if self.chk_grey.isChecked(): return cv2.cvtColor(sharpness_mask, cv2.COLOR_GRAY2BGR)

        depth_mask = sharpness_mask.astype(float) / 255.0
        bokeh_blurred_img = self.generate_bokeh_blur(color_img, self.max_blur_kernel)
        mask_3d = cv2.merge([depth_mask, depth_mask, depth_mask])
        return (color_img * mask_3d + bokeh_blurred_img * (1.0 - mask_3d)).astype('uint8')

    # --- Frame Retrieval & Split Clipping Render ---
    def update_preview(self):
        if not self.cap_color or not self.cap_depth: return

        # Synchronised offset pointer arithmetic
        target_d_frame = max(0, min(self.current_frame_idx + self.depth_frame_offset, int(self.cap_depth.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))

        self.cap_color.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.cap_depth.set(cv2.CAP_PROP_POS_FRAMES, target_d_frame)
        
        ret1, frame_color = self.cap_color.read()
        ret2, frame_depth = self.cap_depth.read()
        if not ret1 or not ret2: return

        c_w, c_h = max(self.canvas.width(), 100), max(self.canvas.height(), 100)
        processed_full = self.process_depth_blur(frame_color, frame_depth)
        orig_h, orig_w = frame_color.shape[:2]

        if self.chk_split.isChecked():
            v_w = c_w // 2
            scale_base = min(v_w / orig_w, c_h / orig_h)
            render_w = int(orig_w * scale_base * self.zoom_level)
            render_h = int(orig_h * scale_base * self.zoom_level)
            if render_w < 1 or render_h < 1: return

            full_resized_src = cv2.resize(frame_color, (render_w, render_h))
            full_resized_proc = cv2.resize(processed_full, (render_w, render_h))

            left_x = int((self.pan_x + (v_w // 2)) - (render_w / 2))
            top_y = int((self.pan_y + (c_h // 2)) - (render_h / 2))

            surface_src = np.zeros((c_h, v_w, 3), dtype=np.uint8)
            surface_proc = np.zeros((c_h, v_w, 3), dtype=np.uint8)

            src_x1, src_y1 = max(0, left_x), max(0, top_y)
            src_x2, src_y2 = min(v_w, left_x + render_w), min(c_h, top_y + render_h)
            img_x1, img_y1 = max(0, -left_x), max(0, -top_y)
            img_x2 = img_x1 + (src_x2 - src_x1)
            img_y2 = img_y1 + (src_y2 - src_y1)

            if (src_x2 > src_x1) and (src_y2 > src_y1):
                surface_src[src_y1:src_y2, src_x1:src_x2] = full_resized_src[img_y1:img_y2, img_x1:img_x2]
                surface_proc[src_y1:src_y2, src_x1:src_x2] = full_resized_proc[img_y1:img_y2, img_x1:img_x2]

            composite_view = np.zeros((c_h, c_w, 3), dtype=np.uint8)
            composite_view[:, :v_w] = surface_src
            composite_view[:, v_w:v_w*2] = surface_proc
            cv2.line(composite_view, (v_w, 0), (v_w, c_h), (37, 37, 37), 3) 
        else:
            scale_base = min(c_w / orig_w, c_h / orig_h)
            render_w = int(orig_w * scale_base * self.zoom_level)
            render_h = int(orig_h * scale_base * self.zoom_level)
            if render_w < 1 or render_h < 1: return

            full_resized_proc = cv2.resize(processed_full, (render_w, render_h))
            left_x = int((self.pan_x + (c_w // 2)) - (render_w / 2))
            top_y = int((self.pan_y + (c_h // 2)) - (render_h / 2))

            composite_view = np.zeros((c_h, c_w, 3), dtype=np.uint8)
            src_x1, src_y1 = max(0, left_x), max(0, top_y)
            src_x2, src_y2 = min(c_w, left_x + render_w), min(c_h, top_y + render_h)
            img_x1, img_y1 = max(0, -left_x), max(0, -top_y)
            img_x2 = img_x1 + (src_x2 - src_x1)
            img_y2 = img_y1 + (src_y2 - src_y1)

            if (src_x2 > src_x1) and (src_y2 > src_y1):
                composite_view[src_y1:src_y2, src_x1:src_x2] = full_resized_proc[img_y1:img_y2, img_x1:img_x2]

        rgb_img = cv2.cvtColor(composite_view, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_img.data, c_w, c_h, c_w * 3, QImage.Format.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(q_img))
        self.lbl_time_status.setText(f"Frame: {self.current_frame_idx + 1} / {self.total_frames}")

    # --- Listeners ---
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview()

    def on_shape_change(self, text):
        self.selected_shape = text
        self.update_preview()

    def on_blur_change(self, val):
        self.max_blur_kernel = val if val % 2 != 0 else val + 1
        self.update_preview()

    def on_focal_change(self, val):
        self.focal_value = val
        self.update_preview()

    def on_dof_change(self, val):
        self.dof_thickness = val
        self.update_preview()

    # --- Full Resolution Video Sequencer Export ---
    def export_video(self):
        if self.is_playing: self.toggle_playback()
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Composite Video Track", "", "MP4 Video (*.mp4)")
        if not file_path: return

        fps = self.cap_color.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self.cap_color.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap_color.get(cv2.CAP_PROP_FRAME_HEIGHT))
        t_depth = int(self.cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Block workspace layout changes during heavy background multi-thread rendering loops
        self.setEnabled(False)
        self.setWindowTitle("🎬 PROCESSING VIDEO SEQUENCES - PLEASE WAIT...")

        for idx in range(self.total_frames):
            td = max(0, min(idx + self.depth_frame_offset, t_depth - 1))
            self.cap_color.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.cap_depth.set(cv2.CAP_PROP_POS_FRAMES, td)
            
            ret1, f_color = self.cap_color.read()
            ret2, f_depth = self.cap_depth.read()
            if not ret1 or not ret2: break

            out.write(self.process_depth_blur(f_color, f_depth))

        out.release()
        self.setEnabled(True)
        self.setWindowTitle("Pro Depth Map Video Sync & Focus Engine - PySide6")
        QMessageBox.information(self, "Success", f"Composite video written natively at original full resolution dimensions to:\n{file_path}")
        self.update_preview()

    
if __name__ == "__main__":
    from PySide6.QtCore import QObject, Signal

    # Parse video paths from sys.argv (set by the plugin host before exec()ing this file)
    color_arg = sys.argv[1] if len(sys.argv) > 1 else None
    depth_arg = sys.argv[2] if len(sys.argv) > 2 else None

    app = QApplication.instance()

    if app is None:
        # ── Standalone mode: running outside LichtFeld ──────────────────────
        app = QApplication(sys.argv)
        window = ProDepthBlurVideoApp(color_arg, depth_arg)
        window.show()
        app.exec()          # no sys.exit() — keeps the process alive cleanly
    else:
        # ── Plugin mode: LichtFeld's QApplication already exists ─────────────
        # We must create the QMainWindow on the main GUI thread.
        # This script is exec()d from a background thread, so we need a
        # thread-safe trampoline.
        #
        # QTimer.singleShot(0) fails with "event dispatcher already destroyed"
        # when called from a non-main thread in some LichtFeld startup states.
        #
        # The reliable fix: create a temporary QObject with a Signal on the
        # background thread, then connect it to a slot.  Qt automatically
        # delivers the signal on the receiver's thread (the main thread) via
        # a queued connection — no timer, no invokeMethod, no @Slot decorator
        # needed.

        import __main__ as _main_module

        class _Trampoline(QObject):
            fire = Signal()

            def __init__(self, rgb, gsc):
                super().__init__()
                self._rgb = rgb
                self._gsc = gsc
                # Move this object to the main thread so the signal is
                # delivered there, not on the background thread.
                self.moveToThread(app.thread())
                self.fire.connect(self._create, Qt.ConnectionType.QueuedConnection)
                self.fire.emit()

            def _create(self):
                old = getattr(_main_module, '_dof_video_window', None)
                if old is not None:
                    try:
                        old.close()
                        old.deleteLater()
                    except Exception:
                        pass
                w = ProDepthBlurVideoApp(self._rgb, self._gsc)
                w.show()
                setattr(_main_module, '_dof_video_window', w)
                # Keep trampoline alive until after _create runs, then release it
                setattr(_main_module, '_dof_video_window_trampoline', None)

        # Store trampoline on __main__ so it isn't GC'd before the signal fires
        _main_module._dof_video_window_trampoline = _Trampoline(color_arg, depth_arg)
