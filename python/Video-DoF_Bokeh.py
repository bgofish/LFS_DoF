import sys
import os
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QProgressDialog, QLineEdit
from PySide6.QtGui import QIntValidator
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
        # (Label, H264_CRF, VP9_CRF, BPP)
        # Bitrate is calculated per-export from BPP * width * height * fps
        # so quality scales automatically with resolution and frame rate.
        self.VIDEO_QUALITIES = [
            ("Very High", 16, 24, 0.50),   # ~50 Mbps @ 1080p30 — pristine master quality
            ("High",      22, 31, 0.20),   # ~20 Mbps @ 1080p30 — crisp, no macroblocking
            ("Medium",    28, 38, 0.08),   # ~ 8 Mbps @ 1080p30 — balanced web standard
            ("Low",       36, 48, 0.03),   # ~ 3 Mbps @ 1080p30 — compressed preview
        ]
        self.export_quality_idx = 1  # default: High
        self.selected_codec      = "H.264 (Default)" 

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
        self.focal_value = 80  
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

        trim_container = QWidget()
        trim_layout = QVBoxLayout(trim_container)
        trim_layout.setContentsMargins(5, 0, 5, 0)
        int_validator = QIntValidator(0, 999999, self)
        start_trim_box = QHBoxLayout()
        start_trim_box.addWidget(QLabel("Trim Start:"))
        self.txt_trim_start = QLineEdit("0")
        self.txt_trim_start.setValidator(int_validator)
        self.txt_trim_start.setFixedWidth(60)
        self.txt_trim_start.textChanged.connect(self._on_trim_changed)
        start_trim_box.addWidget(self.txt_trim_start)
        end_trim_box = QHBoxLayout()
        end_trim_box.addWidget(QLabel("Trim End:"))
        self.txt_trim_end = QLineEdit("0")
        self.txt_trim_end.setValidator(int_validator)
        self.txt_trim_end.setFixedWidth(60)
        self.txt_trim_end.textChanged.connect(self._on_trim_changed)
        end_trim_box.addWidget(self.txt_trim_end)
        trim_layout.addLayout(start_trim_box)
        trim_layout.addLayout(end_trim_box)
        drop_layout.addWidget(self.lbl_color, stretch=2)
        drop_layout.addWidget(self.lbl_depth, stretch=2)
        drop_layout.addWidget(trim_container, stretch=1)
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

        # --- Bottom 3-column: bitrate | export | codec ---
        bottom_layout = QHBoxLayout()
        # Left: quality preset dropdown
        self.quality_menu = QComboBox()
        self.quality_menu.addItems([q[0] for q in self.VIDEO_QUALITIES])
        self.quality_menu.setCurrentIndex(self.export_quality_idx)
        self.quality_menu.currentIndexChanged.connect(self._on_quality_changed)
        left_col = QWidget()
        left_lay = QHBoxLayout(left_col)
        left_lay.setContentsMargins(0,0,0,0)
        left_lay.addWidget(QLabel("Export Quality: "))
        left_lay.addWidget(self.quality_menu)
        bottom_layout.addWidget(left_col, stretch=1)
        # Centre: export button
        self.btn_export = QPushButton("\U0001f3ac Export Layered Composite Video Sequence")
        self.btn_export.clicked.connect(self.export_video)
        self.btn_export.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 13px; padding: 10px;")
        bottom_layout.addWidget(self.btn_export, stretch=1)
        # Right: codec dropdown
        # Codec selector hidden for now (reserved for future use)
        self.codec_menu = QComboBox()
        self.codec_menu.addItems(["H.264 (Default)", "VP8 (WebM)"])
        self.codec_menu.currentTextChanged.connect(self._on_codec_changed)
        self.codec_menu.hide()
        right_col = QWidget()
        right_col.hide()
        bottom_layout.addWidget(right_col, stretch=1)
        main_layout.addLayout(bottom_layout)

    def _on_trim_changed(self, _text):
        if self.total_frames > 0:
            self._recalc_timeline()

    def _on_quality_changed(self, idx):
        self.export_quality_idx = idx

    def _on_codec_changed(self, text):
        self.selected_codec = text

    def _get_trim(self):
        try:
            start = int(self.txt_trim_start.text()) if self.txt_trim_start.text() else 0
            end   = int(self.txt_trim_end.text())   if self.txt_trim_end.text()   else 0
        except ValueError:
            start, end = 0, 0
        if (start + end) >= self.total_frames:
            start, end = 0, 0
        return start, end

    def _recalc_timeline(self):
        if self.total_frames <= 0:
            return
        start, end = self._get_trim()
        max_idx = (self.total_frames - 1) - end
        self.current_frame_idx = max(start, min(self.current_frame_idx, max_idx))
        self.slider_timeline.setRange(0, max(0, max_idx - start))
        self.slider_timeline.setValue(self.current_frame_idx - start)

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
        missing = [p for p in (c_path, d_path) if not p or not os.path.exists(p)]
        if missing:
            # Show which file is missing rather than a generic error
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "File Not Found",
                f"Could not find:\n{chr(10).join(missing)}\n\nUse the panel buttons to load videos manually.")
            return
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
        trim_start, _ = self._get_trim()
        self.current_frame_idx = trim_start + position
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
        trim_start, trim_end = self._get_trim()
        last_frame = (self.total_frames - 1) - trim_end
        if self.current_frame_idx >= last_frame:
            self.current_frame_idx = trim_start
        else:
            self.current_frame_idx += 1

        self.slider_timeline.setValue(self.current_frame_idx - trim_start)
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
        trim_start, trim_end = self._get_trim()
        trimmed_total = max(1, self.total_frames - trim_start - trim_end)
        trimmed_pos = (self.current_frame_idx - trim_start) + 1
        self.lbl_time_status.setText(f"Frame: {trimmed_pos} / {trimmed_total}")

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

        fps     = self.cap_color.get(cv2.CAP_PROP_FPS) or 30.0
        width   = int(self.cap_color.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(self.cap_color.get(cv2.CAP_PROP_FRAME_HEIGHT))
        t_depth = int(self.cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))

        trim_start, trim_end  = self._get_trim()
        frame_range_start     = trim_start
        frame_range_end       = (self.total_frames - 1) - trim_end  # inclusive
        export_frame_count    = max(1, frame_range_end - frame_range_start + 1)

        label, crf_h264, crf_vp9, bpp = self.VIDEO_QUALITIES[self.export_quality_idx]
        # Scale bitrate by actual resolution and frame rate
        bitrate_bps = int(width * height * fps * bpp)

        # --- PyAV encode (same approach as depthmap_panel) ---
        # Tries HW encoders first, falls back through SW codecs until one works.
        CODEC_PRIORITY = ["h264_nvenc", "h264_amf", "h264_qsv", "libopenh264", "libx264", "mpeg4"]
        try:
            import av as _av
            _av_available = True
        except ImportError:
            _av_available = False

        # --- Progress dialog ---
        progress = QProgressDialog("Exporting video…", "Cancel", 0, export_frame_count, self)
        progress.setWindowTitle("🎬 Exporting")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        cancelled = False
        used_codec = None

        if _av_available:
            # ── PyAV path ──────────────────────────────────────────────────────
            from PIL import Image as _PILImage

            container = None
            stream    = None

            # Probe codecs with the first frame
            self.cap_color.set(cv2.CAP_PROP_POS_FRAMES, frame_range_start)
            self.cap_depth.set(cv2.CAP_PROP_POS_FRAMES,
                               max(0, min(frame_range_start + self.depth_frame_offset, t_depth - 1)))
            ret1, f_color = self.cap_color.read()
            ret2, f_depth = self.cap_depth.read()
            first_frame = self.process_depth_blur(f_color, f_depth) if (ret1 and ret2) else None

            if first_frame is not None:
                first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                first_img = _PILImage.fromarray(first_rgb)

                for codec in CODEC_PRIORITY:
                    try:
                        if container:
                            try: container.close()
                            except: pass
                        container = _av.open(file_path, mode="w", format="mp4")

                        # Build codec options dict BEFORE add_stream opens the context.
                        # IMPORTANT: CRF and bitrate targeting are mutually exclusive in x264 —
                        # if both are set, CRF wins and the bitrate target is ignored entirely.
                        # Use pure bitrate-mode (b:v + maxrate + bufsize) for all codecs.
                        maxrate   = int(bitrate_bps * 1.25)
                        bufsize   = int(bitrate_bps * 2)
                        if any(x in codec for x in ("nvenc", "amf", "qsv")):
                            # HW encoders: bitrate mode via stream property + rc options
                            codec_opts = {
                                "rc":      "vbr",
                                "maxrate": str(maxrate),
                                "bufsize": str(bufsize),
                            }
                        elif codec in ("libx264", "libopenh264"):
                            # x264 VBR: b:v sets target, maxrate+bufsize enforce ceiling
                            # Do NOT set crf — it overrides bitrate completely
                            codec_opts = {
                                "b:v":     str(bitrate_bps),
                                "maxrate": str(maxrate),
                                "bufsize": str(bufsize),
                                "preset":  "medium",
                            }
                        else:
                            # mpeg4: b:v is the only reliable lever
                            codec_opts = {
                                "b:v":     str(bitrate_bps),
                                "maxrate": str(maxrate),
                                "bufsize": str(bufsize),
                            }

                        stream = container.add_stream(codec, rate=int(fps), options=codec_opts)
                        stream.width    = width
                        stream.height   = height
                        stream.pix_fmt  = "yuv420p"
                        stream.bit_rate = bitrate_bps

                        frm0 = _av.VideoFrame.from_image(first_img)
                        frm0.pts = 0
                        list(stream.encode(frm0))
                        used_codec = codec
                        break
                    except Exception:
                        stream = None

            if used_codec is None:
                if container:
                    try: container.close()
                    except: pass
                QMessageBox.warning(self, "Export Error", "No working codec found. Install PyAV with libx264 support.")
                progress.close()
                return

            # Encode remaining frames (frame 0 already encoded above)
            for i, idx in enumerate(range(frame_range_start, frame_range_end + 1)):
                if progress.wasCanceled():
                    cancelled = True
                    break

                if i == 0:
                    progress.setValue(1)
                    QApplication.processEvents()
                    continue  # already encoded during probe

                td = max(0, min(idx + self.depth_frame_offset, t_depth - 1))
                self.cap_color.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self.cap_depth.set(cv2.CAP_PROP_POS_FRAMES, td)
                ret1, f_color = self.cap_color.read()
                ret2, f_depth = self.cap_depth.read()
                if not ret1 or not ret2:
                    break

                processed = self.process_depth_blur(f_color, f_depth)
                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                img = _PILImage.fromarray(rgb)
                frame = _av.VideoFrame.from_image(img)
                frame.pts = i
                for pkt in stream.encode(frame):
                    container.mux(pkt)

                progress.setValue(i + 1)
                QApplication.processEvents()

            # Flush + close
            if not cancelled:
                for pkt in stream.encode():
                    container.mux(pkt)
            container.close()

        else:
            # ── OpenCV fallback (no bitrate control) ──────────────────────────
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            used_codec = "mp4v (OpenCV fallback)"

            for i, idx in enumerate(range(frame_range_start, frame_range_end + 1)):
                if progress.wasCanceled():
                    cancelled = True
                    break
                td = max(0, min(idx + self.depth_frame_offset, t_depth - 1))
                self.cap_color.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self.cap_depth.set(cv2.CAP_PROP_POS_FRAMES, td)
                ret1, f_color = self.cap_color.read()
                ret2, f_depth = self.cap_depth.read()
                if not ret1 or not ret2:
                    break
                writer.write(self.process_depth_blur(f_color, f_depth))
                progress.setValue(i + 1)
                QApplication.processEvents()
            writer.release()

        progress.close()

        if cancelled:
            try:
                os.remove(file_path)
            except OSError:
                pass
            QMessageBox.information(self, "Cancelled", "Export was cancelled. Partial file removed.")
        else:
            QMessageBox.information(
                self, "Success",
                f"Exported: {label} quality — {bitrate_bps / 1_000_000:.1f} Mbps (BPP {bpp} @ {width}x{height} {fps:.0f}fps)\n"
                f"Codec: {used_codec}\n"
                f"Frames: {frame_range_start}–{frame_range_end} ({export_frame_count} frames)\n\n{file_path}"
            )

        self.update_preview()

    
if __name__ == "__main__":
    from PySide6.QtCore import QObject, Signal

    # Parse video paths from sys.argv (set by the plugin host before exec()ing this file)
    color_arg = sys.argv[1] if len(sys.argv) > 1 else None
    depth_arg = sys.argv[2] if len(sys.argv) > 2 else None
    # Debug — remove after confirming paths arrive correctly
    import logging as _log
    _log.basicConfig(filename=__file__ + ".log", level=_log.DEBUG, force=True)
    _log.debug(f"argv={sys.argv}  color_arg={color_arg}  depth_arg={depth_arg}")


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
