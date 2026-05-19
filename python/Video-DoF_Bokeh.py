import sys
import os
import cv2
import numpy as np
from PySide6.QtCore import Qt, QMetaObject
from PySide6.QtGui import QImage, QPixmap, QIntValidator
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QCheckBox, QLineEdit,
                             QComboBox, QPushButton, QFileDialog, QMessageBox, QProgressDialog)

class ProDepthBlurVideoApp(QMainWindow):
    def __init__(self, init_color_path=None, init_depth_path=None):
        super().__init__()
        self.setWindowTitle("Pro Depth Map Video Sync & Focus Tool - PySide6 Engine")
        self.setGeometry(100, 100, 1400, 980)

        # Video State variables
        self.color_path = None
        self.depth_path = None
        self.cap_color = None
        self.cap_depth = None
        self.raw_total_frames = 0
        self.current_frame_idx = 0
        
        # Navigation Coordinate Space (Pan & Zoom state)
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_start = None
        
        # Rendering pipeline lifecycle protectors
        self.ui_initialized = False 
        self._current_qimage = None 
        self._current_pixmap = None

        # Optics and Calibration settings
        self.max_blur_kernel = 20 
        self.focal_value = 128  
        self.dof_thickness = 20  
        self.selected_shape = "Hexagon"
        self.target_bitrate_mbps = 20  
        self.selected_codec = "H.264 (Default)"  

        self.setup_ui()
        self.ui_initialized = True

        if init_color_path and init_depth_path:
            self.load_initial_files(init_color_path, init_depth_path)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Drop Zones & Trim Options Row ---
        drop_layout = QHBoxLayout()
        
        self.lbl_color = QLabel("Drag & Drop COLOR Video Here")
        self.lbl_color.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_color.setStyleSheet("background-color: #2b2b2b; color: #aaaaaa; border: 2px dashed #555555; font-size: 13px;")
        self.lbl_color.setFixedHeight(65)
        
        self.lbl_depth = QLabel("Drag & Drop DEPTH Video Here")
        self.lbl_depth.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_depth.setStyleSheet("background-color: #2b2b2b; color: #aaaaaa; border: 2px dashed #555555; font-size: 13px;")
        self.lbl_depth.setFixedHeight(65)

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
        self.txt_trim_start.textChanged.connect(self.on_trim_values_changed)
        start_trim_box.addWidget(self.txt_trim_start)
        
        end_trim_box = QHBoxLayout()
        end_trim_box.addWidget(QLabel("Trim End:"))
        self.txt_trim_end = QLineEdit("0")
        self.txt_trim_end.setValidator(int_validator)
        self.txt_trim_end.setFixedWidth(60)
        self.txt_trim_end.textChanged.connect(self.on_trim_values_changed)
        end_trim_box.addWidget(self.txt_trim_end)
        
        trim_layout.addLayout(start_trim_box)
        trim_layout.addLayout(end_trim_box)

        drop_layout.addWidget(self.lbl_color, stretch=2)
        drop_layout.addWidget(self.lbl_depth, stretch=2)
        drop_layout.addWidget(trim_container, stretch=1)
        main_layout.addLayout(drop_layout)

        # --- Interactive Control Canvas ---
        self.canvas = QLabel()
        self.canvas.setStyleSheet("background-color: #111111;")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.canvas, stretch=1)

        # Connect mouse events directly to canvas layout frame targets for Pan and Zoom
        self.canvas.mousePressEvent = self.start_pan
        self.canvas.mouseMoveEvent = self.execute_pan
        self.canvas.wheelEvent = self.execute_zoom

        # --- Full-Width Yellow Video Scrubber Track ---
        scrubber_container = QWidget()
        scrubber_layout = QVBoxLayout(scrubber_container)
        scrubber_layout.setContentsMargins(0, 5, 0, 5)

        scrubber_bar_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev Frame")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_prev.setFixedWidth(100)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.valueChanged.connect(self.on_scrubber_scrolled)
        self.timeline_slider.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #444; height: 6px; background: #222; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #FFD700; border-radius: 3px; }
            QSlider::handle:horizontal { background: #FFD700; border: 1px solid #B8860B; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px; }
        """)

        self.btn_next = QPushButton("Next Frame ▶")
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_next.setFixedWidth(100)

        scrubber_bar_layout.addWidget(self.btn_prev)
        scrubber_bar_layout.addWidget(self.timeline_slider, stretch=1)
        scrubber_bar_layout.addWidget(self.btn_next)
        scrubber_layout.addLayout(scrubber_bar_layout)

        self.lbl_status = QLabel("Frame: 0 / 0  |  Depth Target Frame: 0")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 13px; color: #FFFFFF; margin-top: 2px;")
        scrubber_layout.addWidget(self.lbl_status)

        main_layout.addWidget(scrubber_container)

        # --- Top Toggles Row ---
        toggle_layout = QHBoxLayout()
        self.chk_split = QCheckBox("Split View Mode")
        self.chk_split.setChecked(True)
        self.chk_split.stateChanged.connect(self.update_preview)
        
        self.chk_invert = QCheckBox("Invert Depth Mask")
        self.chk_invert.stateChanged.connect(self.update_preview)
        
        self.chk_grey = QCheckBox("View Live Focus Mask Only")
        self.chk_grey.stateChanged.connect(self.update_preview)

        self.shape_menu = QComboBox()
        self.shape_menu.addItems([
            "Circle", "Triangle (3 Blade)", "Diamond (4 Blade)", "Pentagon (5 Blade)", 
            "Hexagon (6 Blade)", "Heptagon (7 Blade)", "Octagon (8 Blade)", 
            "Nonagon (9 Blade)", "Decagon (10 Blade)", "Hendecagon (11 Blade)"
        ])
        self.shape_menu.currentTextChanged.connect(self.on_shape_change)

        toggle_layout.addWidget(self.chk_split)
        toggle_layout.addWidget(self.chk_invert)
        toggle_layout.addWidget(self.chk_grey)
        toggle_layout.addWidget(QLabel("Bokeh Shape:"))
        toggle_layout.addWidget(self.shape_menu)
        toggle_layout.addStretch()
        main_layout.addLayout(toggle_layout)

        # --- Settings Tuning and Layout Configurations Panel ---
        sliders_layout = QVBoxLayout()
        
        blur_box = QHBoxLayout()
        blur_box.addWidget(QLabel("Max Bokeh Diameter:"), stretch=2)
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(3, 151)
        self.slider_blur.setSingleStep(2)
        self.slider_blur.setValue(self.max_blur_kernel)
        self.slider_blur.valueChanged.connect(self.on_blur_change)
        blur_box.addWidget(self.slider_blur, stretch=5)
        sliders_layout.addLayout(blur_box)

        focal_box = QHBoxLayout()
        focal_box.addWidget(QLabel("Focal Distance (0-255):"), stretch=2)
        self.slider_focal = QSlider(Qt.Orientation.Horizontal)
        self.slider_focal.setRange(0, 255)
        self.slider_focal.setValue(self.focal_value)
        self.slider_focal.valueChanged.connect(self.on_focal_change)
        focal_box.addWidget(self.slider_focal, stretch=5)
        sliders_layout.addLayout(focal_box)

        dof_box = QHBoxLayout()
        dof_box.addWidget(QLabel("Focus Thickness (DoF):"), stretch=2)
        self.slider_dof = QSlider(Qt.Orientation.Horizontal)
        self.slider_dof.setRange(0, 100)
        self.slider_dof.setValue(self.dof_thickness)
        self.slider_dof.valueChanged.connect(self.on_dof_change)
        dof_box.addWidget(self.slider_dof, stretch=5)
        sliders_layout.addLayout(dof_box)
        
        main_layout.addLayout(sliders_layout)

        # --- THREE-COLUMN STRUCTURAL BOTTOM CONTROL GRID PANEL ---
        bottom_grid_layout = QHBoxLayout()
        
        # 1/3 COLUMN LEFT: Compression Quality Slider Control
        col_left_widget = QWidget()
        col_left_layout = QHBoxLayout(col_left_widget)
        col_left_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_bitrate_readout = QLabel(f"Export Quality: {self.target_bitrate_mbps} Mbps ")
        self.slider_bitrate = QSlider(Qt.Orientation.Horizontal)
        self.slider_bitrate.setRange(1, 50)  
        self.slider_bitrate.setValue(self.target_bitrate_mbps)
        self.slider_bitrate.valueChanged.connect(self.on_bitrate_change)
        col_left_layout.addWidget(self.lbl_bitrate_readout)
        col_left_layout.addWidget(self.slider_bitrate, stretch=1)
        bottom_grid_layout.addWidget(col_left_widget, stretch=1)
        
        # 1/3 COLUMN MIDDLE: The Green Process Execution Button
        self.btn_export = QPushButton("🎬 Export Layered Composite Video Sequence")
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 13px; padding: 10px;")
        self.btn_export.clicked.connect(self.export_video)
        bottom_grid_layout.addWidget(self.btn_export, stretch=1)
        
        # 1/3 COLUMN RIGHT: Codec Dropdown Menu Selector
        col_right_widget = QWidget()
        col_right_layout = QHBoxLayout(col_right_widget)
        col_right_layout.setContentsMargins(0, 0, 0, 0)
        self.codec_menu = QComboBox()
        self.codec_menu.addItems(["H.264 (Default)", "VP8 (For WebM Container Encodings)"])
        self.codec_menu.currentTextChanged.connect(self.on_codec_change)
        col_right_layout.addStretch()
        col_right_layout.addWidget(QLabel("Video Codec: "))
        col_right_layout.addWidget(self.codec_menu)
        bottom_grid_layout.addWidget(col_right_widget, stretch=1)

        main_layout.addLayout(bottom_grid_layout)

    def load_initial_files(self, color_path, depth_path):
        c_path = color_path if isinstance(color_path, list) else color_path
        d_path = depth_path if isinstance(depth_path, list) else depth_path

        if os.path.exists(str(c_path)) and os.path.exists(str(d_path)):
            self.color_path = str(c_path)
            self.depth_path = str(d_path)
            self.init_video_captures()
        else:
            QMessageBox.warning(self, "Path Error", "One or both initialization video target paths missing.")

    # --- DURABLE DROP HANDLER (With fixed array extraction unpacked elements hooks) ---
    def handle_drop(self, event, target):
        urls = event.mimeData().urls()
        if urls and len(urls) > 0:
            first_url = urls[0]  # Extracts element to bypass list AttributeErrors cleanly
            file_path = first_url.toLocalFile()
            
            if target == "color":
                self.color_path = file_path
                self.lbl_color.setText(f"Color Video Loaded:\n{os.path.basename(file_path)}")
                self.lbl_color.setStyleSheet("background-color: #1e3d1e; color: white; border: 2px solid green;")
            else:
                self.depth_path = file_path
                self.lbl_depth.setText(f"Depth Video Loaded:\n{os.path.basename(file_path)}")
                self.lbl_depth.setStyleSheet("background-color: #1e3d1e; color: white; border: 2px solid green;")
            self.init_video_captures()

    def init_video_captures(self):
        if not self.color_path or not self.depth_path: return
        
        if self.cap_color: self.cap_color.release()
        if self.cap_depth: self.cap_depth.release()

        self.cap_color = cv2.VideoCapture(self.color_path)
        self.cap_depth = cv2.VideoCapture(self.depth_path)

        c_frames = int(self.cap_color.get(cv2.CAP_PROP_FRAME_COUNT))
        d_frames = int(self.cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))
        self.raw_total_frames = min(c_frames, d_frames)
        
        if self.raw_total_frames > 0:
            self.recalculate_timeline_ranges()
            self.btn_export.setEnabled(True)
        else:
            QMessageBox.warning(self, "Video Error", "Could not open frames from one or both video clips.")

    def get_trim_values(self):
        try:
            start = int(self.txt_trim_start.text()) if self.txt_trim_start.text() else 0
            end = int(self.txt_trim_end.text()) if self.txt_trim_end.text() else 0
        except ValueError:
            start, end = 0, 0
        
        if (start + end) >= self.raw_total_frames:
            start, end = 0, 0
        return start, end

    def recalculate_timeline_ranges(self):
        if self.raw_total_frames <= 0: return
        trim_start, trim_end = self.get_trim_values()
        
        allowed_max_idx = (self.raw_total_frames - 1) - trim_end
        allowed_min_idx = trim_start
        
        self.current_frame_idx = max(allowed_min_idx, min(self.current_frame_idx, allowed_max_idx))
        
        slider_max = max(0, allowed_max_idx - allowed_min_idx)
        self.timeline_slider.setRange(0, slider_max)
        self.timeline_slider.setValue(self.current_frame_idx - allowed_min_idx)
        
        self.update_preview()

    def on_trim_values_changed(self, text):
        if self.ui_initialized and self.raw_total_frames > 0:
            self.recalculate_timeline_ranges()

    def on_scrubber_scrolled(self, val):
        if self.ui_initialized and self.raw_total_frames > 0:
            trim_start, _ = self.get_trim_values()
            target_absolute_frame = val + trim_start
            if self.current_frame_idx != target_absolute_frame:
                self.current_frame_idx = target_absolute_frame
                self.update_preview()

    def prev_frame(self):
        trim_start, _ = self.get_trim_values()
        if self.current_frame_idx > trim_start:
            self.current_frame_idx -= 1
            self.timeline_slider.setValue(self.current_frame_idx - trim_start)

    def next_frame(self):
        trim_start, trim_end = self.get_trim_values()
        allowed_max_idx = (self.raw_total_frames - 1) - trim_end
        if self.current_frame_idx < allowed_max_idx:
            self.current_frame_idx += 1
            self.timeline_slider.setValue(self.current_frame_idx - trim_start)

    # --- Mouse Pan & Zoom Vector Interceptors ---
    def start_pan(self, event):
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.drag_start = event.position()

    def execute_pan(self, event):
        if not self.cap_color or self.drag_start is None: return
        delta = event.position() - self.drag_start
        self.pan_x += delta.x()
        self.pan_y += delta.y()
        self.drag_start = event.position()
        self.update_preview()

    def execute_zoom(self, event):
        if not self.cap_color: return
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
     
    #########################################    
    # --- Aperture Core Convolution Logic ---
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

    # --- Live Render Display Engine with Interactive Clip Masking ---
    def update_preview(self):
        if not self.ui_initialized or not self.cap_color or not self.cap_depth: return

        self.cap_color.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.cap_depth.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        
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

        composite_rgb = cv2.cvtColor(composite_view, cv2.COLOR_BGR2RGB)
        self._current_qimage = QImage(composite_rgb.data, c_w, c_h, c_w * 3, QImage.Format.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(self._current_qimage)
        self.canvas.setPixmap(self._current_pixmap)
        
        self.lbl_status.setText(f"Color Frame: {self.current_frame_idx + 1} / {self.raw_total_frames}  |  Depth Target Frame: {self.current_frame_idx + 1}")

    def on_shape_change(self, text):
        self.selected_shape = text
        self.update_preview()

    def on_codec_change(self, text):
        self.selected_codec = text

    def on_blur_change(self, val):
        self.max_blur_kernel = val if val % 2 != 0 else val + 1
        self.update_preview()

    def on_focal_change(self, val):
        self.focal_value = val
        self.update_preview()

    def on_dof_change(self, val):
        self.dof_thickness = val
        self.update_preview()

    def on_bitrate_change(self, val):
        self.target_bitrate_mbps = val
        self.lbl_bitrate_readout.setText(f"Export Quality: {self.target_bitrate_mbps} Mbps ")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.ui_initialized: self.update_preview()

    def closeEvent(self, event):
        if self.cap_color: self.cap_color.release()
        if self.cap_depth: self.cap_depth.release()
        self.canvas.clear()
        super().closeEvent(event)

    def export_video(self):
        if not self.cap_color or not self.cap_depth: return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Depth Composited Video", "", 
            "MP4 Video (*.mp4);;Apple QuickTime (*.mov);;Matroska Video (*.mkv);;AVI Video (*.avi);;All Files (*.*)"
        )
        if not save_path: return

        fps = self.cap_color.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self.cap_color.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap_color.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if "VP8" in self.selected_codec:
            fourcc = cv2.VideoWriter_fourcc(*'VP80')  
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

        trim_start, trim_end = self.get_trim_values()
        start_export_idx = trim_start
        end_export_idx = (self.raw_total_frames - 1) - trim_end
        export_duration_length = (end_export_idx - start_export_idx) + 1

        progress = QProgressDialog("Processing Composited Bokeh Video Layers...", "Cancel", 0, export_duration_length, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        out = None
        try:
            raw_bitrate_bps = int(self.target_bitrate_mbps * 1_000_000)
            out = cv2.VideoWriter(
                filename=save_path,
                fourcc=fourcc,
                fps=fps,
                frameSize=(width, height),
                params=[cv2.VIDEOWRITER_PROP_BITRATE, raw_bitrate_bps] if hasattr(cv2, 'VIDEOWRITER_PROP_BITRATE') else []
            )
            
            if not out.isOpened():
                raise IOError("OpenCV frame container could not allocate your export path on disk.")

            for progress_counter, idx in enumerate(range(start_export_idx, end_export_idx + 1)):
                if progress.wasCanceled(): break
                
                self.cap_color.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self.cap_depth.set(cv2.CAP_PROP_POS_FRAMES, idx)
                
                ret1, f_color = self.cap_color.read()
                ret2, f_depth = self.cap_depth.read()
                if not ret1 or not ret2: break

                composited_output_frame = self.process_depth_blur(f_color, f_depth)
                out.write(composited_output_frame)
                
                progress.setValue(progress_counter + 1)
                
            QMessageBox.information(self, "Export Complete", f"Video rendered successfully to your target folder:\n{save_path}")

        except Exception as error_msg:
            QMessageBox.critical(self, "Export Error", f"The background file-write stream dropped:\n{str(error_msg)}")
        
        finally:
            if out is not None: out.release()
            progress.close()
            self.update_preview()

# --- CROSS-THREAD PLUGIN LIFECYCLE MARSHAL ---
if __name__ == "__main__":
    import gc
    from PySide6.QtWidgets import QApplication

    existing_app = QApplication.instance()
    if existing_app is not None:
        print("🧹 Cleaning up lingering dead QApplication instance from memory...")
        existing_app.closeAllWindows()
        existing_app.quit()
        del existing_app
        gc.collect()

    color_arg = None
    depth_arg = None
    if hasattr(sys, 'argv') and isinstance(sys.argv, list):
        video_args = [arg for arg in sys.argv if isinstance(arg, str) and arg.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        if len(video_args) >= 2:
            color_arg, depth_arg = video_args, video_args

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        window = ProDepthBlurVideoApp(color_arg, depth_arg)
        window.show()
        sys.exit(app.exec())
    else:
        print("🔌 Host app instance detected. Marshaling video UI instantiation onto the Main Thread...")
        global plugin_window
        def execute_ui_on_main_thread():
            global plugin_window
            if 'plugin_window' in globals():
                try:
                    plugin_window.close()
                    plugin_window.deleteLater()
                except: pass
            plugin_window = ProDepthBlurVideoApp(color_arg, depth_arg)
            plugin_window.show()
            app.processEvents()
        
        QMetaObject.invokeMethod(app, execute_ui_on_main_thread, Qt.ConnectionType.QueuedConnection)

