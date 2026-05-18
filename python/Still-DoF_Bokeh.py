import sys
import os
import cv2
import numpy as np

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPolygonF
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QCheckBox, 
                               QComboBox, QPushButton, QFileDialog, QMessageBox)


class ProDepthBlurQtApp(QMainWindow):
    def __init__(self, init_color_path=None, init_depth_path=None):
        super().__init__()
        self.setWindowTitle("Pro Depth Map Focus Tool - PyQt6 Engine")
        self.setGeometry(100, 100, 1400, 950)

        # Core Asset Trackers
        self.img_color = None
        self.img_depth = None
        
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

        # Handle command-line arguments passed on startup
        if init_color_path and init_depth_path:
            self.load_initial_files(init_color_path, init_depth_path)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Drag and Drop Panels ---
        drop_layout = QHBoxLayout()
        
        self.lbl_color = QLabel("Drag & Drop COLOR PNG Here")
        self.lbl_color.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_color.setStyleSheet("background-color: #2b2b2b; color: #aaaaaa; border: 2px dashed #555555; font-size: 14px;")
        self.lbl_color.setFixedHeight(70)
        
        self.lbl_depth = QLabel("Drag & Drop DEPTH PNG Here")
        self.lbl_depth.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_depth.setStyleSheet("background-color: #2b2b2b; color: #aaaaaa; border: 2px dashed #555555; font-size: 14px;")
        self.lbl_depth.setFixedHeight(70)

        # Enable native OS Drag and Drop
        self.lbl_color.setAcceptDrops(True)
        self.lbl_depth.setAcceptDrops(True)
        
        self.lbl_color.dragEnterEvent = lambda e: e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
        self.lbl_depth.dragEnterEvent = lambda e: e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
        
        self.lbl_color.dropEvent = lambda e: self.handle_drop(e, "color")
        self.lbl_depth.dropEvent = lambda e: self.handle_drop(e, "depth")

        drop_layout.addWidget(self.lbl_color)
        drop_layout.addWidget(self.lbl_depth)
        main_layout.addLayout(drop_layout)

        # --- Interactive Control Canvas Viewport ---
        self.canvas = QLabel()
        self.canvas.setStyleSheet("background-color: #111111;")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.canvas, stretch=1)

        # Connect mouse events to the canvas for Pan and Zoom
        self.canvas.mousePressEvent = self.start_pan
        self.canvas.mouseMoveEvent = self.execute_pan
        self.canvas.wheelEvent = self.execute_zoom

        # --- Mode Control Toggles Bar ---
        toggle_layout = QHBoxLayout()
        
        self.chk_split = QCheckBox("Split View Mode")
        self.chk_split.setChecked(True)
        self.chk_split.stateChanged.connect(self.update_preview)
        
        self.chk_invert = QCheckBox("Invert Depth Mask")
        self.chk_invert.stateChanged.connect(self.update_preview)
        
        self.chk_grey = QCheckBox("View Isolated Focus Mask Only")
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
        toggle_layout.addWidget(QLabel("Info: Right-Click Drag = PAN | Scroll Wheel = ZOOM"))
        main_layout.addLayout(toggle_layout)

        # --- Optical Settings Sliders ---
        sliders_layout = QVBoxLayout()
        
        # 1. Blur Kernel Size
        blur_box = QHBoxLayout()
        blur_box.addWidget(QLabel("Max Bokeh Diameter:"), stretch=1)
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(3, 151)
        self.slider_blur.setSingleStep(2)
        self.slider_blur.setValue(self.max_blur_kernel)
        self.slider_blur.valueChanged.connect(self.on_blur_change)
        blur_box.addWidget(self.slider_blur, stretch=5)
        sliders_layout.addLayout(blur_box)

        # 2. Focal Plane Locator
        focal_box = QHBoxLayout()
        focal_box.addWidget(QLabel("Focal Distance (0-255):"), stretch=1)
        self.slider_focal = QSlider(Qt.Orientation.Horizontal)
        self.slider_focal.setRange(0, 255)
        self.slider_focal.setValue(self.focal_value)
        self.slider_focal.valueChanged.connect(self.on_focal_change)
        focal_box.addWidget(self.slider_focal, stretch=5)
        sliders_layout.addLayout(focal_box)

        # 3. Depth of Field Margin
        dof_box = QHBoxLayout()
        dof_box.addWidget(QLabel("Focus Thickness (DoF):"), stretch=1)
        self.slider_dof = QSlider(Qt.Orientation.Horizontal)
        self.slider_dof.setRange(0, 100)
        self.slider_dof.setValue(self.dof_thickness)
        self.slider_dof.valueChanged.connect(self.on_dof_change)
        dof_box.addWidget(self.slider_dof, stretch=5)
        sliders_layout.addLayout(dof_box)

        main_layout.addLayout(sliders_layout)

        # --- Master Export Execution Panel Button ---
        self.btn_export = QPushButton("Export Full Resolution Composite Image")
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        self.btn_export.clicked.connect(self.export_image)
        main_layout.addWidget(self.btn_export)

    # --- CLI Loading Logic ---
    def load_initial_files(self, color_path, depth_path):
        if os.path.exists(color_path) and os.path.exists(depth_path):
            self.img_color = cv2.imread(color_path)
            self.img_depth = cv2.imread(depth_path)
            
            if self.img_color is not None and self.img_depth is not None:
                self.lbl_color.setText(f"Color Loaded:\n{os.path.basename(color_path)}")
                self.lbl_color.setStyleSheet("background-color: #1e3d1e; color: white; border: 2px solid green;")
                self.lbl_depth.setText(f"Depth Loaded:\n{os.path.basename(depth_path)}")
                self.lbl_depth.setStyleSheet("background-color: #1e3d1e; color: white; border: 2px solid green;")
                
                self.btn_export.setEnabled(True)
                self.zoom_level = 1.0
                self.pan_x, self.pan_y = 0.0, 0.0
                # Wait briefly for application to build its initial UI boundaries before displaying 
                self.update_preview()
            else:
                QMessageBox.warning(self, "Load Error", "One of the provided files could not be read as an image.")
        else:
            QMessageBox.warning(self, "File Error", "One or both of the command line paths do not exist.")

    # --- PyQt Native Drag and Drop Parsing ---
    def handle_drop(self, event, target):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if target == "color":
                self.img_color = cv2.imread(file_path)
                self.lbl_color.setText(f"Color Loaded:\n{os.path.basename(file_path)}")
                self.lbl_color.setStyleSheet("background-color: #1e3d1e; color: white; border: 2px solid green;")
            else:
                self.img_depth = cv2.imread(file_path)
                self.lbl_depth.setText(f"Depth Loaded:\n{os.path.basename(file_path)}")
                self.lbl_depth.setStyleSheet("background-color: #1e3d1e; color: white; border: 2px solid green;")

            if self.img_color is not None and self.img_depth is not None:
                self.btn_export.setEnabled(True)
                self.zoom_level = 1.0
                self.pan_x, self.pan_y = 0.0, 0.0
                self.update_preview()

    # --- Mouse Event Matrices (Pan & Zoom) ---
    def start_pan(self, event):
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.drag_start = event.position()

    def execute_pan(self, event):
        if self.img_color is None or self.drag_start is None: return
        delta = event.position() - self.drag_start
        self.pan_x += delta.x()
        self.pan_y += delta.y()
        self.drag_start = event.position()
        self.update_preview()

    def execute_zoom(self, event):
        if self.img_color is None: return
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

    # --- Live Render Display Engine ---
    def update_preview(self):
        if self.img_color is None or self.img_depth is None: return

        c_w, c_h = max(self.canvas.width(), 100), max(self.canvas.height(), 100)
        processed_full = self.process_depth_blur(self.img_color, self.img_depth)
        orig_h, orig_w = self.img_color.shape[:2]

        if self.chk_split.isChecked():
            v_w = c_w // 2
            scale_base = min(v_w / orig_w, c_h / orig_h)
            render_w = int(orig_w * scale_base * self.zoom_level)
            render_h = int(orig_h * scale_base * self.zoom_level)
            if render_w < 1 or render_h < 1: return

            full_resized_src = cv2.resize(self.img_color, (render_w, render_h))
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

    # --- Layout Event Listeners ---
    def showEvent(self, event):
        # Trigger an extra update once the parent window geometry expands on screen
        super().showEvent(event)
        self.update_preview()

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

    def export_image(self):
        if self.img_color is None or self.img_depth is None: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Composite Image", "", "PNG Image (*.png);;JPEG Image (*.jpg)")
        if file_path:
            final_composite = self.process_depth_blur(self.img_color, self.img_depth)
            cv2.imwrite(file_path, final_composite)
            QMessageBox.information(self, "Success", f"Composite exported cleanly to:\n{file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Check system arguments array for input image file paths
    color_arg = sys.argv[1] if len(sys.argv) > 1 else None
    depth_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    window = ProDepthBlurQtApp(color_arg, depth_arg)
    window.show()
    sys.exit(app.exec())
