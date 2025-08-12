#!/usr/bin/env python3
"""
Simple GUI demo for PixTrick
Demonstrates Rust engine integration with PySide6
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QSlider, QComboBox,
    QFileDialog, QMessageBox, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage

try:
    import pixtrick_engine as engine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

class FilterWidget(QFrame):
    """Widget for a single filter with controls"""
    
    def __init__(self, filter_type, parent=None):
        super().__init__(parent)
        self.filter_type = filter_type
        self.parent_window = parent
        self.enabled = True
        self.parameters = {}
        
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("QFrame { border: 1px solid gray; margin: 2px; padding: 5px; }")
        
        layout = QVBoxLayout()
        
        # Header with filter name and enable/disable
        header_layout = QHBoxLayout()
        self.title_label = QLabel(filter_type.title())
        self.title_label.setStyleSheet("font-weight: bold;")
        
        self.enable_button = QPushButton("✓")
        self.enable_button.setMaximumWidth(30)
        self.enable_button.setStyleSheet("background-color: lightgreen;")
        self.enable_button.clicked.connect(self.toggle_enabled)
        
        self.remove_button = QPushButton("✗")
        self.remove_button.setMaximumWidth(30)
        self.remove_button.setStyleSheet("background-color: lightcoral;")
        self.remove_button.clicked.connect(self.remove_filter)
        
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.enable_button)
        header_layout.addWidget(self.remove_button)
        
        layout.addLayout(header_layout)
        
        # Add parameter controls based on filter type
        self.add_parameter_controls(layout)
        
        self.setLayout(layout)
    
    def add_parameter_controls(self, layout):
        """Add parameter controls based on filter type"""
        if self.filter_type == "brightness":
            self.add_slider("amount", "Brightness", -100, 100, 0, layout)
        elif self.filter_type == "contrast":
            self.add_slider("amount", "Contrast", -100, 100, 0, layout)
        elif self.filter_type == "saturation":
            self.add_slider("amount", "Saturation", 0, 300, 100, layout)
        elif self.filter_type == "hue_shift":
            self.add_slider("degrees", "Hue Shift", -180, 180, 0, layout)
        elif self.filter_type == "box_blur":
            self.add_slider("radius", "Radius", 0, 20, 1, layout)
        elif self.filter_type == "gaussian_blur":
            self.add_slider("radius", "Radius", 0.0, 10.0, 1.0, layout, is_float=True)
        # invert and grayscale have no parameters
    
    def add_slider(self, param_name, label, min_val, max_val, default_val, layout, is_float=False):
        """Add a slider control for a parameter"""
        param_layout = QHBoxLayout()
        
        param_label = QLabel(f"{label}:")
        param_label.setMinimumWidth(80)
        
        if is_float:
            # For float values, scale by 10 for slider precision
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * 10))
            slider.setMaximum(int(max_val * 10))
            slider.setValue(int(default_val * 10))
            
            value_label = QLabel(f"{default_val:.1f}")
            
            def update_float_value():
                val = slider.value() / 10.0
                value_label.setText(f"{val:.1f}")
                self.parameters[param_name] = val
                if self.parent_window:
                    self.parent_window.process_image()
        else:
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default_val)
            
            value_label = QLabel(str(default_val))
            
            def update_int_value():
                val = slider.value()
                value_label.setText(str(val))
                self.parameters[param_name] = float(val)
                if self.parent_window:
                    self.parent_window.process_image()
        
        if is_float:
            slider.valueChanged.connect(update_float_value)
        else:
            slider.valueChanged.connect(update_int_value)
        
        # Initialize parameter
        self.parameters[param_name] = float(default_val)
        
        param_layout.addWidget(param_label)
        param_layout.addWidget(slider)
        param_layout.addWidget(value_label)
        
        layout.addLayout(param_layout)
    
    def toggle_enabled(self):
        """Toggle filter enabled/disabled"""
        self.enabled = not self.enabled
        if self.enabled:
            self.enable_button.setText("✓")
            self.enable_button.setStyleSheet("background-color: lightgreen;")
            self.setStyleSheet("QFrame { border: 1px solid gray; margin: 2px; padding: 5px; }")
        else:
            self.enable_button.setText("✗")
            self.enable_button.setStyleSheet("background-color: lightgray;")
            self.setStyleSheet("QFrame { border: 1px solid lightgray; margin: 2px; padding: 5px; color: gray; }")
        
        if self.parent_window:
            self.parent_window.process_image()
    
    def remove_filter(self):
        """Remove this filter"""
        if self.parent_window:
            self.parent_window.remove_filter(self)

class PixTrickMainWindow(QMainWindow):
    """Main window for PixTrick demo"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixTrick Demo - Rust Engine + PySide6")
        self.setGeometry(100, 100, 1200, 800)
        
        # Image processing
        self.source_image = None
        self.current_image = None
        self.filter_widgets = []
        
        if ENGINE_AVAILABLE:
            self.graph = engine.NodeGraph()
        
        self.setup_ui()
        
        # Auto-processing timer to avoid too frequent updates
        self.process_timer = QTimer()
        self.process_timer.setSingleShot(True)
        self.process_timer.timeout.connect(self.do_process_image)
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout()
        
        # File operations
        file_group = QFrame()
        file_group.setFrameStyle(QFrame.Box)
        file_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        
        file_layout.addWidget(QLabel("File Operations"))
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.save_button)
        file_group.setLayout(file_layout)
        
        # Add filters
        filter_group = QFrame()
        filter_group.setFrameStyle(QFrame.Box)
        filter_layout = QVBoxLayout()
        
        filter_layout.addWidget(QLabel("Add Filters"))
        
        self.filter_combo = QComboBox()
        if ENGINE_AVAILABLE:
            self.filter_combo.addItems(engine.NodeGraph.get_available_filters())
        else:
            self.filter_combo.addItems(["Engine not available"])
        
        self.add_filter_button = QPushButton("Add Filter")
        self.add_filter_button.clicked.connect(self.add_filter)
        self.add_filter_button.setEnabled(ENGINE_AVAILABLE)
        
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addWidget(self.add_filter_button)
        filter_group.setLayout(filter_layout)
        
        # Filter list (scrollable)
        self.filter_scroll = QScrollArea()
        self.filter_container = QWidget()
        self.filter_list_layout = QVBoxLayout()
        self.filter_list_layout.addStretch()
        self.filter_container.setLayout(self.filter_list_layout)
        self.filter_scroll.setWidget(self.filter_container)
        self.filter_scroll.setWidgetResizable(True)
        
        # Status
        self.status_label = QLabel("Ready")
        if not ENGINE_AVAILABLE:
            self.status_label.setText("⚠️ Rust engine not available. Run: cd src/engine && maturin develop")
            self.status_label.setStyleSheet("color: red;")
        
        left_layout.addWidget(file_group)
        left_layout.addWidget(filter_group)
        left_layout.addWidget(QLabel("Active Filters"))
        left_layout.addWidget(self.filter_scroll)
        left_layout.addStretch()
        left_layout.addWidget(self.status_label)
        
        left_panel.setLayout(left_layout)
        
        # Right panel - Image display
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Image display
        self.image_label = QLabel("Load an image to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; min-height: 400px;")
        self.image_label.setScaledContents(False)
        
        # Image scroll area
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(True)
        
        # Image info
        self.image_info_label = QLabel("No image loaded")
        
        right_layout.addWidget(QLabel("Image Preview"))
        right_layout.addWidget(self.image_scroll)
        right_layout.addWidget(self.image_info_label)
        
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)
        
        central_widget.setLayout(main_layout)
    
    def load_image(self):
        """Load an image file"""
        if not ENGINE_AVAILABLE:
            QMessageBox.warning(self, "Error", "Rust engine not available!")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Image", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            try:
                self.source_image = engine.load_image(file_path)
                self.graph.set_source_image(self.source_image)
                
                self.display_image(self.source_image)
                self.update_image_info()
                self.save_button.setEnabled(True)
                self.status_label.setText(f"Loaded: {Path(file_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
    
    def save_image(self):
        """Save the current processed image"""
        if not self.current_image:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_image.save(file_path)
                self.status_label.setText(f"Saved: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
    
    def add_filter(self):
        """Add a new filter to the processing chain"""
        filter_type = self.filter_combo.currentText()
        
        # Create filter widget
        filter_widget = FilterWidget(filter_type, self)
        
        # Add to layout (insert before stretch)
        self.filter_list_layout.insertWidget(
            self.filter_list_layout.count() - 1, 
            filter_widget
        )
        
        self.filter_widgets.append(filter_widget)
        
        # Process image with new filter
        if self.source_image:
            self.process_image()
    
    def remove_filter(self, filter_widget):
        """Remove a filter from the processing chain"""
        if filter_widget in self.filter_widgets:
            self.filter_widgets.remove(filter_widget)
            filter_widget.setParent(None)
            filter_widget.deleteLater()
            
            # Reprocess image
            if self.source_image:
                self.process_image()
    
    def process_image(self):
        """Process image with current filter chain (debounced)"""
        self.process_timer.start(100)  # 100ms delay
    
    def do_process_image(self):
        """Actually process the image"""
        if not self.source_image or not ENGINE_AVAILABLE:
            return
        
        try:
            # Clear existing nodes
            for node_id in self.graph.get_node_ids():
                self.graph.remove_node(node_id)
            
            # Add current filters
            for i, filter_widget in enumerate(self.filter_widgets):
                if filter_widget.enabled:
                    node_id = f"filter_{i}_{filter_widget.filter_type}"
                    node = engine.FilterNode(node_id, filter_widget.filter_type)
                    
                    # Set parameters
                    for param_name, param_value in filter_widget.parameters.items():
                        node.set_parameter(param_name, param_value)
                    
                    self.graph.add_node(node)
            
            # Process and display
            start_time = __import__('time').time()
            self.current_image = self.graph.process()
            process_time = (__import__('time').time() - start_time) * 1000
            
            self.display_image(self.current_image)
            self.update_image_info()
            
            active_filters = len([fw for fw in self.filter_widgets if fw.enabled])
            self.status_label.setText(f"Processed {active_filters} filters in {process_time:.1f}ms")
            
        except Exception as e:
            self.status_label.setText(f"Processing error: {e}")
    
    def display_image(self, pix_image):
        """Display a PixImage in the GUI"""
        try:
            # Convert PixImage to QPixmap for display
            # This is a simplified conversion - in a real app you'd want more robust handling
            
            # Create a test pattern for display since we don't have direct pixel access
            # In a real implementation, you'd extract pixel data from the PixImage
            width, height = pix_image.width, pix_image.height
            
            # For now, just show image dimensions
            self.image_label.setText(f"Image: {width}x{height}\n(Pixel display not implemented in demo)")
            self.image_label.setStyleSheet("border: 2px solid green; min-height: 400px; background-color: #f0f0f0;")
            
        except Exception as e:
            self.image_label.setText(f"Display error: {e}")
    
    def update_image_info(self):
        """Update image information display"""
        if self.current_image:
            info = (f"Dimensions: {self.current_image.width}x{self.current_image.height} "
                   f"| Channels: {self.current_image.channels} "
                   f"| Filters: {len([fw for fw in self.filter_widgets if fw.enabled])}")
            self.image_info_label.setText(info)

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')  # Modern cross-platform style
    
    window = PixTrickMainWindow()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())