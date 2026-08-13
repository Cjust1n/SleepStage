# widgets/inference_panel.py
"""
Inference information panel showing model details and performance metrics.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class InfoRow(QWidget):
    """Single row of label-value pair."""
    
    def __init__(self, label: str, value: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        
        self.label = QLabel(label)
        self.label.setFont(QFont("Segoe UI", 8))
        self.label.setStyleSheet("color: #7f8c8d; border: none;")
        
        self.value = QLabel(value)
        self.value.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.value.setStyleSheet("color: #2c3e50; border: none;")
        
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(self.value, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
    
    def set_value(self, value: str) -> None:
        self.value.setText(value)


class InferencePanel(QWidget):
    """
    Panel displaying model information and inference metrics.
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        
        # Title
        title = QLabel("Model Info")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; border: none;")
        layout.addWidget(title)
        
        # Info rows
        self.board_row = InfoRow("Board", "Grove Vision AI V2")
        self.mcu_row = InfoRow("MCU", "Cortex-M55 + Ethos-U55")
        self.runtime_row = InfoRow("Runtime", "TensorFlow Lite Micro")
        self.model_row = InfoRow("Model", "gru_int8_vela.tflite")
        self.arch_row = InfoRow("Architecture", "GRU (30-step)")
        self.quant_row = InfoRow("Quantization", "INT8")
        self.input_row = InfoRow("Input Shape", "[1 × 30 × 16]")
        self.output_row = InfoRow("Output Shape", "[1 × 4]")
        self.infer_row = InfoRow("Inference Time", "---")
        
        for row in [self.board_row, self.mcu_row, self.runtime_row, 
                     self.model_row, self.arch_row, self.quant_row,
                     self.input_row, self.output_row, self.infer_row]:
            layout.addWidget(row)
        
        layout.addStretch()
        
        self.setStyleSheet("""
            InferencePanel {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
    
    def update_inference_time(self, time_ms: float) -> None:
        """Update inference time display."""
        self.infer_row.set_value(f"{time_ms:.1f} ms")
    
    def clear(self) -> None:
        """Reset inference time."""
        self.infer_row.set_value("---")