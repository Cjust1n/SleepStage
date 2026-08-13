# widgets/ai_panel.py
"""
Combined AI dashboard panel with prediction, confidence, buffer, and model info.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter
from PySide6.QtCore import Qt

from widgets.prediction_card import PredictionCard
from widgets.confidence_chart import ConfidenceChart
from widgets.buffer_progress import BufferProgress
from widgets.inference_panel import InferencePanel
from models.prediction_model import PredictionModel


class AIPanel(QWidget):
    """
    Combined AI inference dashboard.
    
    Layout:
    ┌─────────────────────┬──────────────────┐
    │  Prediction Card    │  Inference Info  │
    ├─────────────────────┤                  │
    │  Confidence Chart   │                  │
    ├─────────────────────┴──────────────────┤
    │  Buffer Progress                       │
    └────────────────────────────────────────┘
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Top row: Prediction + Confidence | Model Info
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Left: Prediction + Confidence
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        
        self.prediction_card = PredictionCard()
        self.confidence_chart = ConfidenceChart()
        
        left_layout.addWidget(self.prediction_card, 2)
        left_layout.addWidget(self.confidence_chart, 3)
        
        # Right: Model info
        self.inference_panel = InferencePanel()
        
        top_splitter.addWidget(left_widget)
        top_splitter.addWidget(self.inference_panel)
        top_splitter.setSizes([400, 250])
        
        layout.addWidget(top_splitter, 3)
        
        # Bottom: Buffer progress
        self.buffer_progress = BufferProgress()
        layout.addWidget(self.buffer_progress, 1)
    
    def update_prediction(self, model: PredictionModel) -> None:
        """
        Update all AI widgets with new prediction.
        
        Args:
            model: PredictionModel instance
        """
        self.prediction_card.update_prediction(
            model.stage_name,
            model.confidence,
            model.get_color()
        )
        self.confidence_chart.update_scores(model.scores)
        self.buffer_progress.update_progress(model.buffer_filled, model.buffer_total)
        
        if model.inference_time_ms > 0:
            self.inference_panel.update_inference_time(model.inference_time_ms)
    
    def update_buffer(self, filled: int, total: int = 30) -> None:
        """Update buffer progress only."""
        self.buffer_progress.update_progress(filled, total)
    
    def clear(self) -> None:
        """Clear all AI widgets."""
        self.prediction_card.clear()
        self.confidence_chart.clear()
        self.buffer_progress.clear()
        self.inference_panel.clear()