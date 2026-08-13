# gui/dashboard.py (ADD Peak Detector section)
"""
Dashboard with Peak Detector Analyzer tab.
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QHBoxLayout, QTabWidget
)
from PySide6.QtCore import Qt

from widgets.accel_plot import AccelPlot
from widgets.ppg_plot import PPGPlot
from widgets.feature_panel import FeaturePanel
from widgets.prediction_card import PredictionCard
from widgets.confidence_chart import ConfidenceChart
from widgets.inference_panel import InferencePanel
from widgets.console_widget import ConsoleWidget
from widgets.ppg_peak_plot import PPGPeakPlot
from widgets.rr_interval_plot import RRIntervalPlot
from widgets.heart_rate_plot import HeartRatePlot
from widgets.peak_statistics import PeakStatisticsPanel
from models.feature_model import FeatureModel
from models.prediction_model import PredictionModel
from models.peak_model import PeakModel


class DashboardPanel(QWidget):
    """
    Main dashboard with tabbed interface.
    
    Tab 1: Main Dashboard (plots + AI + features)
    Tab 2: Peak Detector Analyzer
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("DashboardPanel")
        
        # Peak model for analyzer
        self.peak_model = PeakModel()
        self._sample_time = 0.0
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #d4d4d4;
                background-color: #f0f0f5;
            }
            QTabBar::tab {
                padding: 6px 16px;
                font-size: 10px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom: 2px solid #3498db;
            }
        """)
        
        # === Tab 1: Main Dashboard ===
        self.main_tab = self._create_main_tab()
        self.tab_widget.addTab(self.main_tab, "📊 Main Dashboard")
        
        # === Tab 2: Peak Detector ===
        self.peak_tab = self._create_peak_tab()
        self.tab_widget.addTab(self.peak_tab, "🔍 Peak Detector")
        
        layout.addWidget(self.tab_widget)
    
    def _create_main_tab(self) -> QWidget:
        """Create main dashboard tab (existing layout)."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Left side (plots + AI + features)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        
        # Plots
        plots_splitter = QSplitter(Qt.Horizontal)
        self.accel_plot = AccelPlot(history_duration=10.0, sample_rate=50.0)
        self.ppg_plot = PPGPlot(history_duration=10.0, sample_rate=25.0)
        plots_splitter.addWidget(self.accel_plot)
        plots_splitter.addWidget(self.ppg_plot)
        plots_splitter.setSizes([400, 400])
        
        # AI row
        ai_row = QSplitter(Qt.Horizontal)
        self.prediction_card = PredictionCard()
        self.confidence_chart = ConfidenceChart()
        ai_row.addWidget(self.prediction_card)
        ai_row.addWidget(self.confidence_chart)
        ai_row.setSizes([220, 350])
        
        # Features
        self.feature_panel = FeaturePanel()
        
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(plots_splitter)
        left_splitter.addWidget(ai_row)
        left_splitter.addWidget(self.feature_panel)
        left_splitter.setSizes([280, 160, 200])
        
        left_layout.addWidget(left_splitter)
        
        # Right side (model info + console)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        
        self.inference_panel = InferencePanel()
        self.inference_panel.setMaximumHeight(200)
        
        self.console = ConsoleWidget()
        self.console.setMinimumWidth(280)
        
        right_layout.addWidget(self.inference_panel)
        right_layout.addWidget(self.console)
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left)
        main_splitter.addWidget(right)
        main_splitter.setSizes([950, 320])
        
        layout.addWidget(main_splitter)
        return widget
    
    def _create_peak_tab(self) -> QWidget:
        """Create peak detector analyzer tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # 1. PPG + Peaks + Threshold plot (top)
        self.ppg_peak_plot = PPGPeakPlot(history_duration=10.0, sample_rate=50.0)
        
        # 2. RR Interval + Heart Rate (middle)
        mid_splitter = QSplitter(Qt.Horizontal)
        self.rr_plot = RRIntervalPlot()
        self.hr_plot = HeartRatePlot()
        mid_splitter.addWidget(self.rr_plot)
        mid_splitter.addWidget(self.hr_plot)
        mid_splitter.setSizes([400, 400])
        
        # 3. Peak Statistics (bottom)
        self.peak_stats = PeakStatisticsPanel()
        
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(self.ppg_peak_plot)
        main_splitter.addWidget(mid_splitter)
        main_splitter.addWidget(self.peak_stats)
        main_splitter.setSizes([300, 250, 100])
        main_splitter.setChildrenCollapsible(False)
        
        layout.addWidget(main_splitter)
        return widget
    
    # ===== Peak Detector Update Methods =====
    def add_ppg_sample(self, value: float) -> None:
        """Add PPG sample to peak model."""
        self._sample_time += 1.0 / 50.0  # 50 Hz
        self.peak_model.add_ppg(value, self._sample_time)
        
        # Update plot
        times, values = self.peak_model.get_ppg_arrays()
        self.ppg_peak_plot.update_ppg(times, values)
    
    def add_peak(self, value: float, accepted: bool = True) -> None:
        """Add detected peak."""
        self.peak_model.add_peak(value, self._sample_time, accepted)
        
        peak_times, peak_values = self.peak_model.get_peak_arrays()
        self.ppg_peak_plot.update_peaks(peak_times, peak_values)
        
        # Update statistics
        self.peak_stats.update_statistics(self.peak_model.statistics)
    
    def add_threshold(self, value: float) -> None:
        """Update threshold."""
        self.peak_model.add_threshold(value)
        
        times, _ = self.peak_model.get_ppg_arrays()
        thresh_values = np.full(len(times), value) if len(times) > 0 else np.array([])
        self.ppg_peak_plot.update_threshold(times, thresh_values)
    
    def add_rr_interval(self, rr_ms: float) -> None:
        """Add RR interval."""
        self.peak_model.add_rr(rr_ms, self._sample_time)
        
        indices, values = self.peak_model.get_rr_arrays()
        self.rr_plot.update_data(indices, values)
        
        self.peak_stats.update_statistics(self.peak_model.statistics)
    
    def add_hr(self, hr: float) -> None:
        """Add heart rate."""
        self.peak_model.add_hr(hr, self._sample_time)
        
        times, values = self.peak_model.get_hr_arrays()
        self.hr_plot.update_data(times, values)
    
    # ===== Existing methods =====
    def update_prediction(self, model: PredictionModel) -> None:
        self.prediction_card.update_prediction(model.stage_name, model.confidence, model.get_color())
        self.confidence_chart.update_scores(model.scores)
        self.confidence_chart.update_buffer(model.buffer_filled, model.buffer_total)
        if model.inference_time_ms > 0:
            self.inference_panel.update_inference_time(model.inference_time_ms)
    
    def update_buffer(self, filled: int, total: int = 30) -> None:
        self.confidence_chart.update_buffer(filled, total)
    
    def update_features(self, features: FeatureModel) -> None:
        self.feature_panel.update_features(features)
    
    def clear_all(self) -> None:
        self.accel_plot.clear()
        self.ppg_plot.clear()
        self.prediction_card.clear()
        self.confidence_chart.clear()
        self.inference_panel.clear()
        self.feature_panel.clear()
        self.ppg_peak_plot.clear()
        self.rr_plot.clear()
        self.hr_plot.clear()
        self.peak_stats.clear()
        self.peak_model.clear()
        self._sample_time = 0.0