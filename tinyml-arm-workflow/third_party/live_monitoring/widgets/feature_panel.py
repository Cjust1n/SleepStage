# widgets/feature_panel.py
"""
Compact feature panel with cards and mini trend graphs.
"""

from typing import Optional, Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QScrollArea,
    QLabel, QFrame, QHBoxLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from models.feature_model import FeatureModel
from models.feature_history import FeatureHistory
from widgets.feature_card import FeatureCard
from widgets.feature_history_plot import FeatureHistoryPlot


class FeaturePanel(QWidget):
    """
    Compact panel displaying feature cards and trend graphs.
    
    Features that get trend graphs: rolling_mean_hr(3), energy(6), rmssd(4),
    zero_crossing(13), rolling_mean_acc(9)
    """
    
    # Features with trend graphs: (index, title, color)
    TREND_FEATURES = {
        3: ("HR Trend", (231, 76, 60)),
        6: ("Energy Trend", (241, 196, 15)),
        4: ("RMSSD Trend", (46, 204, 113)),
        13: ("Zero Cross", (142, 68, 173)),
        9: ("Mean Acc", (230, 126, 34)),
    }
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self.feature_cards: Dict[int, FeatureCard] = {}
        self.trend_graphs: Dict[int, FeatureHistoryPlot] = {}
        self.feature_history = FeatureHistory(max_epochs=100)
        self.current_features: Optional[FeatureModel] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Create compact layout with cards and trends."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # Title bar
        title_bar = QHBoxLayout()
        title_label = QLabel("Features")
        title_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        title_bar.addWidget(title_label)
        title_bar.addStretch()
        
        # Epoch counter
        self.epoch_label = QLabel("Epoch: ---")
        self.epoch_label.setFont(QFont("Segoe UI", 9))
        self.epoch_label.setStyleSheet("color: #7f8c8d;")
        title_bar.addWidget(self.epoch_label)
        
        main_layout.addLayout(title_bar)
        
        # Feature cards in 6 columns x 3 rows (compact)
        cards_widget = QWidget()
        cards_layout = QGridLayout(cards_widget)
        cards_layout.setSpacing(4)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        
        for i in range(18):
            model = FeatureModel()
            name = model.get_display_name(i)
            unit = model.get_unit(i)
            
            card = FeatureCard(name, unit)
            self.feature_cards[i] = card
            cards_layout.addWidget(card, i // 6, i % 6)
        
        main_layout.addWidget(cards_widget)
        
        # Trend graphs (horizontal row)
        trends_label = QLabel("Trends")
        trends_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        trends_label.setStyleSheet("color: #7f8c8d;")
        main_layout.addWidget(trends_label)
        
        trends_widget = QWidget()
        trends_layout = QHBoxLayout(trends_widget)
        trends_layout.setSpacing(4)
        trends_layout.setContentsMargins(0, 0, 0, 0)
        
        for feature_idx, (title, color) in self.TREND_FEATURES.items():
            graph = FeatureHistoryPlot(title, color)
            graph.setMinimumWidth(120)
            graph.setMaximumHeight(60)
            self.trend_graphs[feature_idx] = graph
            trends_layout.addWidget(graph)
        
        main_layout.addWidget(trends_widget)
    
    def update_features(self, features: FeatureModel) -> None:
        """Update all feature cards with new epoch data."""
        self.current_features = features
        self.feature_history.append(features)
        
        # Update epoch label
        self.epoch_label.setText(f"Epoch: {features.epoch_index}")
        
        # Update cards
        for i in range(18):
            formatted_value = features.get_formatted_value(i)
            self.feature_cards[i].set_value(formatted_value)
        
        # Update trends
        epochs = self.feature_history.get_epochs()
        for feature_idx, graph in self.trend_graphs.items():
            history = self.feature_history.get_history(feature_idx)
            graph.update_data(epochs, history)
    
    def clear(self) -> None:
        """Clear all data."""
        self.current_features = None
        self.feature_history.clear()
        self.epoch_label.setText("Epoch: ---")
        
        for card in self.feature_cards.values():
            card.clear()
        for graph in self.trend_graphs.values():
            graph.clear()