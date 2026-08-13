# main.py (UPDATED - Milestone 3)
"""
Sleep Stage Development Studio - Main Entry Point
A desktop application for sleep stage monitoring and development.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from gui.main_window import MainWindow


def setup_light_theme(app: QApplication) -> None:
    """Configure a professional light theme for the application."""
    app.setStyle("Fusion")
    
    palette = QPalette()
    
    # Base colors
    palette.setColor(QPalette.Window, QColor(240, 240, 245))
    palette.setColor(QPalette.WindowText, QColor(30, 30, 30))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 250))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(30, 30, 30))
    palette.setColor(QPalette.Text, QColor(30, 30, 30))
    
    # Button colors
    palette.setColor(QPalette.Button, QColor(240, 240, 245))
    palette.setColor(QPalette.ButtonText, QColor(30, 30, 30))
    palette.setColor(QPalette.BrightText, Qt.red)
    
    # Highlight colors
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    # Disabled colors
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(128, 128, 128))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(255, 255, 255))
    
    app.setPalette(palette)
    
    # Global stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f5;
        }
        QToolBar {
            background-color: #ffffff;
            border-bottom: 1px solid #d4d4d4;
            spacing: 4px;
            padding: 4px;
        }
        QToolBar QToolButton {
            background-color: #f8f8f8;
            border: 1px solid #d4d4d4;
            border-radius: 4px;
            padding: 4px 8px;
            margin: 2px;
        }
        QToolBar QToolButton:hover {
            background-color: #e8e8e8;
            border-color: #b4b4b4;
        }
        QToolBar QToolButton:pressed {
            background-color: #d0d0d0;
        }
        QToolBar QToolButton:disabled {
            background-color: #f0f0f0;
            color: #a0a0a0;
        }
        QComboBox {
            background-color: #ffffff;
            border: 1px solid #d4d4d4;
            border-radius: 4px;
            padding: 4px 8px;
            min-width: 120px;
        }
        QComboBox:hover {
            border-color: #0078d7;
        }
        QComboBox:focus {
            border-color: #0078d7;
            border-width: 2px;
        }
        QStatusBar {
            background-color: #ffffff;
            border-top: 1px solid #d4d4d4;
            color: #505050;
            padding: 4px;
        }
        QSplitter::handle {
            background-color: #e0e0e0;
            width: 3px;
        }
        QSplitter::handle:hover {
            background-color: #0078d7;
        }
        QSplitter::handle:vertical {
            height: 3px;
        }
    """)


def main() -> None:
    """Initialize and run the Sleep Stage Development Studio application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Sleep Stage Development Studio")
    app.setOrganizationName("SleepStageDev")
    
    # Apply light theme
    setup_light_theme(app)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()