# test_visualization.py
"""
Test script for visualization components.
Simulates data to verify plots work correctly without hardware.
"""

import sys
import time
import math
import random
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from gui.main_window import MainWindow


def test_plots_standalone():
    """Test plots with simulated data (no serial connection needed)."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    # Simulated data generator
    t = 0.0
    sample_count = 0
    
    def update_plots():
        nonlocal t, sample_count
        
        # Simulate accelerometer data
        accel_x = 0.1 * math.sin(t * 2.0) + random.gauss(0, 0.02)
        accel_y = 0.1 * math.cos(t * 1.5) + random.gauss(0, 0.02)
        accel_z = 0.98 + 0.05 * math.sin(t * 0.5) + random.gauss(0, 0.01)
        
        # Simulate PPG data
        ppg = 512 + 20 * math.sin(t * 6.28) + random.gauss(0, 3)
        
        # Update plots
        window.dashboard.accel_plot.update(accel_x, accel_y, accel_z)
        window.dashboard.ppg_plot.update(ppg)
        
        # Update counters (simulate)
        sample_count += 1
        if sample_count % 100 == 0:
            window.status_indicator.setText("Simulating")
            window.dashboard.console.append(f"Simulated {sample_count} samples")
        
        t += 0.02  # 50 Hz
    
    # Create timer for simulated updates
    timer = QTimer()
    timer.timeout.connect(update_plots)
    timer.start(20)  # 50 Hz update rate
    
    # Update connection status
    window.status_indicator.setText("Simulating")
    window.status_indicator.setStyleSheet(
        "background-color: #f39c12; color: white; padding: 2px 8px; "
        "border-radius: 3px; font-weight: bold;"
    )
    window.port_label.setText("Simulated Data")
    window.dashboard.console.append("Running in simulation mode")
    window.dashboard.console.append("Generating test accelerometer and PPG data...")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    print("Sleep Stage Development Studio - Visualization Test")
    print("=" * 50)
    print("Starting in simulation mode...")
    print("Accelerometer: 3-axis (50 Hz)")
    print("PPG: Single channel (25 Hz)")
    print("\nClose the window to exit.")
    
    test_plots_standalone()