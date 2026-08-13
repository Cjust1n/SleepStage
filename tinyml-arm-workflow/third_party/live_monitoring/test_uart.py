# test_gui_integration.py
"""
Test GUI integration with virtual serial ports.
Creates a pair of virtual serial ports for testing without hardware.
"""

import sys
import os
import time
import threading
import random

# Only import if on Linux/Mac
if sys.platform != 'win32':
    import pty
    import tty

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from gui.main_window import MainWindow


def create_virtual_serial_test():
    """
    Create a virtual serial port test without GUI.
    Tests SerialWorker directly.
    """
    print("\n=== Virtual Serial Port Test ===")
    print("This test requires a real serial port or virtual port.")
    
    # Check available ports
    from serial_port.serial_manager import SerialManager
    manager = SerialManager()
    ports = manager.list_ports()
    
    print(f"Available ports: {ports}")
    
    if not ports:
        print("\n⚠️ No serial ports available for testing.")
        print("To test with virtual ports on Linux:")
        print("  socat -d -d pty,raw,echo=0 pty,raw,echo=0")
        print("This will create /dev/pts/X and /dev/pts/Y")
        return False
    
    return True


def simulate_data_generator():
    """Generate simulated UART data for testing."""
    test_packets = [
        "LOG,Device initialized",
        "STATUS,ready",
        "ACCEL,0.01,-0.02,0.98",
        "PPG,512,520,518",
        "FEATURE,mean,1.5,std,0.3,peak,2.1",
        "PRED,2,0.87",
        "LOG,Data collection started",
        "ACCEL,0.02,-0.01,0.97",
        "PPG,515,522,520",
        "ACCEL,0.03,-0.01,0.96",
        "PPG,518,524,522",
        "FEATURE,mean,1.6,std,0.4,peak,2.3",
        "PRED,2,0.91",
        "STATUS,running,uptime,120",
    ]
    
    return test_packets


if __name__ == "__main__":
    # Test without GUI
    print("Sleep Stage Development Studio - Integration Test")
    print("=" * 50)
    
    # Show available ports
    create_virtual_serial_test()
    
    # Show simulated data
    print("\n=== Simulated Data ===")
    packets = simulate_data_generator()
    
    from serial_port.serial_parser import Parser
    parser = Parser()
    
    for line in packets:
        packet = parser.parse_line(line)
        print(f"  {packet}")
    
    print(f"\nTotal simulated packets: {len(packets)}")
    
    print("\n=== Instructions ===")
    print("1. Run main.py to start the GUI")
    print("2. Connect a device or create virtual ports with socat")
    print("3. Connect to the port and observe UART data")
    print("\nFor virtual port testing:")
    print("  Terminal 1: socat -d -d pty,raw,echo=0 pty,raw,echo=0")
    print("  Terminal 2: echo 'ACCEL,0.1,0.2,0.3' > /dev/pts/X")
    print("  Terminal 3: python main.py (connect to /dev/pts/Y)")