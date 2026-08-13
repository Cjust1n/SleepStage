# README.md
# Sleep Stage Development Studio

A professional desktop application for sleep stage monitoring and development, built with PySide6 and Python.

## Overview

Sleep Stage Development Studio provides a modular, extensible foundation for real-time sleep stage analysis. The application features a clean, professional interface with serial communication capabilities, ready for future expansion with signal processing, machine learning inference, and data visualization.

## Features (Current Milestone)

- 🎨 Professional light theme interface
- 🔌 Serial port management (COM port enumeration, connection control)
- 📊 Resizable dashboard/console layout
- 💻 Real-time UART console with timestamps
- 🎯 Modular architecture ready for extension
- 📱 Responsive design with QSplitter panels

## Architecture
sleep_stage_monitor/
├── main.py # Application entry point
├── gui/
│ ├── init.py
│ ├── main_window.py # Main application window
│ ├── dashboard.py # Visualization panel (placeholder)
│ └── toolbar.py # Connection controls toolbar
├── serial/
│ ├── init.py
│ └── serial_manager.py # Serial port management
├── widgets/
│ ├── init.py
│ └── console_widget.py # UART console display
├── resources/
│ └── icons/ # Application icons (future)
├── requirements.txt
└── README.md

text

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd sleep_stage_monitor
Create a virtual environment (recommended):

bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
Install dependencies:

bash
pip install -r requirements.txt
Usage
Run the application:

bash
python main.py
Interface Guide
Toolbar: Select COM port and baud rate, then click "Connect"

Dashboard: Left panel (future visualization area)

Console: Right panel showing UART communication log

Status Bar: Shows connection state and selected port