# serial/serial_worker.py
"""
Serial worker thread for asynchronous UART communication.
Provides continuous reading from serial port without blocking the GUI.
"""

from typing import Optional
import time

from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

import serial
from serial_port.serial_parser import Parser, Packet


class SerialWorker(QThread):
    """
    Background worker thread for continuous serial port reading.
    
    Runs in a separate QThread to prevent GUI freezing during
    continuous UART data reception. Emits signals for received
    data, parsed packets, errors, and connection status.
    
    Signals:
        data_received: Emitted with raw line string when data is received
        packet_received: Emitted with parsed Packet object
        error_occurred: Emitted with error message string
        connection_status: Emitted with bool (True=connected, False=disconnected)
    
    Attributes:
        port_name: Name of the serial port to connect to
        baud_rate: Communication speed in bits per second
        running: Flag to control the reading loop
        mutex: Mutex for thread-safe access to shared resources
    """
    
    # Signals
    data_received = Signal(str)           # Raw line received
    packet_received = Signal(object)      # Parsed Packet object
    error_occurred = Signal(str)          # Error messages
    connection_status = Signal(bool)      # Connection state changes
    
    def __init__(self, port_name: str, baud_rate: int = 921600) -> None:
        """
        Initialize the serial worker thread.
        
        Args:
            port_name: Serial port device name (e.g., '/dev/ttyACM0')
            baud_rate: Communication speed (default: 921600)
        """
        super().__init__()
        
        self.port_name: str = port_name
        self.baud_rate: int = baud_rate
        self._running: bool = False
        self._serial_port: Optional[serial.Serial] = None
        self._parser: Parser = Parser()
        self._mutex: QMutex = QMutex()
        self._buffer: str = ""
    
    def run(self) -> None:
        """
        Main thread execution method.
        
        Opens serial port and continuously reads data until stopped.
        Emits signals for received data and handles errors gracefully.
        """
        try:
            # Open serial port
            self._open_serial_port()
            
            # Emit connection status
            self.connection_status.emit(True)
            
            # Set running flag
            with QMutexLocker(self._mutex):
                self._running = True
            
            # Main reading loop
            while self._is_running():
                try:
                    self._read_serial_data()
                except serial.SerialException as e:
                    self.error_occurred.emit(f"Serial read error: {str(e)}")
                    break
                except Exception as e:
                    self.error_occurred.emit(f"Unexpected error: {str(e)}")
                    break
                
                # Small sleep to prevent CPU spinning
                self.msleep(10)
        
        except serial.SerialException as e:
            self.error_occurred.emit(f"Failed to open port: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Worker error: {str(e)}")
        finally:
            # Cleanup
            self._cleanup()
            self.connection_status.emit(False)
    
    def stop(self) -> None:
        """
        Stop the worker thread gracefully.
        
        Sets the running flag to False and waits for the thread to finish.
        Safe to call multiple times.
        """
        with QMutexLocker(self._mutex):
            self._running = False
        
        # Wait for thread to finish (with timeout)
        if self.isRunning():
            self.wait(1000)  # Wait up to 1 second
        
        # Force cleanup if still running
        if self.isRunning():
            self._cleanup()
            self.terminate()
            self.wait()
    
    def _is_running(self) -> bool:
        """
        Thread-safe check if worker should continue running.
        
        Returns:
            True if worker should continue, False otherwise
        """
        with QMutexLocker(self._mutex):
            return self._running
    
    def _open_serial_port(self) -> None:
        """
        Open the serial port with configured settings.
        
        Raises:
            serial.SerialException: If port cannot be opened
        """
        self._serial_port = serial.Serial(
            port=self.port_name,
            baudrate=self.baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,  # Non-blocking read with short timeout
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        
        if not self._serial_port.is_open:
            self._serial_port.open()
    
    def send_command(self, command: str) -> None:
        """
        Send a command string to the board via UART.
        
        Thread-safe: can be called from the GUI thread while the worker
        thread is running. Appends a newline so the firmware's line-based
        command parser (xgets) sees a complete line.
        
        Args:
            command: Command string WITHOUT trailing newline.
        """
        with QMutexLocker(self._mutex):
            if not self._serial_port or not self._serial_port.is_open:
                self.error_occurred.emit("Cannot send: serial port not open")
                return
            
            payload = command if command.endswith('\n') else command + '\n'
            try:
                self._serial_port.write(payload.encode('ascii', errors='replace'))
            except Exception as e:
                self.error_occurred.emit(f"Send error: {str(e)}")
    
    def _read_serial_data(self) -> None:
        """
        Read available data from serial port and process complete lines.
        
        Handles partial line buffering and emits signals for complete lines.
        """
        if not self._serial_port or not self._serial_port.is_open:
            return
        
        try:
            # Check if data is available
            if self._serial_port.in_waiting > 0:
                # Read available data
                data = self._serial_port.read(self._serial_port.in_waiting)
                
                # Decode bytes to string
                try:
                    text = data.decode('utf-8', errors='replace')
                except UnicodeDecodeError:
                    text = data.decode('ascii', errors='replace')
                
                # Add to buffer
                self._buffer += text
                
                # Process complete lines
                self._process_buffer()
                
        except serial.SerialException:
            # Port disconnected unexpectedly
            raise
        except Exception as e:
            self.error_occurred.emit(f"Read error: {str(e)}")
    
    def _process_buffer(self) -> None:
        """
        Process the internal buffer for complete lines.
        
        Splits on newline characters, emits complete lines as signals,
        and retains incomplete lines in the buffer.
        """
        while '\n' in self._buffer:
            # Split on first newline
            line, self._buffer = self._buffer.split('\n', 1)
            
            # Clean the line (remove carriage return and whitespace)
            line = line.strip()
            
            if line:
                # Emit raw data
                self.data_received.emit(line)
                
                # Parse and emit packet
                try:
                    packet = self._parser.parse_line(line)
                    self.packet_received.emit(packet)
                except Exception as e:
                    self.error_occurred.emit(f"Parse error: {str(e)}")
    
    def _cleanup(self) -> None:
        """
        Clean up serial port resources.
        
        Ensures serial port is properly closed regardless of state.
        """
        try:
            if self._serial_port and self._serial_port.is_open:
                self._serial_port.close()
        except Exception:
            pass
        finally:
            self._serial_port = None
    
    def __del__(self) -> None:
        """Destructor to ensure thread cleanup."""
        self.stop()