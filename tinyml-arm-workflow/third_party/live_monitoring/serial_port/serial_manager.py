# serial_port/serial_manager.py
"""
Serial port manager for UART communication.
Provides port enumeration, connection management, and basic I/O operations.
"""

from typing import List, Optional, Dict, Any
import serial.tools.list_ports
import serial


class SerialManager:
    """
    Manager for serial port operations.
    
    Handles port discovery, connection lifecycle, and configuration.
    Designed for future extension with threading and data parsing.
    
    Attributes:
        port: Current serial port instance (None if disconnected)
        connected: Connection state flag
        current_port: Name of currently connected port
        current_baud: Current baud rate setting
    """
    
    # Default serial configuration
    DEFAULT_SETTINGS: Dict[str, Any] = {
        'bytesize': serial.EIGHTBITS,
        'parity': serial.PARITY_NONE,
        'stopbits': serial.STOPBITS_ONE,
        'timeout': 1,
        'xonxoff': False,
        'rtscts': False,
        'dsrdtr': False
    }
    
    # Port filters - only detect these types of serial devices
    ALLOWED_PORT_PATTERNS = ['ttyACM', 'ttyUSB']
    
    def __init__(self) -> None:
        """Initialize the serial manager with default state."""
        self.port: Optional[serial.Serial] = None
        self.connected: bool = False
        self.current_port: Optional[str] = None
        self.current_baud: Optional[int] = None
    
    def list_ports(self) -> List[str]:
        """
        List available serial ports on the system.
        
        Only detects ttyACM* and ttyUSB* devices.
        Filters out system ports (ttyS*, ttyAMA*, etc.) and virtual consoles.
        
        Returns:
            List of port device names (e.g., ['/dev/ttyUSB0', '/dev/ttyACM0'])
        """
        ports = serial.tools.list_ports.comports()
        available_ports = []
        
        for port in ports:
            device_name = port.device.lower()
            
            # Only include ports matching allowed patterns
            if any(pattern.lower() in device_name for pattern in self.ALLOWED_PORT_PATTERNS):
                available_ports.append(port.device)
        
        return available_ports
    
    def get_port_details(self) -> List[Dict[str, str]]:
        """
        Get detailed information about available ports.
        
        Only returns details for ttyACM* and ttyUSB* devices.
        
        Returns:
            List of dictionaries with port information
            [{'device': '/dev/ttyACM0', 'description': 'STM32 Virtual COM Port', 'hwid': 'USB VID:PID=...'}]
        """
        ports = serial.tools.list_ports.comports()
        port_details = []
        
        for port in ports:
            device_name = port.device.lower()
            
            # Only include ports matching allowed patterns
            if any(pattern.lower() in device_name for pattern in self.ALLOWED_PORT_PATTERNS):
                port_info = {
                    'device': port.device,
                    'description': port.description or 'Unknown Device',
                    'hwid': port.hwid or 'N/A',
                    'manufacturer': getattr(port, 'manufacturer', None) or 'Unknown',
                    'serial_number': getattr(port, 'serial_number', None) or 'N/A'
                }
                port_details.append(port_info)
        
        return port_details
    
    def connect(self, port_name: str, baud_rate: int) -> bool:
        """
        Establish connection to specified serial port.
        
        Args:
            port_name: Name of the port to connect to (e.g., '/dev/ttyACM0')
            baud_rate: Communication speed in bits per second
        
        Returns:
            True if connection successful, False otherwise
        
        Note:
            Automatically disconnects existing connection before establishing new one.
        """
        # Disconnect if already connected
        if self.connected:
            self.disconnect()
        
        try:
            # Create serial port with configuration
            self.port = serial.Serial(
                port=port_name,
                baudrate=baud_rate,
                **self.DEFAULT_SETTINGS
            )
            
            # Verify port is open
            if self.port.is_open:
                self.connected = True
                self.current_port = port_name
                self.current_baud = baud_rate
                return True
            else:
                # Port didn't open properly
                self.port = None
                return False
                
        except serial.SerialException as e:
            # Connection failed
            print(f"Serial connection error: {e}")
            self.port = None
            self.connected = False
            self.current_port = None
            self.current_baud = None
            return False
        except Exception as e:
            # Unexpected error
            print(f"Unexpected error connecting to serial port: {e}")
            self.port = None
            self.connected = False
            self.current_port = None
            self.current_baud = None
            return False
    
    def disconnect(self) -> None:
        """
        Disconnect from the current serial port.
        
        Safely closes the serial connection and resets state.
        No exception raised if already disconnected.
        """
        try:
            if self.port and self.port.is_open:
                self.port.close()
        except Exception as e:
            print(f"Error during disconnect: {e}")
        finally:
            self.port = None
            self.connected = False
            self.current_port = None
            self.current_baud = None
    
    def get_port_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current connection.
        
        Returns:
            Dictionary with port details or None if disconnected
        """
        if not self.connected or not self.port:
            return None
        
        return {
            'port': self.current_port,
            'baud_rate': self.current_baud,
            'is_open': self.port.is_open,
            'in_waiting': self.port.in_waiting if self.port.is_open else 0
        }
    
    def __del__(self) -> None:
        """Destructor to ensure clean disconnection."""
        self.disconnect()