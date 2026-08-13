# serial/protocol.py (FIXED VERSION)
"""
Protocol definitions for Sleep Stage Development Studio.
Provides packet type identification without payload parsing.
"""

from enum import Enum, auto
from typing import List, Optional


class PacketType(Enum):
    """Enumeration of supported packet types."""
    LOG = auto()        # Debug/informational log messages
    ACCEL = auto()      # Accelerometer data (3-axis)
    PPG = auto()        # Photoplethysmography sensor data
    FEATURE = auto()    # Extracted features from signal processing
    PRED = auto()       # Sleep stage predictions
    STATUS = auto()     # Device status information
    UNKNOWN = auto()    # Unrecognized packet type


class Protocol:
    """
    Protocol identification layer for UART communication.
    
    Identifies packet types based on prefix strings.
    """
    
    # Standard comma-separated prefixes
    PREFIX_MAP = {
        "LOG": PacketType.LOG,
        "ACCEL": PacketType.ACCEL,
        "PPG": PacketType.PPG,
        "FEATURE": PacketType.FEATURE,
        "PRED": PacketType.PRED,
        "STATUS": PacketType.STATUS,
    }
    
    @classmethod
    def identify_packet(cls, line: str) -> PacketType:
        """
        Identify the packet type from a raw UART line.
        
        Handles multiple formats:
        - Standard: PREFIX,val1,val2,...
        - Legacy: F=val1,val2,...
        - Key-value: PREDICTION=value, SCORES=values
        
        Args:
            line: Raw line received from UART
        
        Returns:
            PacketType enum value
        
        Examples:
            >>> Protocol.identify_packet("ACCEL,0.01,-0.02,0.98")
            PacketType.ACCEL
            >>> Protocol.identify_packet("F=0.1,0.2,0.3")
            PacketType.FEATURE
            >>> Protocol.identify_packet("PREDICTION=2 (REM)")
            PacketType.PRED
        """
        line = line.strip()
        
        if not line:
            return PacketType.UNKNOWN
        
        # Check for equals-sign format first
        if '=' in line:
            key = line.split('=', 1)[0].strip().upper()
            
            # Legacy feature format
            if key == 'F':
                return PacketType.FEATURE
            
            # Prediction format
            if key == 'PREDICTION':
                return PacketType.PRED
            
            # Scores format (route to LOG for console display)
            if key == 'SCORES':
                return PacketType.LOG
        
        # Standard comma-separated format
        if ',' in line:
            prefix = line.split(',', 1)[0].strip().upper()
        else:
            prefix = line.strip().upper()
        
        return cls.PREFIX_MAP.get(prefix, PacketType.UNKNOWN)
    
    @classmethod
    def get_supported_prefixes(cls) -> List[str]:
        """
        Get list of all supported protocol prefixes.
        
        Returns:
            List of supported prefix strings
        """
        prefixes = list(cls.PREFIX_MAP.keys())
        prefixes.extend(['F (legacy)', 'PREDICTION', 'SCORES'])
        return sorted(prefixes)
    
    @classmethod
    def is_valid_packet(cls, line: str) -> bool:
        """
        Check if a line contains a valid protocol packet.
        
        Args:
            line: Raw line to validate
        
        Returns:
            True if recognized, False otherwise
        """
        return cls.identify_packet(line) != PacketType.UNKNOWN