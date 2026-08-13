# serial/parser.py (FIXED VERSION)
"""
Packet parser for Sleep Stage Development Studio.
Converts raw UART lines into structured packet objects.
Handles both old (F=) and new (PREFIX,) board formats.
"""

from typing import Dict, List, Optional, Any
from serial_port.serial_protocol import Protocol, PacketType


class Packet:
    """Structured representation of a parsed UART packet."""
    
    def __init__(self, packet_type: PacketType, raw_line: str, payload: List[str]) -> None:
        self.packet_type: PacketType = packet_type
        self.raw_line: str = raw_line.strip()
        self.payload: List[str] = payload
        self.timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.packet_type.name,
            "payload": self.payload,
            "raw": self.raw_line,
            "timestamp": self.timestamp
        }
    
    def __repr__(self) -> str:
        return f"Packet(type={self.packet_type.name}, payload_count={len(self.payload)})"
    
    def __str__(self) -> str:
        return f"[{self.packet_type.name}] {', '.join(self.payload[:3])}{'...' if len(self.payload) > 3 else ''}"


class Parser:
    """
    UART line parser that converts raw text into structured Packet objects.
    
    Handles multiple board output formats:
    - Standard: PREFIX,val1,val2,...
    - Legacy: F=val1,val2,...
    - Key-value: KEY=value
    - PREDICTION=stage (name)
    - SCORES=score1,score2,...
    """
    
    def __init__(self) -> None:
        """Initialize the parser."""
        self.protocol = Protocol()
    
    def parse_line(self, line: str) -> Packet:
        """
        Parse a single UART line into a structured Packet object.
        
        Args:
            line: Raw line from UART
        
        Returns:
            Packet object with type and payload fields
        
        Examples:
            >>> parser = Parser()
            >>> p = parser.parse_line("ACCEL,0.02,-0.01,0.98")
            >>> p.packet_type == PacketType.ACCEL
            True
            >>> p = parser.parse_line("F=0.1,0.2,0.3")
            >>> p.packet_type == PacketType.FEATURE
            True
        """
        cleaned_line = line.strip()
        
        if not cleaned_line:
            return Packet(PacketType.UNKNOWN, cleaned_line, [])
        
        # Handle different formats
        if '=' in cleaned_line and not cleaned_line.startswith(tuple(self.protocol.PREFIX_MAP.keys())):
            return self._parse_equals_format(cleaned_line)
        else:
            return self._parse_comma_format(cleaned_line)
    
    def _parse_comma_format(self, line: str) -> Packet:
        """
        Parse standard comma-separated format: PREFIX,val1,val2,...
        
        Args:
            line: Raw line in comma format
        
        Returns:
            Parsed Packet
        """
        # Extract prefix (before first comma)
        if ',' in line:
            prefix = line.split(',', 1)[0].strip().upper()
            payload_str = line.split(',', 1)[1] if ',' in line else ""
            payload = [field.strip() for field in payload_str.split(',')] if payload_str else []
        else:
            prefix = line.strip().upper()
            payload = [line]
        
        # Get packet type from protocol
        packet_type = self.protocol.identify_packet(line)
        
        return Packet(packet_type=packet_type, raw_line=line, payload=payload)
    
    def _parse_equals_format(self, line: str) -> Packet:
        """
        Parse equals-sign format: F=val1,val2,...
        """
        key, value = line.split('=', 1)
        key = key.strip().upper()
        value = value.strip()
        
        if key == 'F':
            if ',' in value:
                # F=0.1,0.2,...,0.9 (16 values)
                payload = [v.strip() for v in value.split(',')]
                return Packet(PacketType.FEATURE, line, payload)
            else:
                return Packet(PacketType.FEATURE, line, [value])
        
        elif key == 'PREDICTION':
            parts = value.split()
            payload = [parts[0]] if parts else [value]
            return Packet(PacketType.PRED, line, payload)
        
        elif key == 'SCORES':
            # SCORES carries the per-class confidence scores. Route it as PRED
            # so main_window can update the prediction card & confidence chart.
            payload = [v.strip() for v in value.split(',')]
            return Packet(PacketType.PRED, line, payload)
        
        else:
            return Packet(PacketType.UNKNOWN, line, [line])
    
    def parse_multiple_lines(self, lines: List[str]) -> List[Packet]:
        """
        Parse multiple UART lines into Packet objects.
        
        Args:
            lines: List of raw UART lines
        
        Returns:
            List of parsed Packet objects
        """
        return [self.parse_line(line) for line in lines if line.strip()]