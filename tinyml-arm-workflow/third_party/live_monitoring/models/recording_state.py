# models/recording_state.py
"""
Recording state management for Sleep Stage Development Studio.
Tracks recording status, duration, and session metadata.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


class RecordingStatus(Enum):
    """Recording state machine states."""
    IDLE = auto()
    RECORDING = auto()
    PAUSED = auto()
    STOPPED = auto()


@dataclass
class SessionMetadata:
    """
    Session metadata stored in session.json.
    
    Attributes:
        session_id: Unique session identifier (timestamp)
        start_time: Recording start datetime
        end_time: Recording end datetime
        duration_seconds: Total recording duration
        board_name: Name of the connected board
        firmware_version: Firmware version string
        serial_port: Serial port used
        baudrate: Baud rate used
        sample_counts: Dictionary of packet counts by type
    """
    session_id: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    board_name: str = "Grove Vision AI V2"
    firmware_version: str = "0.9.4"
    serial_port: str = ""
    baudrate: int = 921600
    sample_counts: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "date": self.start_time.strftime("%Y-%m-%d") if self.start_time else "",
            "start_time": self.start_time.strftime("%H:%M:%S") if self.start_time else "",
            "end_time": self.end_time.strftime("%H:%M:%S") if self.end_time else "",
            "duration_seconds": round(self.duration_seconds, 1),
            "board_name": self.board_name,
            "firmware_version": self.firmware_version,
            "serial_port": self.serial_port,
            "baudrate": self.baudrate,
            "sample_counts": self.sample_counts,
            "total_packets": sum(self.sample_counts.values()) if self.sample_counts else 0
        }


@dataclass
class RecordingState:
    """
    Current recording state.
    
    Attributes:
        status: Current recording status
        session_id: Active session ID
        start_time: When recording started
        elapsed_seconds: Elapsed recording time
        packet_count: Total packets recorded
        session_path: Path to session folder
        metadata: Session metadata
    """
    status: RecordingStatus = RecordingStatus.IDLE
    session_id: str = ""
    start_time: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    packet_count: int = 0
    session_path: str = ""
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.status == RecordingStatus.RECORDING
    
    @property
    def is_idle(self) -> bool:
        """Check if idle."""
        return self.status == RecordingStatus.IDLE
    
    @property
    def formatted_duration(self) -> str:
        """Get formatted duration string HH:MM:SS."""
        hours = int(self.elapsed_seconds // 3600)
        minutes = int((self.elapsed_seconds % 3600) // 60)
        seconds = int(self.elapsed_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def reset(self) -> None:
        """Reset recording state to idle."""
        self.status = RecordingStatus.IDLE
        self.session_id = ""
        self.start_time = None
        self.elapsed_seconds = 0.0
        self.packet_count = 0
        self.session_path = ""
        self.metadata = SessionMetadata()