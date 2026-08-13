# models/prediction_model.py
"""
Prediction model for sleep stage inference results.
Stores current prediction, confidence scores, and inference metadata.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, ClassVar
from datetime import datetime


@dataclass
class PredictionModel:
    """
    Data model for one inference result.
    
    Stores the predicted sleep stage, confidence scores for all classes,
    and inference performance metrics.
    
    Attributes:
        timestamp: When prediction was received
        stage: Predicted sleep stage index (0-3)
        stage_name: Human-readable stage name
        confidence: Confidence of the predicted stage (0-1)
        scores: Confidence scores for all classes [wake, rem, light, deep]
        inference_time_ms: Inference duration in milliseconds
        buffer_filled: Number of epochs in sequence buffer
        buffer_total: Total epochs needed for inference
        raw_packet: Original UART packet text
    """
    timestamp: Optional[datetime] = None
    stage: int = -1
    stage_name: str = "Unknown"
    confidence: float = 0.0
    scores: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    inference_time_ms: float = 0.0
    buffer_filled: int = 0
    buffer_total: int = 30
    raw_packet: str = ""

    # ------------------------------------------------------------------
    # Stage metadata (class-level constants, NOT dataclass fields).
    # NOTE: matches board kSleepStageNames order in cvapp_mb_cls.cpp:
    #   0=Wake, 1=Light Sleep, 2=Deep Sleep, 3=REM
    # ------------------------------------------------------------------
    STAGE_NAMES: ClassVar[Dict[int, str]] = {
        0: "Wake",
        1: "Light Sleep",
        2: "Deep Sleep",
        3: "REM",
    }

    # Stage colors for visualization
    STAGE_COLORS: ClassVar[Dict[int, str]] = {
        0: "#f39c12",  # Wake - Orange
        1: "#3498db",  # Light - Blue
        2: "#2ecc71",  # Deep - Green
        3: "#9b59b6",  # REM - Purple
    }

    # Class display order and colors for confidence chart
    CLASS_ORDER: ClassVar[List[Dict[str, Any]]] = [
        {"index": 0, "name": "Wake", "color": (243, 156, 18)},
        {"index": 1, "name": "Light", "color": (52, 152, 219)},
        {"index": 2, "name": "Deep", "color": (46, 204, 113)},
        {"index": 3, "name": "REM", "color": (155, 89, 182)},
    ]

    @classmethod
    def from_uart(cls, packet_type: str, payload: List[str]) -> Optional['PredictionModel']:
        """
        Create PredictionModel from UART packet.
        
        Handles multiple formats:
        - PREDICTION=stage (name)
        - PREDICTION=stage,confidence
        - PRED,score0,score1,score2,score3,stage_name
        - SCORES=score0,score1,score2,score3
        
        Args:
            packet_type: "PRED" or "SCORES"
            payload: List of string values
        
        Returns:
            PredictionModel or None if parsing fails
        """
        try:
            model = cls(timestamp=datetime.now())
            
            if packet_type == "PRED":
                # Format: PRED,score0,score1,score2,score3,stage_name
                # or: PREDICTION=stage (name)
                if len(payload) >= 4:
                    scores = [float(v) for v in payload[:4]]
                    model.scores = scores
                    model.stage = scores.index(max(scores))
                    model.confidence = max(scores)
                    
                    if len(payload) >= 5:
                        model.stage_name = payload[4].strip()
                    else:
                        model.stage_name = cls.STAGE_NAMES.get(model.stage, "Unknown")
            
            elif packet_type == "SCORES":
                # Format: SCORES=score0,score1,score2,score3
                if len(payload) >= 4:
                    scores = [float(v) for v in payload[:4]]
                    model.scores = scores
                    model.stage = scores.index(max(scores))
                    model.confidence = max(scores)
                    model.stage_name = cls.STAGE_NAMES.get(model.stage, "Unknown")
            
            return model
            
        except (ValueError, IndexError):
            return None
    
    @classmethod
    def from_status(cls, payload: List[str]) -> Optional[Dict[str, int]]:
        """
        Parse buffer status from STATUS packet.
        
        Format: STATUS,epoch,current,total,total_epochs
        or: STATUS,filled,total
        
        Returns:
            Dict with 'filled' and 'total' keys, or None
        """
        try:
            if len(payload) >= 4 and payload[0].lower() == 'epoch':
                return {
                    'filled': int(payload[1]),
                    'total': int(payload[2])
                }
            elif len(payload) >= 2:
                return {
                    'filled': int(payload[0]),
                    'total': int(payload[1])
                }
        except (ValueError, IndexError):
            pass
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
            "stage": self.stage,
            "stage_name": self.stage_name,
            "confidence": round(self.confidence, 4),
            "scores": [round(s, 4) for s in self.scores],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "buffer_filled": self.buffer_filled,
            "buffer_total": self.buffer_total,
        }
    
    def get_color(self) -> str:
        """Get display color for current stage."""
        return self.STAGE_COLORS.get(self.stage, "#95a5a6")
    
    def clear(self) -> None:
        """Reset all values."""
        self.stage = -1
        self.stage_name = "Unknown"
        self.confidence = 0.0
        self.scores = [0.0, 0.0, 0.0, 0.0]