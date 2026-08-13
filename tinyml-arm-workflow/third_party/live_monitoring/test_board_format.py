# test_board_format.py
"""Test parser with both old and new board output formats."""

from serial_port.serial_parser import Parser
from serial_port.serial_protocol import PacketType


def test_board_outputs():
    """Test parser with actual board output formats."""
    parser = Parser()
    
    # Old format (what board currently sends)
    old_format_line = "F=0.123456,0.234567,0.345678,0.456789,0.567890,0.678901,0.789012,0.890123,0.901234,0.012345,0.123456,0.000000,0.234567,0.345678,0.456789,0.567890"
    
    # 18-feature format (what board sends after 16->18 fix)
    feature_18 = "0.123456,0.234567,0.345678,0.456789,0.567890,0.678901,0.789012,0.890123,0.901234,0.012345,0.123456,0.000000,0.234567,0.345678,0.456789,0.567890,0.678901,0.789012"
    
    # New format (what board should send after fix)
    new_format_lines = [
        "ACCEL,0.0200,-0.0100,0.9800",
        "PPG,51234",
        f"FEATURE,{feature_18}",
        "STATUS,epoch,1,total,960,uptime,30",
        "LOG,Epoch 1/960 completed",
        "PRED,2,0.87",
        "PREDICTION=2 (REM)",
        "SCORES=0.123456,0.234567,0.345678,0.567890",
    ]
    
    print("=== Testing Old Board Format ===")
    packet = parser.parse_line(old_format_line)
    print(f"Input:  {old_format_line[:60]}...")
    print(f"Type:   {packet.packet_type.name}")
    print(f"Payload count: {len(packet.payload)}")
    print(f"First 3 values: {packet.payload[:3]}")
    print()
    
    print("=== Testing New Board Format ===")
    for line in new_format_lines:
        packet = parser.parse_line(line)
        print(f"[{packet.packet_type.name}] {line[:50]}...")
        
        if packet.packet_type == PacketType.ACCEL:
            assert len(packet.payload) == 3, f"ACCEL should have 3 values, got {len(packet.payload)}"
        elif packet.packet_type == PacketType.PPG:
            assert len(packet.payload) == 1, f"PPG should have 1 value, got {len(packet.payload)}"
        elif packet.packet_type == PacketType.FEATURE:
            assert len(packet.payload) == 18, f"FEATURE should have 18 values, got {len(packet.payload)}"
    
    print("\n✅ All format tests passed!")


if __name__ == "__main__":
    test_board_outputs()