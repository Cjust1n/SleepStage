# test_parse_debug.py
"""Test parser with exact board output format."""
from serial_port.serial_parser import Parser

parser = Parser()

# Simulasi output board
test_lines = [
    "F=0.003128,37.443292,19.016666,78.125000,1388.136832,77.459664,0.960839,0.047007,305,0,0,0,0,2,1.5,67",
    "ACCEL,0.023,-0.015,0.987",
    "PPG,51234",
    "STATUS,epoch,1,uptime,30",
]

for line in test_lines:
    packet = parser.parse_line(line)
    print(f"Input:  {line[:60]}...")
    print(f"Type:   {packet.packet_type.name}")
    print(f"Payload: {packet.payload[:3]}...")
    print()