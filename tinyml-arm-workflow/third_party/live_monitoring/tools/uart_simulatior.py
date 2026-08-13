# tools/uart_simulator.py
"""
UART Data Simulator for Sleep Stage Development Studio.
Generates realistic sleep stage monitoring data for testing.
"""

import time
import random
import math
import sys


class SleepDataSimulator:
    """Simulates sleep monitoring device data output."""
    
    def __init__(self):
        self.sample_count = 0
        self.time = 0.0
        
    def generate_accel(self) -> str:
        """Generate simulated accelerometer data."""
        x = 0.01 * math.sin(self.time * 2) + random.gauss(0, 0.01)
        y = 0.01 * math.cos(self.time * 1.5) + random.gauss(0, 0.01)
        z = 0.98 + 0.02 * math.sin(self.time * 0.5) + random.gauss(0, 0.005)
        return f"ACCEL,{x:.4f},{y:.4f},{z:.4f}"
    
    def generate_ppg(self) -> str:
        """Generate simulated PPG data."""
        value = 512 + 10 * math.sin(self.time * 3) + random.gauss(0, 2)
        return f"PPG,{int(value)}"
    
    def generate_feature(self) -> str:
        """Generate simulated feature data."""
        if self.sample_count % 50 == 0:  # Every 50 samples
            mean = 512 + random.gauss(0, 5)
            std = 10 + random.gauss(0, 2)
            peak = mean + 3 * std + random.gauss(0, 5)
            return f"FEATURE,mean,{mean:.1f},std,{std:.1f},peak,{peak:.1f}"
        return None
    
    def generate_prediction(self) -> str:
        """Generate simulated sleep stage prediction."""
        if self.sample_count % 100 == 0:  # Every 100 samples
            stage = random.choice([0, 1, 2, 3, 4])  # Sleep stages
            confidence = 0.7 + random.random() * 0.3
            return f"PRED,{stage},{confidence:.2f}"
        return None
    
    def generate_status(self) -> str:
        """Generate simulated status message."""
        if self.sample_count % 200 == 0:  # Every 200 samples
            uptime = self.sample_count // 100
            return f"STATUS,running,uptime,{uptime}"
        return None
    
    def generate_log(self) -> str:
        """Generate simulated log message."""
        if self.sample_count == 1:
            return "LOG,Device initialized"
        if self.sample_count % 500 == 0:
            return f"LOG,Sample count: {self.sample_count}"
        return None
    
    def get_next_line(self) -> str:
        """Generate the next data line."""
        self.sample_count += 1
        self.time += 0.01  # 100 Hz sampling
        
        # Always generate ACCEL and PPG
        lines = [
            self.generate_accel(),
            self.generate_ppg()
        ]
        
        # Occasionally generate other packet types
        feature = self.generate_feature()
        if feature:
            lines.append(feature)
        
        prediction = self.generate_prediction()
        if prediction:
            lines.append(prediction)
        
        status = self.generate_status()
        if status:
            lines.append(status)
        
        log = self.generate_log()
        if log:
            lines.append(log)
        
        return random.choice(lines)


def main():
    """Run the UART simulator."""
    print("Sleep Stage Development Studio - UART Simulator")
    print("=" * 50)
    print("Generating simulated sleep monitoring data...")
    print("Press Ctrl+C to stop\n")
    
    simulator = SleepDataSimulator()
    
    try:
        while True:
            line = simulator.get_next_line()
            print(line, flush=True)
            time.sleep(0.01)  # 100 Hz output rate
    except KeyboardInterrupt:
        print("\n\nSimulator stopped.")
        print(f"Total samples generated: {simulator.sample_count}")


if __name__ == "__main__":
    main()