"""
Global configuration.
"""

ACC_FS = 50.0          # Hz
HR_FS = 0.2            # Hz

EPOCH_LENGTH = 30      # seconds

ACC_SAMPLES_PER_EPOCH = int(ACC_FS * EPOCH_LENGTH)   # 1500

# ===========================
# Epoch Cleaning
# ===========================

EPOCH_LENGTH = 30

MOTION_FS = 50

EXPECTED_MOTION_SAMPLES = MOTION_FS * EPOCH_LENGTH

EXPECTED_HR_SAMPLES = 6

MIN_MOTION_PERCENT = 0.80
MIN_HR_PERCENT = 0.50

MIN_MOTION_SAMPLES = int(
    EXPECTED_MOTION_SAMPLES * MIN_MOTION_PERCENT
)

MIN_HR_SAMPLES = int(
    EXPECTED_HR_SAMPLES * MIN_HR_PERCENT
)

DROP_UNKNOWN_LABEL = True

HRV_WINDOW_MINUTES = 5
# NOTE: sequence_length is controlled by configs/baseline.yaml (TrainConfig)
# Keep this value in sync for any legacy code that still imports config.py.
SEQUENCE_LENGTH = 30


UNKNOWN_LABEL = 5

INTERPOLATE_HR = False

