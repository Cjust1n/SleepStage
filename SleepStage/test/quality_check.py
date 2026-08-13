"""
quality_check.py

Dataset quality and signal validation utilities.

Author: Justin
"""

from typing import Any, Dict, Sequence, Tuple

import numpy as np


DEFAULT_HR_MIN = 30
DEFAULT_HR_MAX = 220
DEFAULT_SAMPLE_RATE_HZ = 50.0
DEFAULT_SAMPLE_RATE_TOLERANCE_HZ = 5.0


def _to_numpy_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def check_no_nan_or_inf(values: Sequence[Any]) -> Tuple[bool, str]:
    arr = _to_numpy_array(values)
    if np.isnan(arr).any():
        return False, "contains NaN"
    if np.isposinf(arr).any() or np.isneginf(arr).any():
        return False, "contains infinite values"
    return True, "no NaN or Inf"


def check_timestamps_monotonic(timestamps: Sequence[Any]) -> Tuple[bool, str]:
    arr = _to_numpy_array(timestamps)
    valid, msg = check_no_nan_or_inf(arr)
    if not valid:
        return False, msg
    if arr.size < 2:
        return True, "too few timestamps to detect monotonicity"
    diff = np.diff(arr)
    if np.any(diff <= 0):
        count = int(np.sum(diff <= 0))
        return False, f"timestamps are not strictly increasing ({count} non-positive interval(s))"
    return True, "timestamps are monotonically increasing"


def estimate_sampling_rate_hz(timestamps: Sequence[Any]) -> float:
    arr = _to_numpy_array(timestamps)
    if arr.size < 2:
        return 0.0
    diff = np.diff(arr)
    diff = diff[np.isfinite(diff) & (diff > 0)]
    if diff.size == 0:
        return 0.0
    return 1.0 / np.mean(diff)


def check_sampling_rate_around_50hz(
    timestamps: Sequence[Any],
    target_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    tolerance_hz: float = DEFAULT_SAMPLE_RATE_TOLERANCE_HZ,
) -> Tuple[bool, float, str]:
    if target_hz <= 0 or tolerance_hz < 0:
        raise ValueError("target_hz must be > 0 and tolerance_hz must be >= 0")

    fs = estimate_sampling_rate_hz(timestamps)
    if fs <= 0:
        return False, fs, "invalid or empty timestamp series"

    min_rate = target_hz - tolerance_hz
    max_rate = target_hz + tolerance_hz
    if min_rate <= fs <= max_rate:
        return True, fs, "sampling rate is within expected range"

    return False, fs, f"sampling rate {fs:.2f} Hz is outside {min_rate:.1f}-{max_rate:.1f} Hz"


def check_hr_values(
    hr_values: Sequence[Any],
    min_bpm: int = DEFAULT_HR_MIN,
    max_bpm: int = DEFAULT_HR_MAX,
) -> Dict[str, Any]:
    arr = _to_numpy_array(hr_values)
    results = {
        "has_nan": False,
        "has_inf": False,
        "positive": True,
        "in_range": True,
        "min_value": float(np.nan),
        "max_value": float(np.nan),
    }

    if arr.size == 0:
        results.update({
            "positive": False,
            "in_range": False,
            "min_value": float("nan"),
            "max_value": float("nan"),
        })
        return results

    results["has_nan"] = np.isnan(arr).any()
    results["has_inf"] = np.isposinf(arr).any() or np.isneginf(arr).any()
    results["min_value"] = float(np.nanmin(arr))
    results["max_value"] = float(np.nanmax(arr))
    results["positive"] = np.all(arr > 0)
    results["in_range"] = np.all((arr >= min_bpm) & (arr <= max_bpm))

    return results


def check_night_quality(night: Any) -> Dict[str, Any]:
    motion_ts = night.motion["Timestamp"]
    hr_ts = night.hr["Timestamp"]
    hr_values = night.hr["HR"]

    motion_ts_valid, motion_ts_msg = check_timestamps_monotonic(motion_ts)
    hr_ts_valid, hr_ts_msg = check_timestamps_monotonic(hr_ts)
    motion_nan_ok, motion_nan_msg = check_no_nan_or_inf(motion_ts)
    hr_nan_ok, hr_nan_msg = check_no_nan_or_inf(hr_ts)
    hr_values_check = check_hr_values(hr_values)
    motion_rate_ok, motion_rate_hz, motion_rate_msg = check_sampling_rate_around_50hz(motion_ts)
    hr_rate_ok, hr_rate_hz, hr_rate_msg = check_sampling_rate_around_50hz(hr_ts)

    return {
        "motion_timestamp_monotonic": motion_ts_valid,
        "motion_timestamp_message": motion_ts_msg,
        "hr_timestamp_monotonic": hr_ts_valid,
        "hr_timestamp_message": hr_ts_msg,
        "motion_timestamp_no_nan_inf": motion_nan_ok,
        "motion_timestamp_message_nan_inf": motion_nan_msg,
        "hr_timestamp_no_nan_inf": hr_nan_ok,
        "hr_timestamp_message_nan_inf": hr_nan_msg,
        "motion_sampling_rate_hz": motion_rate_hz,
        "motion_sampling_rate_ok": motion_rate_ok,
        "motion_sampling_rate_message": motion_rate_msg,
        "hr_sampling_rate_hz": hr_rate_hz,
        "hr_sampling_rate_ok": hr_rate_ok,
        "hr_sampling_rate_message": hr_rate_msg,
        "hr_has_nan": hr_values_check["has_nan"],
        "hr_has_inf": hr_values_check["has_inf"],
        "hr_positive": hr_values_check["positive"],
        "hr_in_physio_range": hr_values_check["in_range"],
        "hr_min_value": hr_values_check["min_value"],
        "hr_max_value": hr_values_check["max_value"],
    }
