"""
hrv.py

Extract HRV features from a rolling HR window.

Author: Justin
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch


class HRVFeatureExtractor:

    def __init__(
        self,
        window_minutes=5,
        resample_fs=4,
        trend_window=5,
    ):
        self.window_minutes = window_minutes
        self.resample_fs = resample_fs
        self.trend_window = trend_window
        self._history: list[float] = []

    def extract(self, hr_window):
        """
        Parameters
        ----------
        hr_window : pandas.DataFrame
            HR samples covering a window of multiple epochs.

        Returns
        -------
        dict
        """
        if hr_window is None or len(hr_window) < 2:
            return self.empty()

        hr = hr_window["HR"].to_numpy(dtype=np.float64)
        timestamps = hr_window["Timestamp"].to_numpy(dtype=np.float64)

        if len(hr) < 2 or len(timestamps) < 2:
            return self.empty()

        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        hr = hr[sort_idx]

        ibi, ibi_timestamps = self.compute_ibi(hr, timestamps)
        diff = np.diff(ibi)

        time_domain = self.compute_time_domain(ibi, diff)
        sd1, sd2 = self.compute_poincare(time_domain["sdnn"], time_domain["sdsd"])
        lf, hf, lf_hf = self.compute_frequency_domain(ibi_timestamps, ibi)

        features = {
            "mean_ibi": float(time_domain["mean_ibi"]),
            "mean_hr": float(time_domain["mean_hr"]),
            "sdnn": float(time_domain["sdnn"]),
            "rmssd": float(time_domain["rmssd"]),
            "sdsd": float(time_domain["sdsd"]),
            "nn50": int(time_domain["nn50"]),
            "pnn50": float(time_domain["pnn50"]),
            "lf": float(lf),
            "hf": float(hf),
            "lf_hf": float(lf_hf),
            "sd1": float(sd1),
            "sd2": float(sd2),
            "hr_delta": 0.0,
            "rolling_mean_hr": float(time_domain["mean_hr"]),
            "rolling_std_hr": 0.0,
            "hr_slope": 0.0,
            "rolling_hr_range": 0.0,
        }

        current_mean_hr = float(time_domain["mean_hr"])
        history = self._history[-max(1, int(self.trend_window) - 1):]
        prev_means = history[-1:] if history else []

        if prev_means:
            features["hr_delta"] = float(current_mean_hr - prev_means[-1])

        rolling_values = history + [current_mean_hr]
        if len(rolling_values) >= 1:
            features["rolling_mean_hr"] = float(np.mean(rolling_values))
            features["rolling_std_hr"] = float(np.std(rolling_values)) if len(rolling_values) > 1 else 0.0
            features["rolling_hr_range"] = float(np.max(rolling_values) - np.min(rolling_values)) if len(rolling_values) > 1 else 0.0

        if len(rolling_values) >= 2:
            x = np.arange(len(rolling_values), dtype=np.float64)
            slope = np.polyfit(x, np.asarray(rolling_values, dtype=np.float64), deg=1)[0]
            features["hr_slope"] = float(slope)

        self._history.append(current_mean_hr)
        if len(self._history) > max(1, int(self.trend_window)):
            self._history = self._history[-max(1, int(self.trend_window)) :]

        return features

    def compute_ibi(self, hr, timestamps):
        ibi = 60000.0 / hr
        return ibi, timestamps

    def compute_time_domain(self, ibi, diff):
        mean_ibi = np.mean(ibi) if len(ibi) else 0.0
        mean_hr = 60000.0 / mean_ibi if mean_ibi > 0 else 0.0
        sdnn = np.std(ibi) if len(ibi) else 0.0
        rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) else 0.0
        sdsd = np.std(diff) if len(diff) else 0.0
        nn50 = int(np.sum(np.abs(diff) > 50)) if len(diff) else 0
        pnn50 = float(np.mean(np.abs(diff) > 50)) if len(diff) else 0.0

        return {
            "mean_ibi": mean_ibi,
            "mean_hr": mean_hr,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "sdsd": sdsd,
            "nn50": nn50,
            "pnn50": pnn50,
        }

    def compute_poincare(self, sdnn, sdsd):
        sd1 = np.sqrt(0.5) * sdsd
        temp = 2 * (sdnn ** 2) - 0.5 * (sdsd ** 2)
        sd2 = np.sqrt(max(temp, 0.0))
        return sd1, sd2

    def compute_frequency_domain(self, ibi_timestamps, ibi):
        timestamps = np.asarray(ibi_timestamps, dtype=np.float64)
        ibi_values = np.asarray(ibi, dtype=np.float64)

        if len(timestamps) < 4 or len(ibi_values) < 4:
            return 0.0, 0.0, 0.0

        if not np.isfinite(timestamps).all() or not np.isfinite(ibi_values).all():
            return 0.0, 0.0, 0.0

        if timestamps[-1] <= timestamps[0]:
            return 0.0, 0.0, 0.0

        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        ibi_values = ibi_values[sort_idx]

        if np.std(ibi_values) < 1e-6:
            return 0.0, 0.0, 0.0

        kind = "cubic" if len(ibi_values) >= 6 else "linear"
        interpolator = interp1d(
            timestamps,
            ibi_values,
            kind=kind,
            fill_value="extrapolate",
            assume_sorted=True,
        )

        resample_start = timestamps[0]
        resample_end = timestamps[-1]
        resample_step = 1.0 / self.resample_fs
        resample_ts = np.arange(
            resample_start,
            resample_end + resample_step / 2.0,
            resample_step,
        )
        if len(resample_ts) < 8:
            return 0.0, 0.0, 0.0

        ibi_interp = interpolator(resample_ts)
        if not np.isfinite(ibi_interp).all():
            return 0.0, 0.0, 0.0

        ibi_detrended = ibi_interp - np.mean(ibi_interp)
        if np.std(ibi_detrended) < 1e-6:
            return 0.0, 0.0, 0.0

        nperseg = min(256, len(ibi_detrended))
        nperseg = max(8, nperseg)
        if nperseg > len(ibi_detrended):
            nperseg = len(ibi_detrended)
        if nperseg < 8:
            return 0.0, 0.0, 0.0

        f, pxx = welch(
            ibi_detrended,
            fs=self.resample_fs,
            nperseg=nperseg,
        )

        if len(f) < 2:
            return 0.0, 0.0, 0.0

        lf_mask = (f >= 0.04) & (f < 0.15)
        hf_mask = (f >= 0.15) & (f < 0.40)

        trapz_func = getattr(np, "trapezoid", None)
        if trapz_func is None:
            trapz_func = np.trapz
        lf_raw = trapz_func(pxx[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
        hf_raw = trapz_func(pxx[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0

        if not np.isfinite(lf_raw) or lf_raw < 0.0:
            lf_raw = 0.0
        if not np.isfinite(hf_raw) or hf_raw < 0.0:
            hf_raw = 0.0

        if lf_raw + hf_raw <= 0.0:
            return 0.0, 0.0, 0.0

        eps = 1e-6
        if hf_raw < 1e-4:
            return 0.0, 0.0, 0.0

        lf = float(np.log1p(max(lf_raw, eps)))
        hf = float(np.log1p(max(hf_raw, eps)))
        lf_hf = float(np.log1p(max(lf_raw / max(hf_raw, eps), 0.0)))

        return float(lf), float(hf), float(lf_hf)

    def empty(self):
        return {
            "mean_ibi": 0.0,
            "mean_hr": 0.0,
            "sdnn": 0.0,
            "rmssd": 0.0,
            "sdsd": 0.0,
            "nn50": 0,
            "pnn50": 0.0,
            "lf": 0.0,
            "hf": 0.0,
            "lf_hf": 0.0,
            "sd1": 0.0,
            "sd2": 0.0,
            "hr_delta": 0.0,
            "rolling_mean_hr": 0.0,
            "rolling_std_hr": 0.0,
            "hr_slope": 0.0,
            "rolling_hr_range": 0.0,
        }

    @staticmethod
    def feature_names():
        return [
            "mean_ibi",
            "mean_hr",
            "sdnn",
            "rmssd",
            "sdsd",
            "nn50",
            "pnn50",
            "lf",
            "hf",
            "lf_hf",
            "sd1",
            "sd2",
            "hr_delta",
            "rolling_mean_hr",
            "rolling_std_hr",
            "hr_slope",
            "rolling_hr_range",
        ]

    def to_vector(self, feature_dict):
        return np.array(
            [feature_dict[name] for name in self.feature_names()],
            dtype=np.float32,
        )
