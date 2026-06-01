"""Preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

from .errors import InputFormatError


def interpolate_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].interpolate(method="linear", limit_direction="both")
    return out


def validate_time_axis(df: pd.DataFrame) -> None:
    time = df["time_s"].to_numpy(dtype=float)
    if not np.all(np.isfinite(time)):
        raise InputFormatError("time_s contains missing or non-finite values.")
    if np.any(np.diff(time) <= 0):
        raise InputFormatError("time_s must be strictly increasing.")


def estimate_sample_rate_hz(time_s: np.ndarray) -> float | None:
    if len(time_s) < 2:
        return None
    dt = np.diff(time_s)
    med_dt = float(np.median(dt))
    if med_dt <= 0:
        return None
    return 1.0 / med_dt


def time_jitter_warning(time_s: np.ndarray) -> str | None:
    if len(time_s) < 4:
        return None
    dt = np.diff(time_s)
    med_dt = float(np.median(dt))
    if med_dt <= 0:
        return None
    if float(np.std(dt) / med_dt) > 0.05:
        return "Sample interval jitter is above 5%."
    return None


def slice_run(df: pd.DataFrame, start_time: float, end_time: float) -> pd.DataFrame:
    if end_time <= start_time:
        raise InputFormatError("end_time must be greater than start_time.")
    mask = (df["time_s"] >= start_time) & (df["time_s"] <= end_time)
    out = df.loc[mask].copy()
    if len(out) < 3:
        raise InputFormatError("Analysis interval is too short.")
    if float(out["time_s"].iloc[-1] - out["time_s"].iloc[0]) < 1.0:
        raise InputFormatError("Analysis interval must be at least 1 second.")
    return out


def smooth_signal(
    values: np.ndarray,
    time_s: np.ndarray,
    enabled: bool = True,
    method: str = "savgol",
    kalman_process_noise: float = 0.05,
    kalman_measurement_noise: float = 0.5,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not enabled or method == "none" or len(values) < 7:
        return values.copy()
    if method == "savgol":
        window = max(5, int(round(len(values) * 0.05)) | 1)
        window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
        if window < 5:
            return values.copy()
        polyorder = min(3, window - 2)
        return savgol_filter(values, window_length=window, polyorder=polyorder)
    if method == "butter":
        if len(time_s) < 12:
            return values.copy()
        dt = float(np.median(np.diff(time_s)))
        fs = 1.0 / dt
        cutoff = min(8.0, fs * 0.2)
        if cutoff <= 0 or cutoff >= fs / 2:
            return values.copy()
        b, a = butter(3, cutoff / (fs / 2), btype="low")
        return filtfilt(b, a, values)
    if method == "kalman":
        return kalman_rts_smooth_1d(
            values,
            time_s,
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
        )
    return values.copy()


def kalman_rts_smooth_1d(
    values: np.ndarray,
    time_s: np.ndarray,
    process_noise: float = 0.05,
    measurement_noise: float = 0.5,
) -> np.ndarray:
    """Random-walk Kalman filter followed by RTS smoothing for one signal."""
    z = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if len(z) < 2:
        return z.copy()
    if process_noise <= 0 or measurement_noise <= 0:
        raise InputFormatError("Kalman process and measurement noise must be positive.")

    finite = np.isfinite(z)
    if not np.all(finite):
        z = z.copy()
        valid_x = np.flatnonzero(finite)
        if len(valid_x) == 0:
            return np.zeros_like(z)
        z[~finite] = np.interp(np.flatnonzero(~finite), valid_x, z[finite])

    dt = np.diff(time_s)
    med_dt = float(np.median(dt)) if len(dt) else 1.0
    if med_dt <= 0 or not np.isfinite(med_dt):
        med_dt = 1.0

    n = len(z)
    filtered = np.empty(n, dtype=float)
    pred_state = np.empty(n, dtype=float)
    pred_cov = np.empty(n, dtype=float)
    filt_cov = np.empty(n, dtype=float)

    state = float(z[0])
    cov = float(measurement_noise)
    filtered[0] = state
    pred_state[0] = state
    pred_cov[0] = cov
    filt_cov[0] = cov

    for i in range(1, n):
        dt_scale = max(float(time_s[i] - time_s[i - 1]) / med_dt, 1e-3)
        q = process_noise * dt_scale
        state_pred = state
        cov_pred = cov + q
        gain = cov_pred / (cov_pred + measurement_noise)
        state = state_pred + gain * (float(z[i]) - state_pred)
        cov = (1.0 - gain) * cov_pred
        pred_state[i] = state_pred
        pred_cov[i] = cov_pred
        filtered[i] = state
        filt_cov[i] = cov

    smoothed = filtered.copy()
    smooth_cov = filt_cov.copy()
    for i in range(n - 2, -1, -1):
        denom = pred_cov[i + 1]
        if denom <= 1e-12:
            continue
        gain = filt_cov[i] / denom
        smoothed[i] = filtered[i] + gain * (smoothed[i + 1] - pred_state[i + 1])
        smooth_cov[i] = filt_cov[i] + gain * gain * (smooth_cov[i + 1] - pred_cov[i + 1])
    return smoothed
