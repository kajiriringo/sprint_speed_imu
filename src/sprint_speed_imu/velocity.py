"""Velocity and distance estimation from forward acceleration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .errors import InvalidOptionError, ProcessingError
from .preprocess import smooth_signal


@dataclass(slots=True)
class VelocityResult:
    estimated_speed_mps: np.ndarray
    estimated_distance_m: np.ndarray
    smoothed_forward_mps2: np.ndarray
    diagnostics: dict[str, float | None]
    warnings: list[str]


def cumulative_trapezoid(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    out = np.zeros_like(values, dtype=float)
    if len(values) < 2:
        return out
    dt = np.diff(time_s)
    out[1:] = np.cumsum((values[:-1] + values[1:]) * 0.5 * dt)
    return out


def trapezoid_area(values: np.ndarray, time_s: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.trapezoid(values, time_s))


def estimate_velocity(
    time_s: np.ndarray,
    a_forward_mps2: np.ndarray,
    config: RunConfig,
) -> VelocityResult:
    warnings: list[str] = []
    if config.distance_source == "manual" and config.distance_m is None:
        raise InvalidOptionError("--distance-m is required when --distance-source=manual.")
    if config.distance_source not in {"manual", "auto"}:
        raise InvalidOptionError("--distance-source must be manual or auto.")

    a_smoothed = smooth_signal(
        a_forward_mps2,
        time_s,
        enabled=config.smooth,
        method=config.smoothing_method,
        kalman_process_noise=config.kalman_process_noise,
        kalman_measurement_noise=config.kalman_measurement_noise,
    )
    peak_retention_ratio = _peak_retention_ratio(a_forward_mps2, a_smoothed)
    if (
        config.smooth
        and config.smoothing_method == "kalman"
        and peak_retention_ratio is not None
        and peak_retention_ratio < 0.65
    ):
        warnings.append(
            f"Kalman smoothing retained {peak_retention_ratio:.2f} of peak forward acceleration; "
            "check noise parameters if acceleration peaks look flattened."
        )
    v_raw = cumulative_trapezoid(a_smoothed, time_s)

    if config.distance_source == "auto":
        if config.method == "heading" and len(v_raw) >= 2:
            drift = float(v_raw[-1] - v_raw[0])
            v_shape = v_raw - np.linspace(float(v_raw[0]), float(v_raw[-1]), len(v_raw))
            v_est = v_shape - float(np.nanmin(v_shape))
            warnings.append(
                "Heading auto velocity removed linear integration drift without distance or stride constraints."
            )
        else:
            drift = None
            v_est = v_raw - float(np.nanmin(v_raw))
        v_est = _smooth_post_integration(v_est, time_s, config)
        v_est = np.maximum(v_est, 0.0)
        x_est = cumulative_trapezoid(v_est, time_s)
        warnings.append(
            "WARNING: distance-source=auto estimates distance from IMU integration only. "
            "This is low confidence and may drift substantially."
        )
        return VelocityResult(
            estimated_speed_mps=v_est,
            estimated_distance_m=x_est,
            smoothed_forward_mps2=a_smoothed,
            diagnostics={
                "raw_distance_m": float(x_est[-1]) if len(x_est) else None,
                "manual_distance_correction_ratio": None,
                "forward_accel_peak_retention_ratio": peak_retention_ratio,
                "auto_velocity_drift_removed_mps": drift,
            },
            warnings=warnings,
        )

    target_distance = float(config.distance_m)
    mode = config.correction_mode
    raw_speed_shape = np.maximum(v_raw - float(np.nanmin(v_raw)), 0.0)
    raw_speed_area = trapezoid_area(raw_speed_shape, time_s)
    correction_ratio_for_qc: float | None = (
        target_distance / raw_speed_area if raw_speed_area > 1e-9 else None
    )
    shape_scale: float | None = None
    if mode == "mean-speed-shape":
        v_est, x_est, shape_scale = _mean_speed_shape(v_raw, time_s, target_distance, config)
    elif mode == "scale":
        v_shape = raw_speed_shape
        v_shape = _smooth_post_integration(v_shape, time_s, config)
        area = trapezoid_area(v_shape, time_s)
        if area <= 1e-9:
            raise ProcessingError("Velocity shape area is zero; cannot scale to manual distance.")
        shape_scale = target_distance / area
        correction_ratio_for_qc = shape_scale
        v_est = np.maximum(v_shape * shape_scale, 0.0)
        x_est = cumulative_trapezoid(v_est, time_s)
    elif mode == "bias":
        raw_distance = float(cumulative_trapezoid(v_raw, time_s)[-1])
        run_time = float(time_s[-1] - time_s[0])
        if run_time <= 0:
            raise ProcessingError("Run time must be positive.")
        bias = 2.0 * (raw_distance - target_distance) / (run_time**2)
        v_est = cumulative_trapezoid(a_smoothed - bias, time_s)
        v_est = np.maximum(v_est - float(np.nanmin(v_est)), 0.0)
        x_est = cumulative_trapezoid(v_est, time_s)
        if x_est[-1] > 1e-9:
            v_est *= target_distance / float(x_est[-1])
            x_est = cumulative_trapezoid(v_est, time_s)
        shape_scale = abs(bias)
    elif mode == "raw-integration":
        v_est = raw_speed_shape
        x_est = cumulative_trapezoid(v_est, time_s)
    else:
        raise InvalidOptionError(
            "--correction-mode must be mean-speed-shape, bias, scale, or raw-integration."
        )

    if len(x_est) and x_est[-1] > 1e-9:
        final_scale = target_distance / float(x_est[-1])
        v_est = np.maximum(v_est * final_scale, 0.0)
        x_est = cumulative_trapezoid(v_est, time_s)
    elif mode == "raw-integration":
        raise ProcessingError("Raw integration produced zero distance; cannot scale to manual distance.")

    if correction_ratio_for_qc is not None and abs(float(correction_ratio_for_qc)) > config.max_correction_ratio:
        warnings.append(
            f"Manual distance correction ratio {float(correction_ratio_for_qc):.3f} exceeds "
            f"{config.max_correction_ratio:.3f}."
        )

    return VelocityResult(
        estimated_speed_mps=v_est,
        estimated_distance_m=x_est,
        smoothed_forward_mps2=a_smoothed,
        diagnostics={
            "raw_distance_m": float(cumulative_trapezoid(v_raw, time_s)[-1]),
            "manual_distance_correction_ratio": None
            if correction_ratio_for_qc is None
            else float(correction_ratio_for_qc),
            "velocity_shape_scale": shape_scale,
            "forward_accel_peak_retention_ratio": peak_retention_ratio,
        },
        warnings=warnings,
    )


def _mean_speed_shape(
    v_raw: np.ndarray,
    time_s: np.ndarray,
    target_distance: float,
    config: RunConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    trend = np.linspace(float(v_raw[0]), float(v_raw[-1]), len(v_raw))
    v_shape = v_raw - trend
    v_shape = v_shape - float(np.nanmin(v_shape))
    v_shape = _smooth_post_integration(v_shape, time_s, config)
    v_shape = np.maximum(v_shape, 0.0)
    area = trapezoid_area(v_shape, time_s)
    if area <= 1e-9:
        run_time = float(time_s[-1] - time_s[0])
        if run_time <= 0:
            raise ProcessingError("Run time must be positive.")
        v_est = np.full_like(time_s, target_distance / run_time, dtype=float)
        x_est = cumulative_trapezoid(v_est, time_s)
        return v_est, x_est, 1.0
    correction_ratio = target_distance / area
    v_est = v_shape * correction_ratio
    x_est = cumulative_trapezoid(v_est, time_s)
    return v_est, x_est, correction_ratio


def _smooth_with_config(values: np.ndarray, time_s: np.ndarray, config: RunConfig) -> np.ndarray:
    return smooth_signal(
        values,
        time_s,
        enabled=config.smooth,
        method=config.smoothing_method,
        kalman_process_noise=config.kalman_process_noise,
        kalman_measurement_noise=config.kalman_measurement_noise,
    )


def _smooth_post_integration(values: np.ndarray, time_s: np.ndarray, config: RunConfig) -> np.ndarray:
    if config.smoothing_method == "kalman":
        return np.asarray(values, dtype=float).copy()
    return _smooth_with_config(values, time_s, config)


def _peak_retention_ratio(raw: np.ndarray, smoothed: np.ndarray) -> float | None:
    raw_peak = float(np.nanmax(np.abs(raw))) if len(raw) else 0.0
    if raw_peak <= 1e-9 or not np.isfinite(raw_peak):
        return None
    smooth_peak = float(np.nanmax(np.abs(smoothed)))
    return smooth_peak / raw_peak
