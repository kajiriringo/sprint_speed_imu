"""Top-level analysis pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig
from .direction import compute_direction
from .distance import build_speed_curve
from .errors import InputFormatError, InvalidOptionError, ProcessingError
from .io import read_imu_csv
from .preprocess import interpolate_numeric, slice_run, time_jitter_warning, validate_time_axis
from .qc import add_qc_warnings, build_qc, confidence_labels
from .report import ASSUMPTIONS, ensure_output_dir, write_artifacts
from .schema import validate_required_columns
from .start_detect import detect_start_time
from .velocity import estimate_velocity

ESTIMATED_WARNING = "Speed, distance, and max-speed location are estimated from IMU data."


def preflight_analysis(config: RunConfig) -> None:
    _validate_config(config)
    df, _, _ = read_imu_csv(config)
    df = interpolate_numeric(df)
    validate_required_columns(df, config.method)
    validate_time_axis(df)
    start_time, _ = detect_start_time(df, config)
    end_time = _resolve_end_time(df, config, start_time)
    run_df = slice_run(df, start_time, end_time)
    actual_start = float(run_df["time_s"].iloc[0])
    actual_end = float(run_df["time_s"].iloc[-1])
    direction = compute_direction(df, run_df, config, actual_start, actual_end)
    time = run_df["time_s"].to_numpy(dtype=float)
    estimate_velocity(time, direction.a_forward_mps2, config)


def run_analysis(config: RunConfig) -> dict[str, Any]:
    _validate_config(config)
    ensure_output_dir(config.output_dir, config.overwrite)

    df, input_meta, warnings = read_imu_csv(config)
    df = interpolate_numeric(df)
    validate_required_columns(df, config.method)
    validate_time_axis(df)
    jitter = time_jitter_warning(df["time_s"].to_numpy(dtype=float))
    if jitter:
        warnings.append(jitter)

    start_time, start_warnings = detect_start_time(df, config)
    warnings.extend(start_warnings)
    end_time = _resolve_end_time(df, config, start_time)
    run_df = slice_run(df, start_time, end_time)
    actual_start = float(run_df["time_s"].iloc[0])
    actual_end = float(run_df["time_s"].iloc[-1])

    if config.end_mode == "data-end" and config.duration_s is None:
        warnings.append("end-mode=data-end uses the last CSV timestamp; trim post-run data when possible.")

    direction = compute_direction(df, run_df, config, actual_start, actual_end)
    warnings.extend(direction.warnings)

    time = run_df["time_s"].to_numpy(dtype=float)
    velocity = estimate_velocity(time, direction.a_forward_mps2, config)
    warnings.extend(velocity.warnings)
    warnings.append(ESTIMATED_WARNING)

    qc = build_qc(df, velocity.estimated_speed_mps, direction.diagnostics, velocity.diagnostics, config)
    warnings.extend(add_qc_warnings(qc, config))
    warnings = _dedupe(warnings)
    _raise_for_strict_warnings(config, warnings)
    confidence = confidence_labels(config, direction.diagnostics, warnings)

    curve = build_speed_curve(
        time_s=time,
        distance_m=velocity.estimated_distance_m,
        estimated_speed_mps=velocity.estimated_speed_mps,
        a_forward_mps2=velocity.smoothed_forward_mps2,
        distance_bin_m=config.distance_bin_m,
        confidence_flag=confidence["overall"],
    )

    max_idx = int(np.nanargmax(velocity.estimated_speed_mps))
    final_distance = float(velocity.estimated_distance_m[-1])
    run_time = float(actual_end - actual_start)
    average_speed = final_distance / run_time if run_time > 0 else None
    if config.distance_source == "manual" and config.distance_m is not None and run_time > 0:
        average_speed = float(config.distance_m) / run_time

    summary: dict[str, Any] = {
        "tool": "sprint-speed-imu",
        "version": "0.1.0",
        "input": input_meta,
        "run": {
            "method": config.method,
            "distance_source": config.distance_source,
            "distance_m": config.distance_m,
            "start_time_s": actual_start,
            "end_time_s": actual_end,
            "run_time_s": run_time,
            "correction_mode": config.correction_mode,
            "smooth": config.smooth,
            "smoothing_method": config.smoothing_method,
            "kalman_process_noise": config.kalman_process_noise
            if config.smoothing_method == "kalman"
            else None,
            "kalman_measurement_noise": config.kalman_measurement_noise
            if config.smoothing_method == "kalman"
            else None,
        },
        "results": {
            "average_speed_mps": average_speed,
            "final_estimated_distance_m": final_distance,
            "max_estimated_speed_mps": float(velocity.estimated_speed_mps[max_idx]),
            "max_estimated_speed_distance_m": float(velocity.estimated_distance_m[max_idx]),
            "max_estimated_speed_time_s": float(time[max_idx]),
        },
        "confidence": confidence,
        "qc": qc,
        "diagnostics": direction.diagnostics,
        "velocity_diagnostics": velocity.diagnostics,
        "warnings": warnings,
        "assumptions": ASSUMPTIONS,
    }

    debug_df = _build_debug_dataframe(run_df, direction, velocity) if config.debug else None
    write_artifacts(config, curve, debug_df, summary, warnings)
    return summary


def _validate_config(config: RunConfig) -> None:
    if config.method not in {"pca", "attitude"}:
        raise InvalidOptionError("--method must be attitude or pca.")
    if config.distance_source == "manual" and config.distance_m is None:
        raise InvalidOptionError("--distance-m is required when --distance-source=manual.")
    if config.distance_source == "manual" and config.distance_m is not None and config.distance_m <= 0:
        raise InvalidOptionError("--distance-m must be positive.")
    if config.distance_source not in {"manual", "auto"}:
        raise InvalidOptionError("--distance-source must be manual or auto.")
    if config.start_mode == "manual" and config.start_time is None:
        raise InvalidOptionError("--start-time is required when --start-mode=manual.")
    if config.end_mode == "manual" and config.end_time is None and config.duration_s is None:
        raise InvalidOptionError("--end-time is required when --end-mode=manual.")
    if config.distance_bin_m <= 0:
        raise InvalidOptionError("--distance-bin-m must be positive.")
    if config.kalman_process_noise <= 0:
        raise InvalidOptionError("--kalman-process-noise must be positive.")
    if config.kalman_measurement_noise <= 0:
        raise InvalidOptionError("--kalman-measurement-noise must be positive.")


def _resolve_end_time(df: pd.DataFrame, config: RunConfig, start_time: float) -> float:
    data_end = float(df["time_s"].iloc[-1])
    if config.duration_s is not None:
        end_time = float(start_time + config.duration_s)
    elif config.end_mode == "manual":
        if config.end_time is None:
            raise InvalidOptionError("--end-time is required when --end-mode=manual.")
        end_time = float(config.end_time)
    elif config.end_mode == "data-end":
        end_time = data_end
    else:
        raise InvalidOptionError("--end-mode must be data-end or manual.")
    if end_time > data_end + 1e-9:
        raise InputFormatError("Requested end time exceeds input data range.")
    return end_time


def _build_debug_dataframe(run_df, direction, velocity) -> pd.DataFrame:
    def col(name: str) -> np.ndarray:
        if name in run_df.columns:
            return run_df[name].to_numpy(dtype=float)
        return np.full(len(run_df), np.nan)

    return pd.DataFrame(
        {
            "time_s": run_df["time_s"].to_numpy(dtype=float),
            "ax_mps2": col("ax"),
            "ay_mps2": col("ay"),
            "az_mps2": col("az"),
            "gx_radps": col("gx"),
            "gy_radps": col("gy"),
            "gz_radps": col("gz"),
            "a_forward_mps2": velocity.smoothed_forward_mps2,
            "a_forward_raw_mps2": direction.a_forward_mps2,
            "a_lateral_mps2": direction.a_lateral_mps2,
            "a_vertical_mps2": direction.a_vertical_mps2,
            "a_norm_mps2": direction.a_norm_mps2,
            "estimated_speed_mps": velocity.estimated_speed_mps,
            "estimated_distance_m": velocity.estimated_distance_m,
            "method_specific_1": direction.method_specific_1,
            "method_specific_2": direction.method_specific_2,
        }
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _raise_for_strict_warnings(config: RunConfig, warnings: list[str]) -> None:
    if not config.strict:
        return
    strict_warnings = [warning for warning in warnings if warning != ESTIMATED_WARNING]
    if strict_warnings:
        raise ProcessingError(
            "Strict mode failed due to warnings: " + "; ".join(strict_warnings)
        )
