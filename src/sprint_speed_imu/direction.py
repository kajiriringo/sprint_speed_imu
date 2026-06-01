"""Direction extraction for attitude, heading, and PCA methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .config import RunConfig
from .errors import ProcessingError
from .orientation import course_vectors, rotation_from_dataframe
from .preprocess import smooth_signal
from .schema import MAG_COLUMNS, valid_euler_columns, valid_magnetometer_columns, valid_quaternion_columns


@dataclass(slots=True)
class DirectionResult:
    run_df: pd.DataFrame
    a_forward_mps2: np.ndarray
    a_lateral_mps2: np.ndarray
    a_vertical_mps2: np.ndarray
    a_norm_mps2: np.ndarray
    diagnostics: dict[str, object]
    warnings: list[str]
    method_specific_1: np.ndarray
    method_specific_2: np.ndarray
    heading_deg: np.ndarray | None = None


def compute_direction(
    full_df: pd.DataFrame,
    run_df: pd.DataFrame,
    config: RunConfig,
    start_time: float,
    end_time: float,
) -> DirectionResult:
    if config.method == "attitude":
        return _compute_attitude(full_df, run_df, config, start_time)
    if config.method == "heading":
        return _compute_heading(full_df, run_df, config, start_time)
    if config.method == "pca":
        return _compute_pca(full_df, run_df, config, start_time, end_time)
    raise ProcessingError(f"Unknown method: {config.method}")


def _compute_attitude(
    full_df: pd.DataFrame,
    run_df: pd.DataFrame,
    config: RunConfig,
    start_time: float,
) -> DirectionResult:
    warnings: list[str] = []
    acc_sensor = np.array(run_df[["ax", "ay", "az"]], dtype=float, copy=True)
    rotation, source, orientation_warnings = rotation_from_dataframe(run_df, config)
    warnings.extend(orientation_warnings)
    acc_world = rotation.apply(acc_sensor)
    linear_world = acc_world - np.array([0.0, 0.0, config.gravity])

    forward, lateral, vertical = course_vectors(config.course_yaw_deg)
    a_forward = linear_world @ forward
    a_lateral = linear_world @ lateral
    a_vertical = linear_world @ vertical
    a_norm = np.linalg.norm(linear_world, axis=1)

    yaw_std_deg: float | None = None
    yaw_range_deg: float | None = None
    if "yaw" in run_df.columns:
        yaw = run_df["yaw"].to_numpy(dtype=float)
        if config.angle_unit == "rad":
            yaw = np.rad2deg(yaw)
        yaw_std_deg = float(np.nanstd(yaw))
        yaw_range_deg = float(np.nanmax(yaw) - np.nanmin(yaw))
        if yaw_std_deg > 30.0:
            warnings.append("Yaw is unstable; attitude direction confidence is low.")

    residual, residual_warnings = _attitude_static_residual(full_df, config, start_time)
    warnings.extend(residual_warnings)
    if residual is not None and residual > config.gravity * 0.5:
        warnings.append("Static gravity-removal residual is large for attitude method.")

    diagnostics: dict[str, object] = {
        "rotation_source": source,
        "euler_order": config.euler_order if source == "euler" else None,
        "angle_unit": config.angle_unit,
        "yaw_std_deg": yaw_std_deg,
        "yaw_range_deg": yaw_range_deg,
        "world_static_residual_mps2": residual,
    }
    method_specific_1 = np.full(len(run_df), yaw_std_deg if yaw_std_deg is not None else np.nan)
    method_specific_2 = np.full(len(run_df), residual if residual is not None else np.nan)
    return DirectionResult(
        run_df=run_df,
        a_forward_mps2=a_forward,
        a_lateral_mps2=a_lateral,
        a_vertical_mps2=a_vertical,
        a_norm_mps2=a_norm,
        diagnostics=diagnostics,
        warnings=warnings,
        method_specific_1=method_specific_1,
        method_specific_2=method_specific_2,
    )


def _compute_heading(
    full_df: pd.DataFrame,
    run_df: pd.DataFrame,
    config: RunConfig,
    start_time: float,
) -> DirectionResult:
    warnings: list[str] = []
    acc_sensor = np.array(run_df[["ax", "ay", "az"]], dtype=float, copy=True)
    rotation = None
    rotation_source: str | None = None
    if valid_quaternion_columns(run_df) or valid_euler_columns(run_df):
        rotation, rotation_source, orientation_warnings = rotation_from_dataframe(run_df, config)
        warnings.extend(orientation_warnings)
    else:
        warnings.append(
            "Heading method did not find trusted orientation columns; it may estimate tilt from acceleration."
        )

    heading_deg, heading_source, heading_aux, heading_warnings = _select_heading(run_df, config, rotation)
    warnings.extend(heading_warnings)
    if rotation is None and heading_source == "magnetometer":
        rotation = _rotation_from_acc_tilt_and_heading(run_df, heading_deg)
        rotation_source = "accelerometer_magnetometer_tilt"
    acc_world = rotation.apply(acc_sensor) if rotation is not None else acc_sensor.copy()
    linear_world = acc_world - _gravity_vector_for_heading(full_df, run_df, config, start_time, rotation)
    heading_rad = np.deg2rad(heading_deg)
    forward_vectors = np.column_stack([np.cos(heading_rad), np.sin(heading_rad), np.zeros(len(run_df))])
    lateral_vectors = np.column_stack([-np.sin(heading_rad), np.cos(heading_rad), np.zeros(len(run_df))])

    a_forward = np.sum(linear_world * forward_vectors, axis=1)
    a_lateral = np.sum(linear_world * lateral_vectors, axis=1)
    a_vertical = linear_world[:, 2]
    a_norm = np.linalg.norm(linear_world, axis=1)

    heading_unwrapped = _unwrap_degrees(heading_deg)
    heading_range_deg = float(np.nanmax(heading_unwrapped) - np.nanmin(heading_unwrapped))
    heading_rate_p95 = _heading_rate_p95_deg_s(run_df["time_s"].to_numpy(dtype=float), heading_unwrapped)
    lateral_energy_ratio = _energy_ratio(a_lateral, a_forward)
    if lateral_energy_ratio is not None and lateral_energy_ratio > 1.0:
        warnings.append(
            "Lateral acceleration energy exceeds forward-heading energy; heading may not match movement direction."
        )

    mag_norm_cv = _magnetic_norm_cv(run_df)
    if mag_norm_cv is not None and mag_norm_cv > 0.20:
        warnings.append("Magnetometer norm varies strongly; heading may be affected by magnetic disturbance.")

    if heading_source == "magnetometer":
        warnings.append(
            "Raw magnetometer heading is tilt-compensated from acceleration and remains low confidence during motion."
        )
    warnings.append(
        "Heading method integrates acceleration without stride length or manual distance; absolute speed can drift."
    )

    diagnostics: dict[str, object] = {
        "heading_source": heading_source,
        "rotation_source": rotation_source,
        "heading_offset_deg": config.heading_offset_deg,
        "heading_range_deg": heading_range_deg,
        "heading_rate_p95_deg_s": heading_rate_p95,
        "heading_lateral_energy_ratio": lateral_energy_ratio,
        "magnetic_norm_cv": mag_norm_cv,
        **heading_aux,
    }
    method_specific_2 = (
        _magnetic_norm(run_df) if valid_magnetometer_columns(run_df) else np.full(len(run_df), np.nan)
    )
    return DirectionResult(
        run_df=run_df,
        a_forward_mps2=a_forward,
        a_lateral_mps2=a_lateral,
        a_vertical_mps2=a_vertical,
        a_norm_mps2=a_norm,
        diagnostics=diagnostics,
        warnings=warnings,
        method_specific_1=heading_deg,
        method_specific_2=method_specific_2,
        heading_deg=heading_deg,
    )


def _gravity_vector_for_heading(
    full_df: pd.DataFrame,
    run_df: pd.DataFrame,
    config: RunConfig,
    start_time: float,
    rotation,
) -> np.ndarray:
    if rotation is not None:
        return np.array([0.0, 0.0, config.gravity])

    time = full_df["time_s"].to_numpy(dtype=float)
    baseline_mask = (time >= start_time - 1.0) & (time < start_time)
    if int(np.sum(baseline_mask)) < 3:
        baseline_mask = time <= time[0] + 1.0
    if int(np.sum(baseline_mask)) < 3:
        baseline_mask = np.arange(len(time)) < min(len(time), 10)
    return np.nanmedian(full_df.loc[baseline_mask, ["ax", "ay", "az"]].to_numpy(dtype=float), axis=0)


def _select_heading(
    run_df: pd.DataFrame,
    config: RunConfig,
    rotation,
) -> tuple[np.ndarray, str, dict[str, object], list[str]]:
    source = config.heading_source
    if source not in {"auto", "yaw", "quaternion", "magnetometer"}:
        raise ProcessingError("--heading-source must be auto, yaw, quaternion, or magnetometer.")

    warnings: list[str] = []
    aux: dict[str, object] = {}

    if source in {"auto", "yaw"} and "yaw" in run_df.columns:
        yaw = run_df["yaw"].to_numpy(dtype=float)
        if np.all(np.isfinite(yaw)):
            heading = yaw if config.angle_unit == "deg" else np.rad2deg(yaw)
            heading = _heading_with_offset(heading, config.heading_offset_deg)
            return heading, "yaw", aux, warnings
        if source == "yaw":
            raise ProcessingError("--heading-source=yaw requires finite yaw values.")

    if source in {"auto", "quaternion"} and rotation is not None:
        forward_axis = rotation.apply(np.tile(np.array([1.0, 0.0, 0.0]), (len(run_df), 1)))
        heading = np.rad2deg(np.arctan2(forward_axis[:, 1], forward_axis[:, 0]))
        heading = _heading_with_offset(heading, config.heading_offset_deg)
        return heading, "quaternion_forward_axis", aux, warnings
    if source == "quaternion":
        raise ProcessingError("--heading-source=quaternion requires valid quaternion or Euler orientation.")

    if source in {"auto", "magnetometer"} and valid_magnetometer_columns(run_df):
        heading = _tilt_compensated_magnetic_heading(run_df)
        heading = _heading_with_offset(heading, config.heading_offset_deg)
        aux["magnetometer_heading_note"] = "tilt_compensated_from_acceleration"
        return heading, "magnetometer", aux, warnings
    if source == "magnetometer":
        raise ProcessingError("--heading-source=magnetometer requires finite hx,hy,hz columns.")

    raise ProcessingError("method=heading could not resolve a heading source.")


def _heading_with_offset(heading_deg: np.ndarray, offset_deg: float) -> np.ndarray:
    return _wrap_degrees(np.asarray(heading_deg, dtype=float) + float(offset_deg))


def _tilt_compensated_magnetic_heading(run_df: pd.DataFrame) -> np.ndarray:
    mag = run_df[MAG_COLUMNS].to_numpy(dtype=float)
    roll, pitch = _tilt_from_acceleration(run_df)

    mx, my, mz = mag[:, 0], mag[:, 1], mag[:, 2]
    xh = mx * np.cos(pitch) + my * np.sin(roll) * np.sin(pitch) + mz * np.cos(roll) * np.sin(pitch)
    yh = my * np.cos(roll) - mz * np.sin(roll)
    return np.rad2deg(np.arctan2(-yh, xh))


def _rotation_from_acc_tilt_and_heading(run_df: pd.DataFrame, heading_deg: np.ndarray) -> Rotation:
    roll, pitch = _tilt_from_acceleration(run_df)
    return Rotation.from_euler(
        "xyz",
        np.column_stack([roll, pitch, np.deg2rad(heading_deg)]),
        degrees=False,
    )


def _tilt_from_acceleration(run_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    acc = run_df[["ax", "ay", "az"]].to_numpy(dtype=float)
    time = run_df["time_s"].to_numpy(dtype=float)
    acc_lp = np.column_stack(
        [
            smooth_signal(
                acc[:, idx],
                time,
                enabled=True,
                method="kalman",
                kalman_process_noise=0.01,
                kalman_measurement_noise=1.0,
            )
            for idx in range(3)
        ]
    )
    ax, ay, az = acc_lp[:, 0], acc_lp[:, 1], acc_lp[:, 2]
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, ay * np.sin(roll) + az * np.cos(roll))
    return roll, pitch


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def _unwrap_degrees(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(values, dtype=float))))


def _heading_rate_p95_deg_s(time: np.ndarray, heading_unwrapped_deg: np.ndarray) -> float | None:
    if len(time) < 2:
        return None
    dt = np.diff(time)
    valid = dt > 1e-9
    if not np.any(valid):
        return None
    rate = np.abs(np.diff(heading_unwrapped_deg)[valid] / dt[valid])
    return float(np.nanpercentile(rate, 95)) if len(rate) else None


def _energy_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    den = float(np.nansum(np.asarray(denominator, dtype=float) ** 2))
    if den <= 1e-12:
        return None
    return float(np.nansum(np.asarray(numerator, dtype=float) ** 2) / den)


def _magnetic_norm(run_df: pd.DataFrame) -> np.ndarray:
    return np.linalg.norm(run_df[MAG_COLUMNS].to_numpy(dtype=float), axis=1)


def _magnetic_norm_cv(run_df: pd.DataFrame) -> float | None:
    if not valid_magnetometer_columns(run_df):
        return None
    norm = _magnetic_norm(run_df)
    med = float(np.nanmedian(norm))
    if med <= 1e-9:
        return None
    return float(np.nanstd(norm) / med)


def _attitude_static_residual(
    full_df: pd.DataFrame,
    config: RunConfig,
    start_time: float,
) -> tuple[float | None, list[str]]:
    warnings: list[str] = []
    time = full_df["time_s"].to_numpy(dtype=float)
    baseline_mask = (time >= start_time - 1.0) & (time < start_time)
    if int(np.sum(baseline_mask)) < 3:
        baseline_mask = time <= time[0] + 1.0
        warnings.append("Attitude static residual used first 1 second of data.")
    baseline_df = full_df.loc[baseline_mask]
    if len(baseline_df) < 3:
        return None, warnings
    rotation, _, orientation_warnings = rotation_from_dataframe(baseline_df, config)
    warnings.extend(orientation_warnings)
    acc_sensor = np.array(baseline_df[["ax", "ay", "az"]], dtype=float, copy=True)
    linear_world = rotation.apply(acc_sensor) - np.array([0.0, 0.0, config.gravity])
    return float(np.nanmedian(np.linalg.norm(linear_world, axis=1))), warnings


def _pca_window_mask(time: np.ndarray, start_time: float, end_time: float, mode: str) -> np.ndarray:
    if mode == "run":
        return (time >= start_time) & (time <= end_time)
    if mode == "first-2s":
        return (time >= start_time) & (time <= min(end_time, start_time + 2.0))
    if mode == "first-half":
        return (time >= start_time) & (time <= start_time + (end_time - start_time) / 2.0)
    raise ProcessingError("--pca-window must be run, first-2s, or first-half.")


def _unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9 or not np.isfinite(norm):
        return fallback.astype(float)
    return vector / norm


def _compute_pca(
    full_df: pd.DataFrame,
    run_df: pd.DataFrame,
    config: RunConfig,
    start_time: float,
    end_time: float,
) -> DirectionResult:
    warnings: list[str] = []
    time_full = full_df["time_s"].to_numpy(dtype=float)
    acc_full = full_df[["ax", "ay", "az"]].to_numpy(dtype=float)
    time_run = run_df["time_s"].to_numpy(dtype=float)
    run_index = run_df.index.to_numpy()

    baseline_mask = (time_full >= start_time - 1.0) & (time_full < start_time)
    if int(np.sum(baseline_mask)) < 3:
        baseline_mask = time_full <= time_full[0] + 1.0
        warnings.append("PCA gravity baseline used first 1 second of data.")
    if int(np.sum(baseline_mask)) < 3:
        baseline_mask = np.arange(len(time_full)) < min(len(time_full), 10)
        warnings.append("PCA gravity baseline is short.")

    g_est = np.nanmedian(acc_full[baseline_mask], axis=0)
    g_hat = _unit_vector(g_est, np.array([0.0, 0.0, 1.0]))
    a_linear_full = acc_full - g_est
    vertical_full = a_linear_full @ g_hat
    a_horizontal_full = a_linear_full - np.outer(vertical_full, g_hat)

    window_mask = _pca_window_mask(time_full, start_time, end_time, config.pca_window)
    X = a_horizontal_full[window_mask]
    if len(X) < 3:
        raise ProcessingError("PCA window has too few samples.")
    X_centered = X - np.nanmean(X, axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total = float(np.sum(np.maximum(eigvals, 0.0)))
    ratio = float(eigvals[0] / total) if total > 0 else 0.0
    pc1 = _unit_vector(eigvecs[:, 0], np.array([1.0, 0.0, 0.0]))
    pc2 = _unit_vector(eigvecs[:, 1] if eigvecs.shape[1] > 1 else np.cross(g_hat, pc1), np.array([0.0, 1.0, 0.0]))

    start_mask = (time_full >= start_time) & (time_full <= min(end_time, start_time + 0.5))
    sign_probe = float(np.nanmean(a_horizontal_full[start_mask] @ pc1)) if np.any(start_mask) else 0.0
    sign_flip = sign_probe < 0
    if sign_flip:
        pc1 = -pc1

    a_horizontal_run = a_horizontal_full[run_index]
    a_linear_run = a_linear_full[run_index]
    a_forward = a_horizontal_run @ pc1
    a_lateral = a_horizontal_run @ pc2
    a_vertical = a_linear_run @ g_hat
    a_norm = np.linalg.norm(a_linear_run, axis=1)
    horizontal_energy = float(np.sum(a_horizontal_run**2))
    total_energy = float(np.sum(a_linear_run**2))
    horizontal_energy_ratio = horizontal_energy / total_energy if total_energy > 0 else None

    if ratio < config.min_pca_var_ratio:
        message = (
            f"PCA explained variance ratio {ratio:.3f} is below "
            f"threshold {config.min_pca_var_ratio:.3f}."
        )
        if config.strict:
            raise ProcessingError("PCA direction confidence is too low in --strict mode.")
        warnings.append(message)

    diagnostics: dict[str, object] = {
        "pca_explained_variance_ratio": ratio,
        "pca_window": config.pca_window,
        "pc1_vector": [float(v) for v in pc1],
        "pc2_vector": [float(v) for v in pc2],
        "sign_flip_applied": bool(sign_flip),
        "baseline_gravity_norm_mps2": float(np.linalg.norm(g_est)),
        "horizontal_energy_ratio": horizontal_energy_ratio,
    }
    method_specific_1 = np.full(len(time_run), ratio)
    method_specific_2 = np.full(len(time_run), horizontal_energy_ratio if horizontal_energy_ratio is not None else np.nan)
    return DirectionResult(
        run_df=run_df,
        a_forward_mps2=a_forward,
        a_lateral_mps2=a_lateral,
        a_vertical_mps2=a_vertical,
        a_norm_mps2=a_norm,
        diagnostics=diagnostics,
        warnings=warnings,
        method_specific_1=method_specific_1,
        method_specific_2=method_specific_2,
    )
