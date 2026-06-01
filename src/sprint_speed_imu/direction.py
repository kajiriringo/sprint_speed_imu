"""Direction extraction for attitude and PCA methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RunConfig
from .errors import ProcessingError
from .orientation import course_vectors, rotation_from_dataframe


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


def compute_direction(
    full_df: pd.DataFrame,
    run_df: pd.DataFrame,
    config: RunConfig,
    start_time: float,
    end_time: float,
) -> DirectionResult:
    if config.method == "attitude":
        return _compute_attitude(full_df, run_df, config, start_time)
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
