"""Quality-control metrics, warnings, and confidence labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RunConfig
from .preprocess import estimate_sample_rate_hz


def acceleration_saturation_warning(df: pd.DataFrame, gravity: float) -> bool:
    acc_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
    if not acc_cols:
        return False
    threshold = gravity * 15.5
    values = df[acc_cols].to_numpy(dtype=float)
    return bool(np.nanmean(np.abs(values) >= threshold) > 0.001)


def build_qc(
    df: pd.DataFrame,
    estimated_speed_mps: np.ndarray,
    diagnostics: dict[str, object],
    velocity_diagnostics: dict[str, float | None],
    config: RunConfig,
) -> dict[str, object]:
    time = df["time_s"].to_numpy(dtype=float)
    negative_speed_ratio = float(np.mean(estimated_speed_mps < -1e-6))
    qc: dict[str, object] = {
        "sample_rate_est_hz": estimate_sample_rate_hz(time),
        "missing_ratio": float(df.attrs.get("missing_ratio", 0.0)),
        "negative_speed_ratio": negative_speed_ratio,
        "pca_explained_variance_ratio": diagnostics.get("pca_explained_variance_ratio"),
        "yaw_std_deg": diagnostics.get("yaw_std_deg"),
        "heading_source": diagnostics.get("heading_source"),
        "heading_range_deg": diagnostics.get("heading_range_deg"),
        "heading_rate_p95_deg_s": diagnostics.get("heading_rate_p95_deg_s"),
        "heading_lateral_energy_ratio": diagnostics.get("heading_lateral_energy_ratio"),
        "magnetic_norm_cv": diagnostics.get("magnetic_norm_cv"),
        "manual_distance_correction_ratio": velocity_diagnostics.get("manual_distance_correction_ratio"),
        "acceleration_saturation_warning": acceleration_saturation_warning(df, config.gravity),
        "baseline_gravity_norm_mps2": diagnostics.get("baseline_gravity_norm_mps2"),
    }
    return qc


def add_qc_warnings(qc: dict[str, object], config: RunConfig) -> list[str]:
    warnings: list[str] = []
    if qc["acceleration_saturation_warning"]:
        warnings.append("Acceleration saturation suspected near sensor range.")
    ratio = qc.get("manual_distance_correction_ratio")
    if isinstance(ratio, float) and abs(ratio) > config.max_correction_ratio:
        warnings.append("Manual distance correction is large; speed shape may be distorted.")
    if qc.get("negative_speed_ratio", 0.0) and float(qc["negative_speed_ratio"]) > 0.05:
        warnings.append("Negative speed ratio is high; integration may be unstable.")
    return warnings


def confidence_labels(config: RunConfig, diagnostics: dict[str, object], warnings: list[str]) -> dict[str, str]:
    if config.distance_source == "auto":
        return {
            "overall": "low",
            "distance_axis": "low",
            "speed_curve_shape": "low",
            "absolute_speed": "low",
        }

    shape = "medium"
    if config.method == "pca":
        ratio = diagnostics.get("pca_explained_variance_ratio")
        if isinstance(ratio, float) and ratio < config.min_pca_var_ratio:
            shape = "low"
    if config.method == "attitude":
        yaw_std = diagnostics.get("yaw_std_deg")
        if isinstance(yaw_std, float) and yaw_std > 30.0:
            shape = "low"
    if config.method == "heading":
        lateral_ratio = diagnostics.get("heading_lateral_energy_ratio")
        magnetic_cv = diagnostics.get("magnetic_norm_cv")
        if isinstance(lateral_ratio, float) and lateral_ratio > 1.0:
            shape = "low"
        if isinstance(magnetic_cv, float) and magnetic_cv > 0.20:
            shape = "low"

    overall = "medium" if shape != "low" else "low"
    if config.end_mode == "data-end" and config.duration_s is None:
        overall = "low_to_medium" if overall == "medium" else overall
    warning_text = " ".join(warning.lower() for warning in warnings)
    if "static gravity-removal residual" in warning_text or "peak forward acceleration" in warning_text:
        shape = "low"
        overall = "low"
    if any(
        keyword in warning_text
        for keyword in (
            "correction",
            "saturation",
            "missing values",
            "auto start detection failed",
            "sample interval jitter",
        )
    ):
        overall = "low"

    return {
        "overall": overall,
        "distance_axis": "high",
        "speed_curve_shape": shape,
        "absolute_speed": "low_to_medium",
    }
