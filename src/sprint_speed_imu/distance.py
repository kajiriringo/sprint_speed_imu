"""Distance-bin interpolation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_speed_curve(
    time_s: np.ndarray,
    distance_m: np.ndarray,
    estimated_speed_mps: np.ndarray,
    a_forward_mps2: np.ndarray,
    distance_bin_m: float,
    confidence_flag: str,
) -> pd.DataFrame:
    if distance_bin_m <= 0:
        raise ValueError("distance_bin_m must be positive.")
    final_distance = float(np.nanmax(distance_m)) if len(distance_m) else 0.0
    if final_distance <= 0:
        bins = np.array([0.0])
    else:
        bins = np.arange(0.0, final_distance + 1e-9, distance_bin_m)
        bins = bins[bins <= final_distance + 1e-9]
        if len(bins) == 0 or bins[0] != 0.0:
            bins = np.insert(bins, 0, 0.0)
        if abs(float(bins[-1]) - final_distance) > 1e-9:
            bins = np.append(bins, final_distance)
        else:
            bins[-1] = final_distance
    order = np.argsort(distance_m)
    x = np.maximum.accumulate(distance_m[order])
    keep = np.r_[True, np.diff(x) > 1e-9]
    x_unique = x[keep]
    if len(x_unique) == 1:
        interp_time = np.full_like(bins, float(time_s[0]), dtype=float)
        interp_speed = np.full_like(bins, float(estimated_speed_mps[0]), dtype=float)
        interp_acc = np.full_like(bins, float(a_forward_mps2[0]), dtype=float)
    else:
        interp_time = np.interp(bins, x_unique, time_s[order][keep])
        interp_speed = np.interp(bins, x_unique, estimated_speed_mps[order][keep])
        interp_acc = np.interp(bins, x_unique, a_forward_mps2[order][keep])
    return pd.DataFrame(
        {
            "distance_m": bins,
            "time_s": interp_time,
            "estimated_speed_mps": interp_speed,
            "a_forward_mps2": interp_acc,
            "confidence_flag": confidence_flag,
        }
    )
