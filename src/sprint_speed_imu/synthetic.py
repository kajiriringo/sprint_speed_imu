"""Deterministic synthetic sprint data generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .config import SyntheticConfig


def _cumtrapz(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    out[1:] = np.cumsum((values[:-1] + values[1:]) * 0.5 * np.diff(time_s))
    return out


def generate_synthetic_run(config: SyntheticConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    dt = 1.0 / config.sample_rate_hz
    total_s = config.pre_start_s + config.duration_s
    time = np.arange(0.0, total_s + dt * 0.5, dt)
    run_t = np.clip(time - config.pre_start_s, 0.0, config.duration_s)

    tau = max(config.duration_s * 0.28, 0.5)
    speed_shape = (1.0 - np.exp(-run_t / tau)) * (1.0 - 0.04 * (run_t / config.duration_s) ** 2)
    speed_shape[time < config.pre_start_s] = 0.0
    area = float(np.trapezoid(speed_shape, time))
    speed = speed_shape * (config.distance_m / area)
    speed[time > config.pre_start_s + config.duration_s] = 0.0
    distance = _cumtrapz(speed, time)
    accel_forward = np.gradient(speed, time)
    accel_forward[time < config.pre_start_s] = 0.0

    rotation = Rotation.from_euler("xyz", [0.0, 0.0, config.yaw_deg], degrees=True)
    quat_xyzw = rotation.as_quat()
    gravity_world = np.array([0.0, 0.0, 9.80665])
    acc_world = np.column_stack([accel_forward, np.zeros_like(time), np.zeros_like(time)]) + gravity_world
    acc_sensor = rotation.inv().apply(acc_world)
    if config.noise_std_g > 0:
        acc_sensor = acc_sensor + rng.normal(0.0, config.noise_std_g * 9.80665, size=acc_sensor.shape)

    return pd.DataFrame(
        {
            "time_s": time,
            "ax": acc_sensor[:, 0] / 9.80665,
            "ay": acc_sensor[:, 1] / 9.80665,
            "az": acc_sensor[:, 2] / 9.80665,
            "gx": np.zeros_like(time),
            "gy": np.zeros_like(time),
            "gz": np.zeros_like(time),
            "roll": np.zeros_like(time),
            "pitch": np.zeros_like(time),
            "yaw": np.full_like(time, config.yaw_deg),
            "qw": np.full_like(time, quat_xyzw[3]),
            "qx": np.full_like(time, quat_xyzw[0]),
            "qy": np.full_like(time, quat_xyzw[1]),
            "qz": np.full_like(time, quat_xyzw[2]),
            "true_speed_mps": speed,
            "true_distance_m": distance,
            "true_forward_accel_mps2": accel_forward,
        }
    )


def write_synthetic_csv(config: SyntheticConfig) -> Path:
    if config.output.exists() and not config.overwrite:
        raise FileExistsError(f"Output file already exists: {config.output}. Use --overwrite.")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_run(config)
    df.to_csv(config.output, index=False)
    return config.output
