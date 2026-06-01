"""Orientation conversion helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .config import RunConfig
from .errors import InputFormatError
from .schema import valid_euler_columns, valid_quaternion_columns


def rotation_from_dataframe(df: pd.DataFrame, config: RunConfig) -> tuple[Rotation, str, list[str]]:
    warnings: list[str] = []
    has_quaternion_columns = {"qw", "qx", "qy", "qz"}.issubset(df.columns)
    if has_quaternion_columns and valid_quaternion_columns(df):
        quat_xyzw = np.array(df[["qx", "qy", "qz", "qw"]], dtype=float, copy=True)
        return Rotation.from_quat(quat_xyzw), "quaternion", warnings

    if valid_euler_columns(df):
        if has_quaternion_columns:
            warnings.append("Invalid quaternion values were ignored; using Euler angles.")
        euler = np.array(df[["roll", "pitch", "yaw"]], dtype=float, copy=True)
        degrees = config.angle_unit == "deg"
        return Rotation.from_euler(config.euler_order, euler, degrees=degrees), "euler", warnings

    raise InputFormatError(
        "No valid orientation source: provide finite roll,pitch,yaw or valid non-zero quaternions."
    )


def course_vectors(course_yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.deg2rad(course_yaw_deg)
    forward = np.array([np.cos(theta), np.sin(theta), 0.0])
    lateral = np.array([-np.sin(theta), np.cos(theta), 0.0])
    vertical = np.array([0.0, 0.0, 1.0])
    return forward, lateral, vertical
