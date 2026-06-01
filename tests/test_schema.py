import pandas as pd
import pytest

from sprint_speed_imu.errors import InputFormatError, MissingColumnError
from sprint_speed_imu.schema import validate_required_columns


def test_attitude_requires_orientation_columns():
    df = pd.DataFrame({"time_s": [0.0], "ax": [0.0], "ay": [0.0], "az": [9.8]})
    with pytest.raises(MissingColumnError):
        validate_required_columns(df, "attitude")


def test_pca_rejects_non_numeric_required_values():
    df = pd.DataFrame({"time_s": [0.0, 0.1], "ax": [float("nan"), float("nan")], "ay": [0.0, 0.0], "az": [9.8, 9.8]})
    with pytest.raises(InputFormatError):
        validate_required_columns(df, "pca")


def test_attitude_accepts_valid_euler_when_quaternion_invalid():
    df = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "ax": [0.0, 0.0],
            "ay": [0.0, 0.0],
            "az": [9.8, 9.8],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "qw": [float("nan"), float("nan")],
            "qx": [float("nan"), float("nan")],
            "qy": [float("nan"), float("nan")],
            "qz": [float("nan"), float("nan")],
        }
    )
    validate_required_columns(df, "attitude")


def test_attitude_rejects_invalid_orientation_values():
    df = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "ax": [0.0, 0.0],
            "ay": [0.0, 0.0],
            "az": [9.8, 9.8],
            "qw": [0.0, 0.0],
            "qx": [0.0, 0.0],
            "qy": [0.0, 0.0],
            "qz": [0.0, 0.0],
        }
    )
    with pytest.raises(InputFormatError):
        validate_required_columns(df, "attitude")
