import numpy as np
import pytest

from sprint_speed_imu.config import RunConfig
from sprint_speed_imu.velocity import estimate_velocity


def test_manual_distance_final_distance(tmp_path):
    time = np.linspace(0.0, 6.5, 651)
    speed_shape = 1.0 - np.exp(-time / 1.8)
    speed_shape *= 50.0 / np.trapezoid(speed_shape, time)
    accel = np.gradient(speed_shape, time)
    config = RunConfig(
        input_path=tmp_path / "in.csv",
        output_dir=tmp_path / "out",
        method="pca",
        distance_source="manual",
        distance_m=50.0,
        smoothing_method="none",
    )
    result = estimate_velocity(time, accel, config)
    assert result.estimated_distance_m[-1] == pytest.approx(50.0, rel=1e-3)


def test_auto_distance_low_confidence(tmp_path):
    time = np.linspace(0.0, 2.0, 201)
    accel = np.ones_like(time)
    config = RunConfig(
        input_path=tmp_path / "in.csv",
        output_dir=tmp_path / "out",
        method="pca",
        distance_source="auto",
    )
    result = estimate_velocity(time, accel, config)
    assert any("low confidence" in warning for warning in result.warnings)
