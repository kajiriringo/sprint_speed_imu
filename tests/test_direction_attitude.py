import numpy as np
import pytest

from sprint_speed_imu.config import RunConfig
from sprint_speed_imu.direction import compute_direction
from sprint_speed_imu.io import read_imu_csv
from sprint_speed_imu.preprocess import interpolate_numeric, slice_run


def test_attitude_known_yaw(synthetic_csv, tmp_path):
    config = RunConfig(
        input_path=synthetic_csv,
        output_dir=tmp_path / "out",
        method="attitude",
        distance_source="manual",
        distance_m=50,
    )
    df, _, _ = read_imu_csv(config)
    df = interpolate_numeric(df)
    run_df = slice_run(df, 1.0, 7.5)
    result = compute_direction(df, run_df, config, 1.0, 7.5)
    truth = run_df["true_forward_accel_mps2"].to_numpy()
    assert np.corrcoef(result.a_forward_mps2, truth)[0, 1] > 0.98
    assert result.diagnostics["rotation_source"] == "quaternion"
    assert result.diagnostics["yaw_std_deg"] == pytest.approx(0.0)
    assert result.diagnostics["world_static_residual_mps2"] == pytest.approx(0.0, abs=1e-9)
