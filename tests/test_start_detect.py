import pytest

from sprint_speed_imu.config import RunConfig
from sprint_speed_imu.io import read_imu_csv
from sprint_speed_imu.preprocess import interpolate_numeric
from sprint_speed_imu.start_detect import detect_start_time


def test_start_detect_synthetic(synthetic_csv, tmp_path):
    config = RunConfig(input_path=synthetic_csv, output_dir=tmp_path / "out", method="pca", distance_source="manual", distance_m=50)
    df, _, _ = read_imu_csv(config)
    df = interpolate_numeric(df)
    start, warnings = detect_start_time(df, config)
    assert warnings == []
    assert start == pytest.approx(1.0, abs=0.08)
