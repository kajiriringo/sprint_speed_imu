import numpy as np

from sprint_speed_imu.config import RunConfig
from sprint_speed_imu.direction import compute_direction
from sprint_speed_imu.io import read_imu_csv
from sprint_speed_imu.preprocess import interpolate_numeric, slice_run


def test_pca_known_direction(synthetic_csv, tmp_path):
    config = RunConfig(
        input_path=synthetic_csv,
        output_dir=tmp_path / "out",
        method="pca",
        distance_source="manual",
        distance_m=50,
    )
    df, _, _ = read_imu_csv(config)
    df = interpolate_numeric(df)
    run_df = slice_run(df, 1.0, 7.5)
    result = compute_direction(df, run_df, config, 1.0, 7.5)
    truth = run_df["true_forward_accel_mps2"].to_numpy()
    assert result.diagnostics["pca_explained_variance_ratio"] > 0.95
    assert np.corrcoef(result.a_forward_mps2, truth)[0, 1] > 0.95
