import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from sprint_speed_imu.cli import main
from sprint_speed_imu.config import RunConfig
from sprint_speed_imu.direction import compute_direction
from sprint_speed_imu.io import read_imu_csv
from sprint_speed_imu.preprocess import interpolate_numeric, slice_run


def _heading_csv(path, sample_rate_hz=100.0):
    time = np.arange(0.0, 2.0 + 0.5 / sample_rate_hz, 1.0 / sample_rate_hz)
    heading_deg = 20.0 + 70.0 * time / time[-1]
    heading_rad = np.deg2rad(heading_deg)
    forward_accel = np.sin(2.0 * np.pi * time / time[-1])
    rotations = Rotation.from_euler(
        "xyz",
        np.column_stack([np.zeros_like(time), np.zeros_like(time), heading_deg]),
        degrees=True,
    )
    acc_world = np.column_stack(
        [
            forward_accel * np.cos(heading_rad),
            forward_accel * np.sin(heading_rad),
            np.full_like(time, 9.80665),
        ]
    )
    acc_sensor = rotations.inv().apply(acc_world)
    mag_sensor = rotations.inv().apply(np.tile(np.array([35.0, 0.0, 5.0]), (len(time), 1)))
    quat_xyzw = rotations.as_quat()
    df = pd.DataFrame(
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
            "yaw": heading_deg,
            "qw": quat_xyzw[:, 3],
            "qx": quat_xyzw[:, 0],
            "qy": quat_xyzw[:, 1],
            "qz": quat_xyzw[:, 2],
            "hx": mag_sensor[:, 0],
            "hy": mag_sensor[:, 1],
            "hz": mag_sensor[:, 2],
            "true_forward_accel_mps2": forward_accel,
        }
    )
    df.to_csv(path, index=False)
    return path


def test_heading_method_projects_acceleration_along_yaw(tmp_path):
    csv_path = _heading_csv(tmp_path / "heading.csv")
    config = RunConfig(
        input_path=csv_path,
        output_dir=tmp_path / "out",
        method="heading",
        distance_source="auto",
        smoothing_method="none",
    )
    df, _, _ = read_imu_csv(config)
    df = interpolate_numeric(df)
    run_df = slice_run(df, 0.0, 2.0)
    result = compute_direction(df, run_df, config, 0.0, 2.0)
    truth = run_df["true_forward_accel_mps2"].to_numpy()

    assert np.corrcoef(result.a_forward_mps2, truth)[0, 1] > 0.999
    assert result.heading_deg is not None
    assert result.diagnostics["heading_source"] == "yaw"
    assert result.diagnostics["magnetic_norm_cv"] == pytest.approx(0.0, abs=1e-12)


def test_cli_heading_auto_writes_trajectory(tmp_path):
    csv_path = _heading_csv(tmp_path / "heading.csv")
    out_dir = tmp_path / "out_heading"
    assert (
        main(
            [
                "--input",
                str(csv_path),
                "--method",
                "heading",
                "--distance-source",
                "auto",
                "--start-mode",
                "manual",
                "--start-time",
                "0",
                "--duration-s",
                "2",
                "--smoothing-method",
                "none",
                "--debug",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    trajectory = pd.read_csv(out_dir / "trajectory.csv")
    debug = pd.read_csv(out_dir / "debug_intermediate.csv")

    assert (out_dir / "trajectory.png").exists()
    assert summary["run"]["method"] == "heading"
    assert summary["diagnostics"]["heading_source"] == "yaw"
    assert summary["confidence"]["overall"] == "low"
    assert {"heading_deg", "estimated_east_m", "estimated_north_m"}.issubset(trajectory.columns)
    assert "heading_deg" in debug.columns
    assert any("without stride length or manual distance" in warning for warning in summary["warnings"])


def test_heading_method_can_use_magnetometer_without_orientation(tmp_path):
    csv_path = _heading_csv(tmp_path / "heading_mag.csv")
    df = pd.read_csv(csv_path).drop(columns=["roll", "pitch", "yaw", "qw", "qx", "qy", "qz"])
    df.to_csv(csv_path, index=False)
    config = RunConfig(
        input_path=csv_path,
        output_dir=tmp_path / "out_mag",
        method="heading",
        distance_source="auto",
        heading_source="magnetometer",
        smoothing_method="none",
    )
    loaded, _, _ = read_imu_csv(config)
    loaded = interpolate_numeric(loaded)
    run_df = slice_run(loaded, 0.0, 2.0)
    result = compute_direction(loaded, run_df, config, 0.0, 2.0)

    assert result.heading_deg is not None
    assert result.diagnostics["heading_source"] == "magnetometer"
    assert result.diagnostics["rotation_source"] == "accelerometer_magnetometer_tilt"
    assert np.all(np.isfinite(result.a_forward_mps2))
