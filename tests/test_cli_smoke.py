import json

import pandas as pd
import pytest

from sprint_speed_imu.cli import main


def test_cli_smoke_pca_manual(tmp_path):
    csv_path = tmp_path / "synthetic.csv"
    out_dir = tmp_path / "out_pca"
    assert main(["synthetic", "--distance-m", "50", "--duration-s", "6.5", "--output", str(csv_path)]) == 0
    assert main(
        [
            "--input",
            str(csv_path),
            "--method",
            "pca",
            "--distance-source",
            "manual",
            "--distance-m",
            "50",
            "--output-dir",
            str(out_dir),
            "--debug",
        ]
    ) == 0
    for name in ["speed_curve.png", "speed_curve.csv", "summary.json", "run_config.json", "debug_intermediate.csv"]:
        assert (out_dir / name).exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run"]["method"] == "pca"
    assert "warnings" in summary
    assert "confidence" in summary
    assert "qc" in summary


def test_cli_smoke_attitude_manual(synthetic_csv, tmp_path):
    out_dir = tmp_path / "out_attitude"
    assert main(
        [
            "--input",
            str(synthetic_csv),
            "--method",
            "attitude",
            "--distance-source",
            "manual",
            "--distance-m",
            "50",
            "--output-dir",
            str(out_dir),
        ]
    ) == 0
    assert (out_dir / "speed_curve.png").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run"]["method"] == "attitude"


def test_compare_mode(synthetic_csv, tmp_path):
    out_dir = tmp_path / "compare"
    assert main(
        [
            "compare",
            "--input",
            str(synthetic_csv),
            "--distance-source",
            "manual",
            "--distance-m",
            "50",
            "--output-dir",
            str(out_dir),
        ]
    ) == 0
    assert (out_dir / "summary_compare.json").exists()
    assert (out_dir / "speed_curve_compare.png").exists()


def test_cli_kalman_smoothing_records_config(synthetic_csv, tmp_path):
    out_dir = tmp_path / "out_kalman"
    assert main(
        [
            "--input",
            str(synthetic_csv),
            "--method",
            "pca",
            "--distance-source",
            "manual",
            "--distance-m",
            "50",
            "--smoothing-method",
            "kalman",
            "--kalman-process-noise",
            "0.04",
            "--kalman-measurement-noise",
            "0.6",
            "--output-dir",
            str(out_dir),
        ]
    ) == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    run_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert summary["run"]["smoothing_method"] == "kalman"
    assert summary["run"]["kalman_process_noise"] == 0.04
    assert summary["run"]["kalman_measurement_noise"] == 0.6
    assert run_config["smoothing_method"] == "kalman"
    assert "forward_accel_peak_retention_ratio" in summary["velocity_diagnostics"]


def test_overwrite_removes_stale_debug_artifact(synthetic_csv, tmp_path):
    out_dir = tmp_path / "out_overwrite"
    base_args = [
        "--input",
        str(synthetic_csv),
        "--method",
        "pca",
        "--distance-source",
        "manual",
        "--distance-m",
        "50",
        "--output-dir",
        str(out_dir),
    ]
    assert main([*base_args, "--debug"]) == 0
    assert (out_dir / "debug_intermediate.csv").exists()
    (out_dir / "custom.txt").write_text("stale", encoding="utf-8")
    assert main([*base_args, "--overwrite"]) == 0
    assert not (out_dir / "debug_intermediate.csv").exists()
    assert not (out_dir / "custom.txt").exists()


def test_cli_rejects_non_numeric_required_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "time_s": [i / 10 for i in range(31)],
            "ax": ["bad"] * 31,
            "ay": [0.0] * 31,
            "az": [1.0] * 31,
        }
    ).to_csv(csv_path, index=False)
    assert (
        main(
            [
                "--input",
                str(csv_path),
                "--method",
                "pca",
                "--distance-source",
                "manual",
                "--distance-m",
                "10",
                "--output-dir",
                str(tmp_path / "out_bad"),
            ]
        )
        == 2
    )


def test_cli_attitude_falls_back_to_euler_when_quaternion_invalid(synthetic_csv, tmp_path):
    csv_path = tmp_path / "euler_fallback.csv"
    df = pd.read_csv(synthetic_csv)
    for column in ["qw", "qx", "qy", "qz"]:
        df[column] = float("nan")
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "out_euler"
    assert (
        main(
            [
                "--input",
                str(csv_path),
                "--method",
                "attitude",
                "--distance-source",
                "manual",
                "--distance-m",
                "50",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["diagnostics"]["rotation_source"] == "euler"
    assert any("Invalid quaternion" in warning for warning in summary["warnings"])


def test_compare_preflight_fails_without_partial_outputs(synthetic_csv, tmp_path):
    csv_path = tmp_path / "pca_only.csv"
    df = pd.read_csv(synthetic_csv)
    df = df.drop(columns=["roll", "pitch", "yaw", "qw", "qx", "qy", "qz"])
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "compare_bad"
    assert (
        main(
            [
                "compare",
                "--input",
                str(csv_path),
                "--distance-source",
                "manual",
                "--distance-m",
                "50",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    assert not (out_dir / "pca" / "summary.json").exists()


def test_compare_preflight_runs_direction_without_partial_outputs(synthetic_csv, tmp_path):
    csv_path = tmp_path / "euler_only.csv"
    df = pd.read_csv(synthetic_csv).drop(columns=["qw", "qx", "qy", "qz"])
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "compare_bad_euler_order"
    assert (
        main(
            [
                "compare",
                "--input",
                str(csv_path),
                "--distance-source",
                "manual",
                "--distance-m",
                "50",
                "--euler-order",
                "bad",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    assert not out_dir.exists()


def test_overwrite_removes_compare_subdirectories(synthetic_csv, tmp_path):
    out_dir = tmp_path / "mixed_out"
    assert (
        main(
            [
                "compare",
                "--input",
                str(synthetic_csv),
                "--distance-source",
                "manual",
                "--distance-m",
                "50",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (out_dir / "pca" / "summary.json").exists()
    assert (
        main(
            [
                "--input",
                str(synthetic_csv),
                "--method",
                "pca",
                "--distance-source",
                "manual",
                "--distance-m",
                "50",
                "--output-dir",
                str(out_dir),
                "--overwrite",
            ]
        )
        == 0
    )
    assert not (out_dir / "pca").exists()
    assert not (out_dir / "attitude").exists()
    assert (out_dir / "summary.json").exists()


def test_cli_rejects_invalid_kalman_noise(synthetic_csv, tmp_path):
    assert (
        main(
            [
                "--input",
                str(synthetic_csv),
                "--method",
                "pca",
                "--distance-source",
                "manual",
                "--distance-m",
                "50",
                "--smoothing-method",
                "kalman",
                "--kalman-process-noise",
                "0",
                "--output-dir",
                str(tmp_path / "bad_kalman"),
            ]
        )
        == 2
    )


def test_raw_integration_manual_scales_to_distance(synthetic_csv, tmp_path):
    out_dir = tmp_path / "raw_manual"
    assert (
        main(
            [
                "--input",
                str(synthetic_csv),
                "--method",
                "pca",
                "--distance-source",
                "manual",
                "--distance-m",
                "50",
                "--correction-mode",
                "raw-integration",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(out_dir / "speed_curve.csv")
    assert summary["results"]["final_estimated_distance_m"] == pytest.approx(50.0)
    assert curve["distance_m"].max() == pytest.approx(50.0)


def test_strict_fails_on_data_end_warning(synthetic_csv, tmp_path):
    assert (
        main(
            [
                "--input",
                str(synthetic_csv),
                "--method",
                "pca",
                "--distance-source",
                "manual",
                "--distance-m",
                "50",
                "--strict",
                "--output-dir",
                str(tmp_path / "strict_fail"),
            ]
        )
        == 2
    )


def test_strict_allows_estimated_notice_only(synthetic_csv, tmp_path):
    out_dir = tmp_path / "strict_pass"
    assert (
        main(
            [
                "--input",
                str(synthetic_csv),
                "--method",
                "pca",
                "--distance-source",
                "manual",
                "--distance-m",
                "50",
                "--start-mode",
                "manual",
                "--start-time",
                "1.0",
                "--duration-s",
                "6.5",
                "--strict",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["warnings"] == ["Speed, distance, and max-speed location are estimated from IMU data."]
