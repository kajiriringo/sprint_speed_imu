import json
import math

import pandas as pd
import pytest

from sprint_speed_imu.config import RunConfig
from sprint_speed_imu.io import read_imu_csv


def test_io_standard_csv(tmp_path):
    path = tmp_path / "run.csv"
    pd.DataFrame({"time_s": [0, 0.01], "ax": [1, 1], "ay": [0, 0], "az": [0, 0]}).to_csv(path, index=False)
    df, meta, warnings = read_imu_csv(
        RunConfig(input_path=path, output_dir=tmp_path / "out", method="pca", distance_source="auto")
    )
    assert meta["rows"] == 2
    assert warnings == []
    assert df["ax"].iloc[0] == pytest.approx(9.80665)


def test_io_column_map(tmp_path):
    path = tmp_path / "wit.csv"
    pd.DataFrame({"Time": [0, 0.01], "AccX": [0, 0], "AccY": [0, 0], "AccZ": [1, 1]}).to_csv(path, index=False)
    column_map = tmp_path / "column_map.json"
    column_map.write_text(
        json.dumps({"time_s": "Time", "ax": "AccX", "ay": "AccY", "az": "AccZ"}),
        encoding="utf-8",
    )
    df, _, _ = read_imu_csv(
        RunConfig(
            input_path=path,
            output_dir=tmp_path / "out",
            method="pca",
            distance_source="auto",
            column_map=column_map,
        )
    )
    assert {"time_s", "ax", "ay", "az"}.issubset(df.columns)
    assert df["az"].iloc[0] == pytest.approx(9.80665)


def test_io_wt901ble_tsv_with_units_and_iso_time(tmp_path):
    degree = "\N{DEGREE SIGN}"
    path = tmp_path / "wt901ble.tsv"
    pd.DataFrame(
        {
            "time": ["2026-06-01T19:09:30.876", "2026-06-01T19:09:30.936"],
            "DeviceName": ["WT901BLE67(39CA996AE3DE)", "WT901BLE67(39CA996AE3DE)"],
            "AccX(g)": [0.002, 0.003],
            "AccY(g)": [0.996, 0.995],
            "AccZ(g)": [0.041, 0.040],
            f"AsX({degree}/s)": [0.000, 0.061],
            f"AsY({degree}/s)": [0.183, 0.122],
            f"AsZ({degree}/s)": [-0.061, -0.061],
            f"AngleX({degree})": [88.30, 88.29],
            f"AngleY({degree})": [-0.26, -0.26],
            f"AngleZ({degree})": [29.78, 29.78],
            "Q0()": [0.69696, 0.69699],
            "Q1()": [0.66940, 0.66940],
            "Q2()": [0.17719, 0.17715],
            "Q3()": [0.18610, 0.18610],
        }
    ).to_csv(path, index=False, sep="\t")

    df, meta, warnings = read_imu_csv(
        RunConfig(
            input_path=path,
            output_dir=tmp_path / "out",
            method="attitude",
            distance_source="auto",
        )
    )

    assert meta["rows"] == 2
    assert {"time_s", "ax", "ay", "az", "gx", "gy", "gz"}.issubset(df.columns)
    assert {"roll", "pitch", "yaw", "qw", "qx", "qy", "qz"}.issubset(df.columns)
    assert df["time_s"].tolist() == pytest.approx([0.0, 0.060])
    assert df["ax"].iloc[0] == pytest.approx(0.002 * 9.80665)
    assert df["gy"].iloc[0] == pytest.approx(math.radians(0.183))
    assert df["roll"].iloc[0] == pytest.approx(88.30)
    assert df["qw"].iloc[0] == pytest.approx(0.69696)
    assert warnings == ["time_s was generated from datetime timestamps in the time column."]
