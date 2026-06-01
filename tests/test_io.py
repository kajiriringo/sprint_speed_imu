import json

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
