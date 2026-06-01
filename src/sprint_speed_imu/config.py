"""Configuration dataclasses shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunConfig:
    input_path: Path
    output_dir: Path
    method: str
    distance_source: str
    distance_m: float | None = None
    acc_unit: str = "g"
    gyro_unit: str = "dps"
    angle_unit: str = "deg"
    time_column: str = "time_s"
    sample_rate_hz: float | None = None
    column_map: Path | None = None
    start_mode: str = "auto"
    start_time: float | None = None
    end_mode: str = "data-end"
    end_time: float | None = None
    duration_s: float | None = None
    distance_bin_m: float = 1.0
    smooth: bool = True
    smoothing_method: str = "savgol"
    kalman_process_noise: float = 0.05
    kalman_measurement_noise: float = 0.5
    course_yaw_deg: float = 0.0
    euler_order: str = "xyz"
    gravity: float = 9.80665
    gravity_mode: str = "auto"
    pca_window: str = "first-half"
    min_pca_var_ratio: float = 0.45
    max_correction_ratio: float = 2.0
    correction_mode: str = "mean-speed-shape"
    debug: bool = False
    strict: bool = False
    overwrite: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("input_path", "output_dir", "column_map"):
            if data[key] is not None:
                data[key] = str(data[key])
        return data


@dataclass(slots=True)
class SyntheticConfig:
    output: Path
    distance_m: float = 50.0
    duration_s: float = 6.5
    sample_rate_hz: float = 100.0
    yaw_deg: float = 30.0
    noise_std_g: float = 0.01
    pre_start_s: float = 1.0
    seed: int = 7
    overwrite: bool = False
