from pathlib import Path

import pytest

from sprint_speed_imu.config import SyntheticConfig
from sprint_speed_imu.synthetic import write_synthetic_csv


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_run.csv"
    write_synthetic_csv(
        SyntheticConfig(
            output=path,
            distance_m=50.0,
            duration_s=6.5,
            sample_rate_hz=100.0,
            yaw_deg=30.0,
            noise_std_g=0.0,
            overwrite=True,
        )
    )
    return path
