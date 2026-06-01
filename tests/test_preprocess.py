import numpy as np
import pandas as pd
import pytest

from sprint_speed_imu.errors import InputFormatError
from sprint_speed_imu.preprocess import interpolate_numeric, kalman_rts_smooth_1d, validate_time_axis


def test_interpolate_numeric():
    df = pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "ax": [1.0, np.nan, 3.0]})
    out = interpolate_numeric(df)
    assert out["ax"].iloc[1] == pytest.approx(2.0)


def test_time_axis_must_increase():
    df = pd.DataFrame({"time_s": [0.0, 0.0, 1.0]})
    with pytest.raises(InputFormatError):
        validate_time_axis(df)


def test_kalman_rts_smooth_reduces_noise():
    rng = np.random.default_rng(42)
    time = np.linspace(0.0, 4.0, 401)
    truth = np.sin(time * 2.0)
    noisy = truth + rng.normal(0.0, 0.25, size=len(time))
    smoothed = kalman_rts_smooth_1d(noisy, time, process_noise=0.02, measurement_noise=0.5)
    assert np.std(smoothed - truth) < np.std(noisy - truth)
