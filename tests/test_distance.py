import numpy as np

from sprint_speed_imu.distance import build_speed_curve


def test_distance_bins_include_final_distance():
    curve = build_speed_curve(
        time_s=np.array([0.0, 1.0, 2.0]),
        distance_m=np.array([0.0, 5.0, 10.0]),
        estimated_speed_mps=np.array([0.0, 5.0, 5.0]),
        a_forward_mps2=np.array([1.0, 0.0, -1.0]),
        distance_bin_m=2.0,
        confidence_flag="medium",
    )
    assert curve["distance_m"].iloc[0] == 0.0
    assert curve["distance_m"].iloc[-1] == 10.0
    assert set(curve["confidence_flag"]) == {"medium"}


def test_distance_bins_never_exceed_final_distance():
    curve = build_speed_curve(
        time_s=np.array([0.0, 1.0]),
        distance_m=np.array([0.0, 10.0]),
        estimated_speed_mps=np.array([0.0, 1.0]),
        a_forward_mps2=np.array([0.0, 0.0]),
        distance_bin_m=6.0,
        confidence_flag="medium",
    )
    assert curve["distance_m"].tolist() == [0.0, 6.0, 10.0]
