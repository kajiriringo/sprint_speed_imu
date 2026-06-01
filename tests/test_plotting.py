import pandas as pd

from sprint_speed_imu.plotting import plot_speed_curve


def test_plotting_writes_png(tmp_path):
    curve = pd.DataFrame(
        {
            "distance_m": [0.0, 1.0, 2.0],
            "estimated_speed_mps": [0.0, 1.0, 0.5],
        }
    )
    summary = {
        "run": {"method": "pca", "distance_source": "manual", "distance_m": 2.0},
        "results": {"max_estimated_speed_distance_m": 1.0, "max_estimated_speed_mps": 1.0},
        "confidence": {"overall": "medium"},
    }
    output = tmp_path / "curve.png"
    plot_speed_curve(curve, summary, output)
    assert output.exists()
    assert output.stat().st_size > 0
