"""Plot output for estimated speed curves."""

from __future__ import annotations

from pathlib import Path

from .mpl_setup import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_speed_curve(curve: pd.DataFrame, summary: dict[str, object], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve["distance_m"], curve["estimated_speed_mps"], color="#1f77b4", linewidth=2.2)
    ax.set_title("Estimated Sprint Speed Curve")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Estimated Speed (m/s)")
    ax.grid(True, alpha=0.25)

    results = summary["results"]
    run = summary["run"]
    confidence = summary["confidence"]
    max_x = results.get("max_estimated_speed_distance_m")
    max_y = results.get("max_estimated_speed_mps")
    if max_x is not None and max_y is not None:
        ax.scatter([max_x], [max_y], color="#d62728", zorder=4)
        ax.annotate(
            f"max est. {max_y:.2f} m/s",
            xy=(max_x, max_y),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            color="#d62728",
        )

    distance_label = (
        f"manual {run.get('distance_m'):.1f} m"
        if run.get("distance_source") == "manual" and run.get("distance_m") is not None
        else "auto estimated"
    )
    note = (
        f"Method: {run.get('method')}  Distance: {distance_label}  "
        f"Confidence: {confidence.get('overall')}\n"
        "Estimated from IMU. Not official timing."
    )
    ax.text(
        0.01,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_trajectory(trajectory: pd.DataFrame, summary: dict[str, object], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        trajectory["estimated_east_m"],
        trajectory["estimated_north_m"],
        color="#2ca02c",
        linewidth=2.0,
    )
    ax.scatter(
        [trajectory["estimated_east_m"].iloc[0]],
        [trajectory["estimated_north_m"].iloc[0]],
        color="#1f77b4",
        label="start",
        zorder=3,
    )
    ax.scatter(
        [trajectory["estimated_east_m"].iloc[-1]],
        [trajectory["estimated_north_m"].iloc[-1]],
        color="#d62728",
        label="end",
        zorder=3,
    )
    ax.set_title("Estimated Heading Trajectory")
    ax.set_xlabel("Estimated east displacement (m)")
    ax.set_ylabel("Estimated north displacement (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    run = summary["run"]
    confidence = summary["confidence"]
    note = (
        f"Method: {run.get('method')}  Heading: {run.get('heading_source')}  "
        f"Confidence: {confidence.get('overall')}\n"
        "Acceleration-integrated path. No stride or manual distance."
    )
    ax.text(
        0.01,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
