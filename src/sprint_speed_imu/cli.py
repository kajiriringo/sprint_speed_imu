"""Command line interface."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from .config import RunConfig, SyntheticConfig
from .errors import SprintSpeedImuError


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args and args[0] == "synthetic":
            return _cmd_synthetic(args[1:])
        if args and args[0] == "compare":
            return _cmd_compare(args[1:])
        if args and args[0] == "run":
            return _cmd_run(args[1:])
        return _cmd_run(args)
    except (SprintSpeedImuError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _cmd_run(argv: list[str]) -> int:
    parser = _run_parser()
    args = parser.parse_args(argv)
    from .pipeline import run_analysis

    config = _run_config_from_args(args)
    summary = run_analysis(config)
    print(f"Wrote {config.output_dir}")
    print(f"Overall confidence: {summary['confidence']['overall']}")
    return 0


def _cmd_synthetic(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sprint-speed-imu synthetic",
        description="Generate a deterministic synthetic sprint IMU CSV.",
    )
    parser.add_argument("--distance-m", type=float, default=50.0)
    parser.add_argument("--duration-s", type=float, default=6.5)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--yaw-deg", type=float, default=30.0)
    parser.add_argument("--noise-std-g", type=float, default=0.01)
    parser.add_argument("--pre-start-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    from .synthetic import write_synthetic_csv

    path = write_synthetic_csv(
        SyntheticConfig(
            output=args.output,
            distance_m=args.distance_m,
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate_hz,
            yaw_deg=args.yaw_deg,
            noise_std_g=args.noise_std_g,
            pre_start_s=args.pre_start_s,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    )
    print(f"Wrote {path}")
    return 0


def _cmd_compare(argv: list[str]) -> int:
    parser = _compare_parser()
    args = parser.parse_args(argv)
    from .pipeline import run_analysis
    from .pipeline import preflight_analysis

    base = _run_config_from_args(args, method="pca")
    for method in ("pca", "attitude"):
        preflight_analysis(replace(base, method=method))

    final_output_dir = base.output_dir
    _validate_compare_target(final_output_dir, base.overwrite)
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{final_output_dir.name}.tmp-", dir=final_output_dir.parent)
    )
    temp_base = replace(base, output_dir=temp_dir, overwrite=True)
    summaries = []
    try:
        for method in ("pca", "attitude"):
            config = replace(
                temp_base,
                method=method,
                output_dir=temp_dir / method,
                overwrite=True,
            )
            summaries.append(run_analysis(config))

        _write_compare_outputs(temp_dir, summaries)
        _replace_output_dir(temp_dir, final_output_dir, base.overwrite)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    print(f"Wrote {final_output_dir}")
    return 0


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sprint-speed-imu",
        description=(
            "Estimate sprint speed curves from WT9011DCL or similar IMU CSV logs. "
            "Subcommands: run, compare, synthetic."
        ),
    )
    _add_analysis_options(parser, include_method=True)
    return parser


def _compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sprint-speed-imu compare",
        description="Run both pca and attitude methods and create comparison outputs.",
    )
    _add_analysis_options(parser, include_method=False)
    return parser


def _add_analysis_options(parser: argparse.ArgumentParser, include_method: bool) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    if include_method:
        parser.add_argument("--method", choices=["attitude", "pca"], required=True)
    parser.add_argument("--distance-source", choices=["manual", "auto"], required=True)
    parser.add_argument("--distance-m", type=float)
    parser.add_argument("--acc-unit", choices=["g", "mps2"], default="g")
    parser.add_argument("--gyro-unit", choices=["dps", "radps"], default="dps")
    parser.add_argument("--angle-unit", choices=["deg", "rad"], default="deg")
    parser.add_argument("--time-column", default="time_s")
    parser.add_argument("--sample-rate-hz", type=float)
    parser.add_argument("--column-map", type=Path)
    parser.add_argument("--start-mode", choices=["auto", "manual"], default="auto")
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-mode", choices=["data-end", "manual"], default="data-end")
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--distance-bin-m", type=float, default=1.0)
    parser.add_argument("--smooth", type=_parse_bool, default=True)
    parser.add_argument("--smoothing-method", choices=["savgol", "butter", "kalman", "none"], default="savgol")
    parser.add_argument("--kalman-process-noise", type=float, default=0.05)
    parser.add_argument("--kalman-measurement-noise", type=float, default=0.5)
    parser.add_argument("--course-yaw-deg", type=float, default=0.0)
    parser.add_argument("--euler-order", default="xyz")
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--gravity-mode", choices=["auto", "subtract-world-z", "baseline"], default="auto")
    parser.add_argument("--pca-window", choices=["run", "first-2s", "first-half"], default="first-half")
    parser.add_argument("--min-pca-var-ratio", type=float, default=0.45)
    parser.add_argument("--max-correction-ratio", type=float, default=2.0)
    parser.add_argument(
        "--correction-mode",
        choices=["mean-speed-shape", "bias", "scale", "raw-integration"],
        default="mean-speed-shape",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def _run_config_from_args(args: argparse.Namespace, method: str | None = None) -> RunConfig:
    return RunConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        method=method or args.method,
        distance_source=args.distance_source,
        distance_m=args.distance_m,
        acc_unit=args.acc_unit,
        gyro_unit=args.gyro_unit,
        angle_unit=args.angle_unit,
        time_column=args.time_column,
        sample_rate_hz=args.sample_rate_hz,
        column_map=args.column_map,
        start_mode=args.start_mode,
        start_time=args.start_time,
        end_mode=args.end_mode,
        end_time=args.end_time,
        duration_s=args.duration_s,
        distance_bin_m=args.distance_bin_m,
        smooth=args.smooth,
        smoothing_method=args.smoothing_method,
        kalman_process_noise=args.kalman_process_noise,
        kalman_measurement_noise=args.kalman_measurement_noise,
        course_yaw_deg=args.course_yaw_deg,
        euler_order=args.euler_order,
        gravity=args.gravity,
        gravity_mode=args.gravity_mode,
        pca_window=args.pca_window,
        min_pca_var_ratio=args.min_pca_var_ratio,
        max_correction_ratio=args.max_correction_ratio,
        correction_mode=args.correction_mode,
        debug=args.debug,
        strict=args.strict,
        overwrite=args.overwrite,
    )


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def _validate_compare_target(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_dir}.")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite.")


def _replace_output_dir(temp_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite.")
            shutil.rmtree(output_dir)
        else:
            output_dir.rmdir()
    shutil.move(str(temp_dir), str(output_dir))


def _write_compare_outputs(output_dir: Path, summaries: list[dict]) -> None:
    from .mpl_setup import configure_matplotlib_cache

    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for summary in summaries:
        method = summary["run"]["method"]
        curve_path = output_dir / method / "speed_curve.csv"
        curve = pd.read_csv(curve_path)
        ax.plot(curve["distance_m"], curve["estimated_speed_mps"], label=method, linewidth=2.0)
        rows.append(
            {
                "method": method,
                "overall_confidence": summary["confidence"]["overall"],
                "max_estimated_speed_mps": summary["results"]["max_estimated_speed_mps"],
                "max_estimated_speed_distance_m": summary["results"]["max_estimated_speed_distance_m"],
                "warnings": summary["warnings"],
            }
        )
    ax.set_title("Estimated Sprint Speed Curve Comparison")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Estimated Speed (m/s)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.text(
        0.01,
        0.02,
        "Estimated from IMU. Not official timing.",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "speed_curve_compare.png", dpi=160)
    plt.close(fig)

    (output_dir / "summary_compare.json").write_text(
        json.dumps({"tool": "sprint-speed-imu", "version": "0.1.0", "runs": rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
