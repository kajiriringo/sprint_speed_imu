"""Start-time detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RunConfig
from .errors import InvalidOptionError, ProcessingError


def detect_start_time(df: pd.DataFrame, config: RunConfig) -> tuple[float, list[str]]:
    if config.start_mode == "manual":
        if config.start_time is None:
            raise InvalidOptionError("--start-time is required when --start-mode=manual.")
        return float(config.start_time), []
    if config.start_mode != "auto":
        raise InvalidOptionError("--start-mode must be auto or manual.")

    time = df["time_s"].to_numpy(dtype=float)
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    a_norm = np.linalg.norm(acc, axis=1)
    first_time = float(time[0])
    baseline_mask = time <= first_time + 1.0
    if int(np.sum(baseline_mask)) < 3:
        baseline_mask = np.arange(len(time)) < min(len(time), 10)

    baseline = a_norm[baseline_mask]
    med = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - med)))
    robust_sigma = 1.4826 * mad
    if robust_sigma < 1e-6:
        robust_sigma = max(float(np.std(baseline)), 0.05)
    threshold = med + 6.0 * robust_sigma

    over = np.abs(a_norm - med) > (threshold - med)
    consecutive = 0
    for idx, is_over in enumerate(over):
        if is_over:
            consecutive += 1
            if consecutive >= 3:
                start_idx = max(0, idx - 2)
                return float(time[start_idx]), []
        else:
            consecutive = 0

    message = "Auto start detection failed; using first timestamp as start_time."
    if config.strict:
        raise ProcessingError(message)
    return first_time, [message]
