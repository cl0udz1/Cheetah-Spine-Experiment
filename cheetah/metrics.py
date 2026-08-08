"""
Locomotion metrics.

The four headline numbers requested: peak speed, turning rate, cost of
transport, path-tracking error. Plus the supporting quantities needed to tell
whether those four mean anything (did it stay upright, was the actuator
saturated, what was the duty factor).

Divergence policy: if the rollout diverged, every metric is NaN. There is no
"partial credit" path where the first 60% of a blown-up rollout gets averaged
into a plausible number.
"""
from __future__ import annotations

import numpy as np

from .rollout import LEGS, RolloutLog

G = 9.81

#: Metric keys that must exist in every row, so CSVs stay rectangular.
METRIC_KEYS = (
    "peak_speed_mps",
    "peak_speed_raw_mps",
    "mean_fwd_speed_mps",
    "steady_fwd_speed_mps",
    "speed_tracking_error_mps",
    "ground_speed_mps",
    "net_progress_speed_mps",
    "turn_rate_mean_radps",
    "turn_rate_peak_radps",
    "cost_of_transport",
    "energy_J",
    "mean_power_W",
    "path_error_rms_m",
    "cross_track_rms_m",
    "distance_m",
    "mean_height_m",
    "min_height_m",
    "fell_over",
    "mean_abs_roll_rad",
    "mean_abs_pitch_rad",
    "duty_fl", "duty_fr", "duty_rl", "duty_rr",
    "stride_freq_hz",
    "spine_yaw_amp_rad",
    "spine_pitch_amp_rad",
    "spine_roll_amp_rad",
    "spine_flexion_extension_ratio",
    "clip_fraction",
)

#: Trunk height below which we call it a fall, in metres. Home stance is 0.388.
FALL_HEIGHT = 0.18


def _nan_metrics() -> dict:
    return {k: float("nan") for k in METRIC_KEYS}


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def _cross_track_rms(actual: np.ndarray, ref: np.ndarray, chunk: int = 256) -> float:
    """
    RMS distance from each actual point to the nearest point on the reference
    polyline.

    This is the speed-independent notion of path error: a robot that follows
    the right curve but lags along it scores well here, and badly on
    `path_error_rms_m`. Reporting both separates "went the wrong way" from
    "went the right way too slowly".
    """
    if actual.size == 0 or ref.size == 0:
        return float("nan")
    out = np.empty(len(actual))
    for i in range(0, len(actual), chunk):
        blk = actual[i : i + chunk]
        d = np.linalg.norm(blk[:, None, :] - ref[None, :, :], axis=2)
        out[i : i + len(blk)] = d.min(axis=1)
    return float(np.sqrt(np.mean(out**2)))


def _duty_and_stride(contacts: np.ndarray, dt: float) -> tuple[dict, float]:
    """Per-foot duty factor, and stride frequency from touchdown intervals."""
    duty = {}
    freqs = []
    for k, leg in enumerate(LEGS):
        col = contacts[:, k]
        duty[f"duty_{leg}"] = float(col.mean()) if col.size else float("nan")
        # Touchdowns = rising edges.
        if col.size > 1:
            td = np.flatnonzero((~col[:-1]) & col[1:])
            if td.size >= 2:
                periods = np.diff(td) * dt
                periods = periods[periods > 1e-6]
                if periods.size:
                    freqs.append(1.0 / float(np.median(periods)))
    stride = float(np.mean(freqs)) if freqs else float("nan")
    return duty, stride


def compute_metrics(log: RolloutLog, total_mass: float) -> dict:
    """
    Reduce a rollout to a flat dict of scalars.

    Returns all-NaN if the rollout diverged or produced no usable samples.
    """
    if log.diverged or len(log.t) < 2:
        m = _nan_metrics()
        m["clip_fraction"] = float(log.clip_fraction)
        return m

    dt = log.dt
    duration = float(log.t[-1] - log.t[0]) + dt

    # Peak speed on a 0.1 s moving average. The raw per-step maximum is
    # dominated by contact impulses and is not a speed anyone can use.
    win = max(1, int(round(0.1 / dt)))
    smoothed = _moving_average(log.fwd_speed, win)
    peak = float(np.max(smoothed)) if smoothed.size else float("nan")
    peak_raw = float(np.max(log.fwd_speed))
    tail_start = int(0.4 * len(log.fwd_speed))
    steady = float(np.mean(log.fwd_speed[tail_start:]))

    xy = log.pos[:, :2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    distance = float(seg.sum())
    # Signed forward progress: net displacement projected on the STARTING
    # heading. An unsigned norm here scores a robot running smoothly backwards
    # as a fast robot, which is how a sign error in the gait survives a sweep.
    h0 = np.array([np.cos(log.yaw[0]), np.sin(log.yaw[0])])
    net_progress = float(np.dot(xy[-1] - xy[0], h0))
    ground_speed = distance / duration if duration > 0 else float("nan")

    energy = float(np.sum(log.power) * dt)
    mean_power = float(np.mean(log.power))

    # Cost of transport: dimensionless mechanical energy per unit weight per
    # unit distance. Guarded against the degenerate "did not move" case, where
    # CoT is undefined rather than infinite.
    if distance > 0.05:
        cot = energy / (total_mass * G * distance)
    else:
        cot = float("nan")

    path_err = float(np.sqrt(np.mean(np.sum((xy - log.ref_pos) ** 2, axis=1))))
    cross_err = _cross_track_rms(xy, log.ref_pos)

    heights = log.pos[:, 2]
    duty, stride = _duty_and_stride(log.contacts, dt)

    sa = log.spine_angles
    def _amp(col: int) -> float:
        if sa.shape[0] == 0:
            return float("nan")
        v = sa[:, col]
        if not np.isfinite(v).any():
            return float("nan")  # joint absent in this variant
        return float(np.nanmax(v) - np.nanmin(v))

    # Realised flexion:extension ratio of the sagittal spine. Confirms the
    # asymmetry actually reached the joint rather than just the setpoint.
    #
    # Flexion is NEGATIVE spine_pitch, verified against the model: -0.4 rad
    # puts the tail base at z=0.297 and +0.4 rad at z=0.516 against 0.408 at
    # neutral, so negative pitch lowers the hind end (the gathered phase).
    # A value below 1.0 here means the asymmetry is running BACKWARDS, not
    # that it is attenuated -- attenuation would land between 1.0 and the
    # commanded ratio.
    pitch_col = sa[:, 1] if sa.shape[0] else np.array([])
    if pitch_col.size and np.isfinite(pitch_col).any():
        flex = float(-np.nanmin(pitch_col))
        ext = float(np.nanmax(pitch_col))
        ratio = flex / ext if ext > 1e-6 else float("nan")
    else:
        ratio = float("nan")

    out = {
        "peak_speed_mps": peak,
        "peak_speed_raw_mps": peak_raw,
        "mean_fwd_speed_mps": float(np.mean(log.fwd_speed)),
        # Steady state = the last 60% of the run, past the ramp and the
        # feedback loops' settling. speed_tracking_error is the number that
        # decides whether a config label like "straight_1.0" is honest.
        "steady_fwd_speed_mps": steady,
        "speed_tracking_error_mps": steady - log.command.vx,
        "ground_speed_mps": ground_speed,
        "net_progress_speed_mps": net_progress / duration if duration > 0 else float("nan"),
        "turn_rate_mean_radps": float(np.mean(log.yaw_rate)),
        "turn_rate_peak_radps": float(np.max(np.abs(log.yaw_rate))),
        "cost_of_transport": cot,
        "energy_J": energy,
        "mean_power_W": mean_power,
        "path_error_rms_m": path_err,
        "cross_track_rms_m": cross_err,
        "distance_m": distance,
        "mean_height_m": float(np.mean(heights)),
        "min_height_m": float(np.min(heights)),
        "fell_over": float(np.min(heights) < FALL_HEIGHT),
        "mean_abs_roll_rad": float(np.mean(np.abs(log.roll))),
        "mean_abs_pitch_rad": float(np.mean(np.abs(log.pitch))),
        "stride_freq_hz": stride,
        "spine_yaw_amp_rad": _amp(0),
        "spine_pitch_amp_rad": _amp(1),
        "spine_roll_amp_rad": _amp(2),
        "spine_flexion_extension_ratio": ratio,
        "clip_fraction": float(log.clip_fraction),
    }
    out.update(duty)
    return out
