"""
Figures: contact/gait diagrams, spine joint traces, trajectories.

Uses the Agg backend unconditionally -- these run headless and must never try
to open a window.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .rollout import LEGS, RolloutLog  # noqa: E402

LEG_LABEL = {"fl": "front left", "fr": "front right", "rl": "rear left", "rr": "rear right"}
#: Draw order puts the diagonal pairs adjacent, which makes a trot readable.
ROW_ORDER = ("fl", "rr", "fr", "rl")

SPINE_LABEL = ("spine yaw", "spine pitch", "spine roll")
SPINE_COLOR = ("#1f77b4", "#d62728", "#2ca02c")


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def gait_diagram(logs: dict[str, RolloutLog], path: str | Path, title: str = "") -> Path | None:
    """
    Contact-sequence diagram: a filled bar per foot for every stance phase.

    One panel per variant, sharing a time axis, so the phase relationship
    between variants is directly comparable.
    """
    usable = {k: v for k, v in logs.items() if len(v.t) > 1}
    if not usable:
        print("[plots] WARNING: no usable rollouts for gait diagram", flush=True)
        return None

    n = len(usable)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, (name, log) in zip(axes, usable.items()):
        t = log.t
        for row, leg in enumerate(ROW_ORDER):
            k = LEGS.index(leg)
            col = log.contacts[:, k]
            # Fill contiguous stance runs as rectangles.
            edges = np.diff(col.astype(np.int8))
            starts = list(np.flatnonzero(edges == 1) + 1)
            ends = list(np.flatnonzero(edges == -1) + 1)
            if col[0]:
                starts = [0] + starts
            if col[-1]:
                ends = ends + [len(col) - 1]
            for s, e in zip(starts, ends):
                ax.barh(row, t[e] - t[s], left=t[s], height=0.62,
                        color="#333333", edgecolor="none")
        ax.set_yticks(range(len(ROW_ORDER)))
        ax.set_yticklabels([LEG_LABEL[l] for l in ROW_ORDER], fontsize=9)
        ax.invert_yaxis()
        ax.set_ylim(len(ROW_ORDER) - 0.4, -0.6)
        ax.grid(axis="x", alpha=0.25, linestyle=":")
        note = "  [DIVERGED]" if log.diverged else ""
        ax.set_title(f"{name}{note}", fontsize=10, loc="left")
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title or "Contact sequence (dark = foot on ground)", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def spine_angles(logs: dict[str, RolloutLog], path: str | Path, title: str = "") -> Path | None:
    """
    The three spine joint angles over time.

    The rigid variant is plotted too, as a flat line at zero -- that is the
    honest picture. Those joints do not exist; they are not merely still.
    """
    usable = {k: v for k, v in logs.items() if len(v.t) > 1}
    if not usable:
        print("[plots] WARNING: no usable rollouts for spine plot", flush=True)
        return None

    n = len(usable)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.7 * n + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, (name, log) in zip(axes, usable.items()):
        sa = log.spine_angles
        any_present = False
        for c in range(3):
            v = sa[:, c] if sa.shape[1] > c else np.array([])
            if v.size and np.isfinite(v).any():
                ax.plot(log.t, np.rad2deg(v), color=SPINE_COLOR[c],
                        label=SPINE_LABEL[c], linewidth=1.4)
                any_present = True
        if not any_present:
            ax.axhline(0.0, color="#888888", linewidth=1.6, linestyle="--")
            ax.text(0.5, 0.5, "no spine joints in this variant (rigid trunk)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#666666")
            ax.set_ylim(-40, 40)
        ax.axhline(0.0, color="#cccccc", linewidth=0.8, zorder=0)
        ax.set_ylabel("angle (deg)", fontsize=9)
        note = "  [DIVERGED]" if log.diverged else ""
        ax.set_title(f"{name}{note}", fontsize=10, loc="left")
        ax.grid(alpha=0.25, linestyle=":")
        if any_present:
            ax.legend(fontsize=8, ncol=3, loc="upper right")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title or "Spine joint angles", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def trajectory(logs: dict[str, RolloutLog], path: str | Path, title: str = "") -> Path | None:
    """Top-down actual vs commanded path, for the path-tracking metric."""
    usable = {k: v for k, v in logs.items() if len(v.t) > 1}
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    ref_drawn = False
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, log) in enumerate(usable.items()):
        if not ref_drawn:
            ax.plot(log.ref_pos[:, 0], log.ref_pos[:, 1], "--", color="#888888",
                    linewidth=1.5, label="commanded")
            ref_drawn = True
        ax.plot(log.pos[:, 0], log.pos[:, 1], color=colors[i % len(colors)],
                linewidth=1.8, label=name)
        ax.plot(log.pos[0, 0], log.pos[0, 1], "o", color=colors[i % len(colors)], ms=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)
    ax.set_title(title or "Path tracking (top-down)", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def speed_trace(logs: dict[str, RolloutLog], path: str | Path, title: str = "") -> Path | None:
    """Forward speed over time, with the commanded speed marked."""
    usable = {k: v for k, v in logs.items() if len(v.t) > 1}
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(11, 3.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cmd = None
    for i, (name, log) in enumerate(usable.items()):
        ax.plot(log.t, log.fwd_speed, color=colors[i % len(colors)],
                linewidth=1.3, label=name, alpha=0.9)
        cmd = log.command.vx
    if cmd is not None:
        ax.axhline(cmd, color="#888888", linestyle="--", linewidth=1.3,
                   label=f"commanded {cmd:g} m/s")
    ax.axhline(0.0, color="#cccccc", linewidth=0.8, zorder=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("forward speed (m/s)")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=9)
    ax.set_title(title or "Forward speed", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)
