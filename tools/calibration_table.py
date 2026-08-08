"""
Regenerate the spine-stiffness calibration table.

This is the control the whole study rests on: a spine HELD at neutral must be
dynamically equivalent to a deleted spine. If it is not, every spine-vs-rigid
number is confounded by how well the spine is held rather than by trunk
topology.

Writes results/calibration.csv and prints the table used in the README.

    python tools/calibration_table.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import cheetah  # noqa: F401  -- sets MUJOCO_GL before mujoco is imported
from cheetah.control import CPGController, GaitParams
from cheetah.metrics import compute_metrics
from cheetah.model import build_model
from cheetah.rollout import Command, run_rollout
from harness import DEFAULT_GAIT

SEEDS = tuple(range(8))
SPEEDS = (1.0, 2.0)

#: A true hold zeroes the undulation amplitudes AND the steering gain. Leaving
#: turn_spine_gain non-zero lets the heading loop steer the "held" spine, which
#: is not a held spine at all -- see commit ed10833.
HELD = dict(
    DEFAULT_GAIT,
    spine_pitch_amp=0.0,
    spine_yaw_amp=0.0,
    spine_roll_amp=0.0,
    turn_spine_gain=0.0,
)


def evaluate(model, info, gait: dict, vx: float) -> tuple[float, float, float, float]:
    """(mean net speed, std, fall fraction, mean peak spine deflection in deg)."""
    nets, fells, defl = [], [], []
    for seed in SEEDS:
        cmd = Command(vx=vx, yaw_rate=0.0)
        ctrl = CPGController(model, GaitParams(**gait), command=cmd)
        log = run_rollout(model, ctrl, cmd, duration=8.0, settle=0.5, seed=seed)
        if log.diverged:
            fells.append(1.0)
            continue
        m = compute_metrics(log, info.total_mass)
        nets.append(m["net_progress_speed_mps"])
        fells.append(m["fell_over"])
        if np.isfinite(log.spine_angles).any():
            defl.append(np.rad2deg(np.nanmax(np.abs(log.spine_angles))))
    return (
        float(np.mean(nets)) if nets else float("nan"),
        float(np.std(nets)) if nets else float("nan"),
        float(np.mean(fells)),
        float(np.mean(defl)) if defl else float("nan"),
    )


def main() -> int:
    rows = []
    configs = []

    rigid_model, rigid_info = build_model(variant="rigid")
    configs.append(("rigid (joints deleted)", rigid_model, rigid_info, HELD))

    for kp, kd in ((120.0, 4.0), (400.0, 12.0), (1200.0, 40.0), (4000.0, 120.0)):
        m, i = build_model(variant="spine")
        configs.append((f"actuated, held, kp={kp:g} kd={kd:g}", m, i,
                        dict(HELD, kp_spine=kp, kd_spine=kd)))

    for k, c in ((400.0, 12.0), (100.0, 6.0)):
        m, i = build_model(variant="passive", passive_stiffness=k, passive_damping=c)
        configs.append((f"passive spring k={k:g} c={c:g}", m, i, HELD))

    header = (f"{'configuration':<34}"
              + "".join(f"{'net@' + str(v):>17}{'fell':>7}{'defl':>7}" for v in SPEEDS))
    print("Spine hold calibration: a held spine must match a deleted spine.")
    print(f"{len(SEEDS)} seeds, 8 s, straight command.\n")
    print(header)
    print("-" * len(header))

    for label, model, info, gait in configs:
        cells, rec = "", {"configuration": label}
        for vx in SPEEDS:
            mean, std, fell, defl = evaluate(model, info, gait, vx)
            cells += f"{mean:>10.3f}+-{std:<5.3f}{fell:>7.2f}"
            cells += f"{defl:>7.2f}" if defl == defl else f"{'-':>7}"
            rec[f"net_{vx}"] = round(mean, 4)
            rec[f"std_{vx}"] = round(std, 4)
            rec[f"fell_{vx}"] = round(fell, 4)
            rec[f"defl_deg_{vx}"] = None if defl != defl else round(defl, 3)
        print(f"{label:<34}{cells}")
        rows.append(rec)

    out = Path("results") / "calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    print("\n'defl' is the peak spine deflection in degrees. A held spine that")
    print("deflects a lot is not being held, and its comparison against rigid")
    print("is measuring hold quality rather than trunk topology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
