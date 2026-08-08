"""
The experiment harness.

One config -> a full spine-vs-rigid sweep -> CSV, JSON, figures, and video.
Runs headless. `spine` vs `rigid` is a single flag on the variant, and the
rigid variant is built by deleting joints (see model.py), never by stiffening.

Every result row carries its stability verdict alongside its numbers, so a
diverged run cannot be mistaken for a slow one.
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import glbackend, plots, render
from .control import CPGController, GaitParams
from .metrics import METRIC_KEYS, compute_metrics
from .model import DEFAULT_XML, build_model
from .rollout import Command, RolloutLog, run_rollout
from .stability import check_model_sanity


@dataclass
class ExperimentConfig:
    """A complete, serialisable experiment description."""

    name: str = "baseline"
    xml_path: str = DEFAULT_XML
    variants: tuple[str, ...] = ("spine", "rigid")
    #: (label, vx, yaw_rate) triples.
    commands: tuple[tuple[str, float, float], ...] = (
        ("straight_1.0", 1.0, 0.0),
        ("straight_2.0", 2.0, 0.0),
        ("turn_0.8", 1.0, 0.8),
    )
    duration: float = 6.0
    settle: float = 0.5
    seeds: tuple[int, ...] = (0,)
    gait: dict = field(default_factory=dict)
    timestep: float | None = None
    #: Passive-variant spring-damper. Defaults match kp_spine/kd_spine so a
    #: passive spine and a held actuated spine are the same mechanical system.
    passive_stiffness: float = 400.0
    passive_damping: float = 12.0
    render: bool = False
    video_fps: int = 50
    results_dir: str = "results"
    media_dir: str = "media"

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            raw = json.load(f)
        for key in ("variants", "seeds"):
            if key in raw:
                raw[key] = tuple(raw[key])
        if "commands" in raw:
            raw["commands"] = tuple(tuple(c) for c in raw["commands"])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["variants"] = list(self.variants)
        d["seeds"] = list(self.seeds)
        d["commands"] = [list(c) for c in self.commands]
        return d

    def gait_params(self, variant: str) -> GaitParams:
        """
        Gait parameters for a variant.

        Identical for both by construction; the rigid variant simply has no
        spine actuators for the spine terms to reach. Per-variant overrides go
        under a `variant_overrides` key and are recorded in the output so any
        asymmetry in the comparison is visible in the results file.
        """
        base = dict(self.gait)
        overrides = base.pop("variant_overrides", {})
        base.update(overrides.get(variant, {}))
        return GaitParams(**base)


def _provenance() -> dict:
    """Enough context to reproduce or distrust a result later."""
    import mujoco

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                    stderr=subprocess.DEVNULL).strip()
        )
    except Exception:  # noqa: BLE001
        commit, dirty = None, None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "mujoco_gl": glbackend.current(),
    }


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Execute the config and write all outputs. Returns the results dict."""
    results_dir = Path(cfg.results_dir) / cfg.name
    media_dir = Path(cfg.media_dir) / cfg.name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== experiment: {cfg.name} ===")
    print(f"variants : {list(cfg.variants)}")
    print(f"commands : {[c[0] for c in cfg.commands]}")
    print(f"seeds    : {list(cfg.seeds)}   duration: {cfg.duration}s   render: {cfg.render}")

    render_ok, render_msg = (False, "rendering not requested")
    if cfg.render:
        render_ok, render_msg = render.rendering_available()
        print(f"render   : {render_msg}")

    # Build every variant once and record what was actually compiled.
    models: dict[str, tuple] = {}
    for v in cfg.variants:
        model, info = build_model(
            variant=v,
            xml_path=cfg.xml_path,
            timestep=cfg.timestep,
            passive_stiffness=cfg.passive_stiffness,
            passive_damping=cfg.passive_damping,
        )
        issues = check_model_sanity(model, label=v)
        models[v] = (model, info, issues)
        extra = ""
        if info.spine_stiffness:
            extra = (f" spring k={info.spine_stiffness:g} c={info.spine_damping:g}"
                     f" (unactuated)")
        print(f"  built {v:8s}: nq={info.nq} nv={info.nv} nu={info.nu} "
              f"njnt={info.njnt} mass={info.total_mass:.4f}kg "
              f"spine_joints={list(info.spine_joints)}{extra}")

    # Mass-matching is what makes the A/B interpretable; assert it loudly.
    masses = {v: models[v][1].total_mass for v in cfg.variants}
    if len(set(round(m, 9) for m in masses.values())) > 1:
        print(f"\n!! WARNING: variants are NOT mass-matched: {masses}")
        print("!! Speed and cost-of-transport differences may be a mass artefact.\n")

    rows: list[dict] = []
    stability_records: list[dict] = []
    diverged_any = False

    for cmd_label, vx, yaw_rate in cfg.commands:
        command = Command(vx=vx, yaw_rate=yaw_rate)
        print(f"\n-- command {cmd_label}: vx={vx} m/s, yaw_rate={yaw_rate} rad/s")
        logs_for_plot: dict[str, RolloutLog] = {}
        frames_by_variant: dict[str, list] = {}

        for seed in cfg.seeds:
            for v in cfg.variants:
                model, info, _ = models[v]
                params = cfg.gait_params(v)
                # The command must reach the controller, not only the metrics'
                # reference path -- otherwise every command produces an
                # identical gait and the sweep silently compares nothing.
                controller = CPGController(model, params, command=command)

                want_video = cfg.render and render_ok and seed == cfg.seeds[0]
                recorder = render.Recorder(
                    model, fps=cfg.video_fps, enabled=want_video
                ) if want_video else None

                log = run_rollout(
                    model=model,
                    controller=controller,
                    command=command,
                    duration=cfg.duration,
                    settle=cfg.settle,
                    variant=v,
                    recorder=recorder,
                    seed=seed,
                )
                if recorder is not None:
                    recorder.close()
                    frames_by_variant[v] = recorder.frames

                m = compute_metrics(log, total_mass=info.total_mass)
                diverged_any |= log.diverged

                row = {
                    "experiment": cfg.name,
                    "variant": v,
                    "command": cmd_label,
                    "cmd_vx": vx,
                    "cmd_yaw_rate": yaw_rate,
                    "seed": seed,
                    "diverged": int(log.diverged),
                    "sim_seconds": round(log.duration, 4),
                    **{k: m[k] for k in METRIC_KEYS},
                }
                rows.append(row)
                stability_records.append({
                    "variant": v, "command": cmd_label, "seed": seed,
                    **log.stability.as_dict(),
                })

                status = "DIVERGED" if log.diverged else "ok"
                print(f"   {v:6s} seed={seed} [{status:8s}] "
                      f"peak={m['peak_speed_mps']:6.3f}  "
                      f"net={m['net_progress_speed_mps']:6.3f} m/s  "
                      f"CoT={m['cost_of_transport']:6.2f}  "
                      f"xtrack={m['cross_track_rms_m']:5.3f} m  "
                      f"fell={m['fell_over']:.0f}  "
                      f"clip={m['clip_fraction']:.3f}")
                print(f"          stability: {log.stability.summary()}")

                if seed == cfg.seeds[0]:
                    logs_for_plot[v] = log

        # ---- figures and video for this command -----------------------------
        if logs_for_plot:
            plots.gait_diagram(
                logs_for_plot, media_dir / f"gait_{cmd_label}.png",
                title=f"{cfg.name} / {cmd_label} - contact sequence")
            plots.spine_angles(
                logs_for_plot, media_dir / f"spine_angles_{cmd_label}.png",
                title=f"{cfg.name} / {cmd_label} - spine joint angles")
            plots.trajectory(
                logs_for_plot, media_dir / f"trajectory_{cmd_label}.png",
                title=f"{cfg.name} / {cmd_label} - path tracking")
            plots.speed_trace(
                logs_for_plot, media_dir / f"speed_{cmd_label}.png",
                title=f"{cfg.name} / {cmd_label} - forward speed")
            print(f"   figures -> {media_dir}")

        if cfg.render and render_ok and len(frames_by_variant) >= 2:
            a, b = cfg.variants[0], cfg.variants[1]
            fa, fb = frames_by_variant.get(a, []), frames_by_variant.get(b, [])
            sub_a = _clip_caption(logs_for_plot.get(a))
            sub_b = _clip_caption(logs_for_plot.get(b))
            out = media_dir / f"sidebyside_{cmd_label}.mp4"
            if render.side_by_side(fa, fb, out, left_label=a, right_label=b,
                                   left_sub=sub_a, right_sub=sub_b, fps=cfg.video_fps):
                print(f"   video   -> {out}")
            for v, frames in frames_by_variant.items():
                if frames:
                    render.save_png(frames[len(frames) // 2],
                                    media_dir / f"still_{cmd_label}_{v}.png")

    # ---- write outputs ------------------------------------------------------
    csv_path = results_dir / "metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    payload = {
        "config": cfg.as_dict(),
        "provenance": _provenance(),
        "render": {"requested": cfg.render, "available": render_ok, "message": render_msg},
        "variants": {v: models[v][1].as_dict() for v in cfg.variants},
        "model_sanity_issues": {v: models[v][2] for v in cfg.variants},
        "mass_matched": len(set(round(m, 9) for m in masses.values())) == 1,
        "any_diverged": diverged_any,
        "stability": stability_records,
        "rows": rows,
    }
    json_path = results_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")
    if diverged_any:
        print("\n" + "!" * 78)
        print("!!  AT LEAST ONE ROLLOUT DIVERGED. Its metrics are NaN in the CSV.")
        print("!!  Do not aggregate over this experiment without excluding them.")
        print("!" * 78)
    return payload


def _clip_caption(log: RolloutLog | None) -> str:
    if log is None or len(log.t) < 2:
        return ""
    if log.diverged:
        return "DIVERGED"
    return f"mean {float(np.mean(log.fwd_speed)):.2f} m/s"


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def summarise(rows: list[dict], keys: tuple[str, ...] = (
    "peak_speed_mps", "turn_rate_mean_radps", "cost_of_transport", "cross_track_rms_m",
)) -> str:
    """
    Human-readable spine-vs-rigid table, aggregated over seeds.

    Diverged rows are excluded from the aggregate and counted separately --
    silently averaging NaN into a mean is how a broken run becomes a claim.
    """
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["command"], r["variant"])].append(r)

    seen, ordered_cmds = set(), []
    for r in rows:
        if r["command"] not in seen:
            seen.add(r["command"])
            ordered_cmds.append(r["command"])
    variants = []
    for r in rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])

    #: Column headings short enough to fit but still unambiguous.
    short = {
        "peak_speed_mps": "peak_speed",
        "peak_speed_raw_mps": "peak_raw",
        "mean_fwd_speed_mps": "mean_fwd",
        "ground_speed_mps": "ground_spd",
        "net_progress_speed_mps": "net_progress",
        "turn_rate_mean_radps": "turn_rate",
        "cost_of_transport": "cost_transp",
        "cross_track_rms_m": "cross_track",
        "path_error_rms_m": "path_err",
        "fell_over": "fell_over",
    }
    n_seeds = len({r["seed"] for r in rows})
    lines = []
    width = 17 if n_seeds > 1 else 14
    header = f"{'command':<16}{'variant':<9}" + "".join(
        f"{short.get(k, k)[:width - 1]:>{width}}" for k in keys) + f"{'n_div':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for c in ordered_cmds:
        for v in variants:
            rs = groups.get((c, v), [])
            if not rs:
                continue
            good = [r for r in rs if not r["diverged"]]
            ndiv = len(rs) - len(good)
            cells = ""
            for k in keys:
                vals = [r[k] for r in good if r[k] == r[k]]  # drop NaN
                if not vals:
                    cells += f"{'--':>{width}}"
                elif n_seeds > 1:
                    # A mean over seeds without its spread is not interpretable;
                    # most of the differences here are inside one sigma.
                    cells += f"{np.mean(vals):>{width - 7}.3f}+-{np.std(vals):<5.3f}"
                else:
                    cells += f"{np.mean(vals):>{width}.4f}"
            lines.append(f"{c:<16}{v:<9}{cells}{ndiv:>7}")
    return "\n".join(lines)
