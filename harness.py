"""
Experiment harness CLI. Everything here runs headless.

    python harness.py check                 # environment + model + stability
    python harness.py run                   # default spine-vs-rigid sweep
    python harness.py run --config cfg.json # any config, from JSON
    python harness.py run --render          # also write MP4/PNG to media/
    python harness.py freefall              # torque-matched reorientation test

Rendering is off by default so experiments stay fast; --render turns it on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cheetah  # noqa: F401  -- sets MUJOCO_GL before mujoco is imported
from cheetah import glbackend, render
from cheetah.control import CPGController, GaitParams
from cheetah.experiment import ExperimentConfig, run_experiment, summarise
from cheetah.model import DEFAULT_XML, build_model, set_home_pose
from cheetah.rollout import Command, run_rollout
from cheetah.stability import StabilityMonitor, check_model_sanity, warn_loudly

import mujoco  # noqa: E402
import numpy as np  # noqa: E402


#: Default gait. Tuned only enough to produce locomotion in both variants; it
#: is deliberately not optimised for either, since the point of the open-loop
#: baseline is a controller neither variant got to tune against.
DEFAULT_GAIT = {
    "gait": "trot",
    "freq": 2.4,
    "hip_amp": 0.45,
    "knee_amp": 0.55,
    "spine_yaw_amp": 0.18,
    "spine_pitch_amp": 0.22,
    "spine_phase": 0.0,
    "flexion_ratio": 2.0,
    "turn_abduct_gain": 0.15,
    "turn_spine_gain": 0.35,
}


def cmd_check(args: argparse.Namespace) -> int:
    """Environment, model integrity, and numerical stability shakedown."""
    print("=" * 74)
    print("ENVIRONMENT")
    print("=" * 74)
    print(f"  platform      : {sys.platform}")
    print(f"  python        : {sys.version.split()[0]}")
    print(f"  mujoco        : {mujoco.__version__}")
    print(f"  numpy         : {np.__version__}")
    print(f"  GL candidates : {list(glbackend.candidates())}")
    print(f"  MUJOCO_GL     : {glbackend.current()}"
          f"{' (from environment)' if glbackend.was_explicit() else ' (auto-selected)'}")
    ok, msg = render.rendering_available()
    print(f"  rendering     : {'OK' if ok else 'UNAVAILABLE'} - {msg}")
    if not ok:
        print("  -> numeric experiments will still run; only media output is lost")

    print("\n" + "=" * 74)
    print("MODEL VARIANTS")
    print("=" * 74)
    infos = {}
    for v in ("spine", "rigid"):
        model, info = build_model(spine=(v == "spine"), xml_path=args.xml)
        infos[v] = (model, info)
        print(f"  {v:6s}: nq={info.nq:3d} nv={info.nv:3d} nu={info.nu:3d} "
              f"njnt={info.njnt:3d} mass={info.total_mass:.4f} kg")
        if info.removed_joints:
            print(f"          removed joints: {list(info.removed_joints)}")
            print(f"          removed motors: {list(info.removed_motors)}")
        issues = check_model_sanity(model, label=v)
        print(f"          sanity: {'clean' if not issues else issues}")

    ms, mr = infos["spine"][1], infos["rigid"][1]
    matched = abs(ms.total_mass - mr.total_mass) < 1e-9
    print(f"\n  mass-matched  : {matched} "
          f"({ms.total_mass:.6f} vs {mr.total_mass:.6f} kg)")
    print(f"  DOF removed   : {ms.nv - mr.nv} (nv {ms.nv} -> {mr.nv})")
    print(f"  actuators lost: {ms.nu - mr.nu} (nu {ms.nu} -> {mr.nu})")
    if ms.nv == mr.nv:
        print("  !! rigid variant has the same DOF count as spine - joint removal FAILED")
        return 1

    print("\n" + "=" * 74)
    print("STABILITY SHAKEDOWN (5 s, zero torque / full-amplitude random torque)")
    print("=" * 74)
    failed = False
    for v in ("spine", "rigid"):
        model, _ = infos[v]
        for mode in ("zero", "random"):
            data = mujoco.MjData(model)
            set_home_pose(model, data)
            mon = StabilityMonitor(model, data)
            rng = np.random.default_rng(0)
            n = int(round(5.0 / model.opt.timestep))
            for i in range(n):
                if mode == "random":
                    data.ctrl[:] = rng.uniform(-1, 1, model.nu) * model.actuator_ctrlrange[:, 1]
                mujoco.mj_step(model, data)
                if not mon.check(i):
                    break
            rep = mon.report()
            print(f"  {v:6s} ctrl={mode:6s}: {rep.summary()}")
            if rep.diverged:
                warn_loudly(f"{v}/{mode}", rep)
                failed = True

    print("\n" + ("CHECK FAILED" if failed else "CHECK PASSED"))
    return 1 if failed else 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.config:
        cfg = ExperimentConfig.from_json(args.config)
    else:
        cfg = ExperimentConfig(name=args.name, gait=dict(DEFAULT_GAIT))
    # CLI flags override the config file.
    if args.render:
        cfg.render = True
    if args.duration is not None:
        cfg.duration = args.duration
    if args.seeds is not None:
        cfg.seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.xml != DEFAULT_XML:
        cfg.xml_path = args.xml
    if args.variants is not None:
        cfg.variants = tuple(args.variants.split(","))

    payload = run_experiment(cfg)

    print("\n" + "=" * 74)
    print(f"SUMMARY: {cfg.name}   (means over {len(cfg.seeds)} seed(s), diverged rows excluded)")
    print("=" * 74)
    print(summarise(payload["rows"]))
    print()
    print("peak_speed_mps     m/s, 0.1 s moving average of body-frame forward speed")
    print("turn_rate          rad/s, mean signed yaw rate")
    print("cost_of_transport  dimensionless, mechanical energy / (m g distance)")
    print("cross_track        m, RMS distance to the commanded path (speed-independent)")
    return 0


def cmd_freefall(args: argparse.Namespace) -> int:
    """
    Zero-gravity reorientation, with the control the starter script omitted.

    The original test drove the spine on the spine variant and drove *nothing*
    on the rigid variant, so its 0 deg result measured the absence of a command,
    not the absence of a capability. The rigid trunk still has 12 leg joints and
    a 2-DOF tail, so it can absolutely reorient itself in free fall.

    This version drives every variant with the DOF it actually has, and reports
    the torque budget each one spent, because the variants have different
    actuator limits and an unmatched budget is not a controlled comparison.
    """
    dur = args.duration
    print("=" * 78)
    print(f"FREE-FALL REORIENTATION  (zero gravity, {dur:g} s, no ground contact)")
    print("=" * 78)

    strategies = ("none", "spine_only", "tail_only", "legs_only", "legs_tail", "all")
    results = []
    for variant in ("spine", "rigid"):
        for strategy in strategies:
            if strategy in ("spine_only", "all") and variant == "rigid":
                continue  # no spine actuators exist to drive

            # Fresh model per run: zeroing gravity mutates the model in place,
            # and a shared model would leak that into later rows.
            model, _ = build_model(spine=(variant == "spine"), xml_path=args.xml)
            model.opt.gravity[:] = [0.0, 0.0, 0.0]
            act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                   for i in range(model.nu)}

            data = mujoco.MjData(model)
            set_home_pose(model, data)
            data.qpos[2] = 5.0
            mujoco.mj_forward(model, data)

            mon = StabilityMonitor(model, data)
            yaw0 = _yaw(data.qpos[3:7])
            torque_integral = 0.0
            n = int(round(dur / model.opt.timestep))
            for i in range(n):
                t = i * model.opt.timestep
                ctrl = np.zeros(model.nu)
                s = 1.0 if (t * 2.0) % 1.0 < 0.5 else -1.0
                if strategy in ("spine_only", "all"):
                    ctrl[act["spine_yaw"]] = 40.0 * s
                    ctrl[act["spine_roll"]] = 40.0 * s
                if strategy in ("tail_only", "legs_tail", "all"):
                    ctrl[act["tail_yaw"]] = 8.0 * s
                if strategy in ("legs_only", "legs_tail", "all"):
                    for leg, sign in (("fl", 1), ("rl", 1), ("fr", -1), ("rr", -1)):
                        ctrl[act[f"{leg}_abduct"]] = 20.0 * s * sign
                        ctrl[act[f"{leg}_knee"]] = -15.0 * s * sign
                ctrl = np.clip(ctrl, model.actuator_ctrlrange[:, 0],
                               model.actuator_ctrlrange[:, 1])
                data.ctrl[:] = ctrl
                torque_integral += float(np.abs(ctrl).sum()) * model.opt.timestep
                mujoco.mj_step(model, data)
                if not mon.check(i):
                    break

            rep = mon.report()
            net = float("nan") if rep.diverged else np.rad2deg(_yaw(data.qpos[3:7]) - yaw0)
            if rep.diverged:
                warn_loudly(f"freefall {variant}/{strategy}", rep)
            results.append((variant, strategy, net, torque_integral, rep.diverged))

    print(f"\n{'variant':<9}{'actuation':<13}{'net yaw (deg)':>15}{'torque-time (N.m.s)':>22}")
    print("-" * 60)
    for variant, strategy, net, tq, div in results:
        flag = "  DIVERGED" if div else ""
        net_s = "     nan" if net != net else f"{net:+8.2f}"
        print(f"{variant:<9}{strategy:<13}{net_s:>15}{tq:>22.1f}{flag}")

    print("\nRead this as: the rigid trunk is NOT a rigid body. It keeps 12 leg")
    print("joints and a 2-DOF tail, so it reorients in free fall too. The spine")
    print("adds authority; it does not add the capability. Compare rows at")
    print("similar torque-time before concluding anything about topology.")
    return 0


def _yaw(q) -> float:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return float(np.arctan2(m[3], m[0]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", default=DEFAULT_XML, help="source MJCF")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="environment, model integrity, stability")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("run", help="run an experiment")
    r.add_argument("--config", help="JSON config file")
    r.add_argument("--name", default="baseline")
    r.add_argument("--render", action="store_true",
                   help="write MP4/PNG to media/ (off by default for speed)")
    r.add_argument("--duration", type=float)
    r.add_argument("--seeds", help="comma-separated, e.g. 0,1,2")
    r.add_argument("--variants", help="comma-separated, e.g. spine,rigid")
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("freefall", help="torque-matched reorientation test")
    f.add_argument("--duration", type=float, default=2.0)
    f.set_defaults(func=cmd_freefall)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
