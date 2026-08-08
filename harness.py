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


#: Default gait.
#:
#: The leg parameters were selected by grid search ON THE RIGID MODEL ONLY, at
#: two commanded speeds, scoring the worst case rather than the best. The rigid
#: variant has no spine, so no spine setting could influence the choice. This
#: biases the baseline in the rigid variant's favour: it is running a gait
#: chosen for it, and the spine variant inherits that gait unchanged. Any spine
#: advantage measured against this baseline is therefore a conservative one.
#:
#: Only 6 of 72 refinement combinations stayed upright at both commanded speeds,
#: so this operating point is narrow. kd_leg=3.0 falls over every time; 6.0 is
#: doing real work here, not decoration.
#:
#: Spine undulation parameters are the BEST of a 96-point search over amplitude
#: x phase x frequency-multiple x flexion sign, run on the spine model at 4
#: seeds. This deliberately gives the spine variant the advantage of tuning that
#: the rigid variant cannot use, so the headline comparison shows the spine at
#: its best rather than at a strawman setting.
#:
#: What that search found: only 6 of 96 spine configurations beat the rigid
#: baseline at all, and the winner does so by +3.7% (1.679+-0.002 vs
#: 1.620+-0.022 m/s). The optimum amplitude is 0.05 rad -- under 3 deg
#: commanded, ~4.7 deg realised -- which is an order of magnitude less spine
#: travel than a galloping cheetah uses. At 0.30 rad the robot falls on every
#: seed. See `harness.py spine-sweep`.
DEFAULT_GAIT = {
    "gait": "trot",
    "freq": 2.6,
    "duty": 0.75,
    "hip_amp_max": 0.6,
    "knee_amp": 0.95,
    "kp_leg": 150.0,
    "kd_leg": 6.0,
    "ramp_time": 0.6,
    # Spine stabilising gains, calibrated so that a spine HELD at neutral is
    # dynamically equivalent to the rigid trunk (1.632+-0.015 vs 1.622+-0.020
    # m/s at vx=1.0). That equivalence is the control the comparison needs:
    # with it, any remaining spine-vs-rigid difference is attributable to
    # undulating the spine rather than to a slack joint in the load path.
    # kp_spine=120 (the earlier value) leaves the spine back-driven ~3.8 deg
    # and costs 12% of forward speed before any undulation is commanded.
    # Going the other way, kp_spine>=1200 degrades speed AND inflates cost of
    # transport -- the numerical penalty for faking a weld with stiffness,
    # which is exactly why the rigid variant deletes joints instead.
    "kp_spine": 400.0,
    "kd_spine": 12.0,
    "spine_pitch_amp": 0.05,
    "spine_yaw_amp": 0.0375,
    "spine_phase": 0.375,
    "spine_freq_mult": 1.0,
    "flexion_sign": -1.0,
    "flexion_ratio": 2.0,
    # Turning gains, also selected on the rigid model. These hit +0.819 rad/s
    # against a commanded +0.8. The tail contributes almost nothing here
    # (0.819 vs 0.813 with it enabled), so it stays off to keep the mechanism
    # minimal and the spine's contribution unambiguous.
    "turn_stride_gain": 0.4,
    "turn_abduct_gain": 0.5,
    # Sign matters more than magnitude here. +0.35 FIGHTS the differential
    # stride turn and drops the achieved yaw rate to 0.466 rad/s against a
    # commanded 0.8; 0.0 gives 0.839 (i.e. the same as rigid); -0.35 gives
    # 1.110 with better path tracking, and -0.70 reaches 1.313 while being the
    # only setting that turns and still advances. -0.35 is the balanced choice
    # and beats rigid on turn rate, cross-track error and forward progress
    # simultaneously.
    "turn_spine_gain": -0.35,
    "tail_yaw_gain": 0.0,
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
    print(summarise(payload["rows"], keys=(
        "peak_speed_mps", "net_progress_speed_mps", "turn_rate_mean_radps",
        "cost_of_transport", "cross_track_rms_m", "fell_over",
    )))
    print()
    print("peak_speed      m/s, max of a 0.1 s moving average of body-frame forward speed")
    print("net_progress    m/s, SIGNED net displacement along the start heading / time")
    print("turn_rate       rad/s, mean signed yaw rate")
    print("cost_transport  dimensionless, mechanical energy / (m g distance)")
    print("cross_track     m, RMS distance to the commanded path (speed-independent)")
    print("fell_over       fraction of runs whose trunk dropped below 0.18 m")
    return 0


def cmd_spine_sweep(args: argparse.Namespace) -> int:
    """
    Sweep spine undulation amplitude, including zero.

    The zero-amplitude row is the point of this command. It runs the model that
    HAS spine joints with the spine PD holding them at neutral, which separates
    two very different explanations for any spine-vs-rigid gap:

      * amplitude 0 matches rigid  -> the harm comes from DRIVING the spine,
        and some other amplitude or phase might help.
      * amplitude 0 is already worse than rigid -> the extra compliance and
        back-driven DOF cost you something merely by existing, and no amount of
        undulation tuning recovers it.

    Without this row, "spine is worse" is uninterpretable.
    """
    from cheetah.metrics import compute_metrics

    amps = [float(a) for a in args.amps.split(",")]
    seeds = tuple(int(s) for s in args.seeds.split(","))
    speeds = [float(v) for v in args.speeds.split(",")]

    spine_model, spine_info = build_model(spine=True, xml_path=args.xml)
    rigid_model, rigid_info = build_model(spine=False, xml_path=args.xml)

    rows = []
    print("=" * 96)
    print("SPINE AMPLITUDE SWEEP  (leg gait identical throughout, tuned on rigid)")
    print("=" * 96)
    print(f"amplitudes: {amps}")
    print(f"speeds    : {speeds}   seeds: {list(seeds)}\n")

    configs = [("rigid", None)] + [("spine", a) for a in amps]

    for vx in speeds:
        cmd = Command(vx=vx, yaw_rate=0.0)
        print(f"-- vx = {vx} m/s")
        print(f"   {'config':<22}{'net m/s':>16}{'peak':>16}{'CoT':>16}"
              f"{'fell':>10}{'pitch deg':>11}")
        for label, amp in configs:
            model = rigid_model if label == "rigid" else spine_model
            info = rigid_info if label == "rigid" else spine_info
            gait = dict(DEFAULT_GAIT)
            if amp is not None:
                gait["spine_pitch_amp"] = amp
                gait["spine_yaw_amp"] = amp * args.yaw_ratio
            per_seed = []
            for seed in seeds:
                p = GaitParams(**gait)
                c = CPGController(model, p, command=cmd)
                log = run_rollout(model, c, cmd, duration=args.duration,
                                  settle=0.5, variant=label, seed=seed)
                m = compute_metrics(log, info.total_mass)
                m["diverged"] = int(log.diverged)
                per_seed.append(m)
                rows.append({"vx": vx, "config": label,
                             "spine_amp_rad": amp if amp is not None else "",
                             "seed": seed, "diverged": int(log.diverged), **m})

            def agg(key):
                vals = [m[key] for m in per_seed
                        if not m["diverged"] and m[key] == m[key]]
                return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))

            name = "rigid (no joints)" if amp is None else (
                "spine amp=0 (held)" if amp == 0.0 else f"spine amp={amp:g}")
            n, ns = agg("net_progress_speed_mps")
            pk, pks = agg("peak_speed_mps")
            ct, cts = agg("cost_of_transport")
            fl, _ = agg("fell_over")
            pa, _ = agg("spine_pitch_amp_rad")
            pa_deg = np.rad2deg(pa) if pa == pa else float("nan")
            print(f"   {name:<22}{n:>9.3f}+-{ns:<5.3f}{pk:>9.3f}+-{pks:<5.3f}"
                  f"{ct:>9.3f}+-{cts:<5.3f}{fl:>10.2f}{pa_deg:>11.2f}")
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    print("\n'pitch deg' is the REALISED peak-to-peak spine pitch excursion, not the")
    print("commanded one. A large gap between them means the spine is being")
    print("back-driven by body loads rather than tracking its setpoint.")
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

    s = sub.add_parser("spine-sweep",
                       help="sweep spine amplitude incl. zero, vs the rigid baseline")
    s.add_argument("--amps", default="0.0,0.1,0.2,0.35,0.5",
                   help="comma-separated spine_pitch_amp values in rad")
    s.add_argument("--yaw-ratio", type=float, default=0.75,
                   help="spine_yaw_amp as a multiple of spine_pitch_amp")
    s.add_argument("--speeds", default="1.0,2.0")
    s.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    s.add_argument("--duration", type=float, default=6.0)
    s.add_argument("--out", default="results/spine_sweep.csv")
    s.set_defaults(func=cmd_spine_sweep)

    f = sub.add_parser("freefall", help="torque-matched reorientation test")
    f.add_argument("--duration", type=float, default=2.0)
    f.set_defaults(func=cmd_freefall)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
