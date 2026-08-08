"""
Open-loop CPG gait controller.

This exists so the harness can produce spine-vs-rigid numbers *before* any
policy is trained. It is a fixed, hand-written gait applied identically to
both variants.

Structure per leg is an explicit stance/swing split with a duty factor, not a
raw sinusoid: during stance the foot is planted and sweeps backward (which is
what propels the body), during swing the knee flexes to clear the ground and
the hip returns. Amplitudes ramp in smoothly from the settled stance -- a step
discontinuity at t=0 saturates every actuator and throws the robot before it
takes a stride.

The asymmetric flexion waveform defined here is reused by the RL reward in
step 2, so "phase-correct flexion" has exactly one definition in the codebase.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .model import HOME_HIP_PITCH, HOME_KNEE, actuator_index, joint_qpos_index

LEGS = ("fl", "fr", "rl", "rr")
LEFT_LEGS = ("fl", "rl")

#: Trot: diagonal pairs move together. FL+RR at phase 0, FR+RL at phase 0.5.
TROT_OFFSETS = {"fl": 0.0, "fr": 0.5, "rl": 0.5, "rr": 0.0}
#: Bound: front pair together, rear pair together. The gait a sagittal spine is
#: supposed to help, so the harness must be able to select it.
BOUND_OFFSETS = {"fl": 0.0, "fr": 0.0, "rl": 0.5, "rr": 0.5}
#: Walk: one foot at a time, maximum static stability.
WALK_OFFSETS = {"fl": 0.0, "fr": 0.5, "rl": 0.25, "rr": 0.75}

GAITS = {"trot": TROT_OFFSETS, "bound": BOUND_OFFSETS, "walk": WALK_OFFSETS}

#: Hip-to-foot distance in the home stance, used for the stride/speed mapping.
LEG_REACH = 0.363
#: Lateral distance between the left and right foot lines.
TRACK_WIDTH = 0.18


def asymmetric_wave(phase, ratio: float = 2.0):
    """
    Unit-amplitude oscillation whose negative excursion is `ratio` times its
    positive excursion. This is the spine_pitch target shape directly, with no
    sign flip applied anywhere downstream.

    SIGN CONVENTION, verified against the model rather than assumed:
    setting spine_pitch to -0.4 rad puts the tail base at z=0.297 and +0.4 rad
    puts it at z=0.516, against 0.408 at neutral. Negative spine_pitch lowers
    the hind end, which is the gathered phase -- so NEGATIVE spine_pitch is
    FLEXION and positive is extension. metrics.py uses the same convention.

    A real cheetah's spine flexes roughly twice as far as it extends, so:
      ratio = 2.0  flexion-dominant, the biological case (peaks -1.0 / +0.5)
      ratio = 1.0  symmetric sin(), the control for whether asymmetry matters
      ratio = 0.5  extension-dominant, the anti-biological control

    The `ratio` parameter spans both directions, so no separate sign flag is
    needed. An earlier version had a `flexion_sign` multiplier on top of this,
    which at its shipped value of -1.0 inverted the waveform and commanded an
    extension-dominant spine while the parameter name claimed otherwise. It was
    also redundant: negating this wave is identical to inverting `ratio` and
    shifting `phase` by 0.5, and phase is already swept.
    """
    s = np.sin(2.0 * np.pi * np.asarray(phase, dtype=float))
    return np.where(s < 0.0, s, s / ratio)


def smoothstep(x: float) -> float:
    """C1 ramp from 0 to 1 on [0,1]. Clamped outside."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class GaitParams:
    """Everything the open-loop controller needs. Serialised into results."""

    gait: str = "trot"
    freq: float = 2.0            # gait cycles per second
    duty: float = 0.6            # fraction of the cycle a foot is in stance
    hip_amp_max: float = 0.50    # rad, cap on half the stance sweep
    knee_amp: float = 0.55       # rad, extra knee flexion during swing
    ramp_time: float = 0.6       # s, amplitude fade-in from the settled stance

    spine_yaw_amp: float = 0.0   # rad, lateral undulation
    spine_pitch_amp: float = 0.0 # rad, sagittal flexion/extension
    spine_roll_amp: float = 0.0  # rad
    spine_phase: float = 0.0     # phase lead of spine over the front legs
    #: Flexion:extension excursion ratio of the sagittal spine. 2.0 is the
    #: biological case, 1.0 is a symmetric sinusoid, 0.5 runs the asymmetry
    #: backwards. Flexion is NEGATIVE spine_pitch -- see asymmetric_wave().
    flexion_ratio: float = 2.0
    spine_freq_mult: float = 1.0 # sagittal spine often runs at 1x gait freq

    #: Differential stride between left and right legs, as a fraction of
    #: nominal stride amplitude traded per rad/s of commanded yaw. Deliberately
    #: NOT derived from the physical track width: at a 0.18 m track that yields
    #: a 7% stride difference, which produces no measurable yaw on point feet.
    turn_stride_gain: float = 0.6
    turn_spine_gain: float = 0.35   # spine yaw bias per rad/s of turn command
    turn_abduct_gain: float = 0.0   # differential abduction per rad/s
    tail_yaw_gain: float = 0.0      # tail yaw bias per rad/s

    #: Closed-loop tracking. With these off the controller is purely open-loop:
    #: it has no idea how fast it is going or which way it is pointing, so the
    #: commanded speed is a label rather than a target and a startup yaw impulse
    #: is never corrected. See closed_loop_speed / closed_loop_heading.
    closed_loop_speed: bool = False
    closed_loop_heading: bool = False
    kp_speed: float = 0.18          # fractional freq change per (m/s) of error
    ki_speed: float = 0.06          # per (m/s . s)
    freq_min: float = 0.5           # multiples of nominal freq
    freq_max: float = 1.8
    kp_heading: float = 0.8         # rad/s of yaw correction per rad of error
    kd_heading: float = 0.12        # per (rad/s) of yaw rate error
    #: Cross-track feedback, rad/s of yaw correction per metre of lateral
    #: offset from the reference path. Heading feedback alone cannot fix
    #: cross-track error: driving yaw to the reference heading just makes the
    #: robot run PARALLEL to the path at whatever offset it already had. Only
    #: a position term steers it back onto the line.
    kp_lateral: float = 0.9
    #: Geometric stride->speed mapping calibration. The naive model
    #: v = 2*A*LEG_REACH*freq assumes a planted foot with no slip and no body
    #: pitch; measured throughput is ~1.9x that, so without this the commanded
    #: speed is wrong by nearly 2x and the closed loop spends its whole range
    #: correcting a modelling error instead of a disturbance.
    stride_efficiency: float = 1.9
    #: Clamp on the heading loop's output. This is NOT a free parameter: the
    #: correction is fed through turn_abduct_gain into an abduction joint
    #: limited to +-0.7 rad, and through turn_stride_gain into a stride scale
    #: of (1 -+ gain*yaw). At gain 0.5/0.4 a clamp of 2.5 commands 1.25 rad of
    #: abduction (impossible) and drives one side's stride to zero (a fall).
    #: 0.6 keeps both terms inside their physical range.
    max_yaw_correction: float = 0.6
    #: Hold the loops open until the gait has ramped in. Integrating error
    #: against a robot that has not started walking yet is pure windup.
    feedback_start: float = 1.0

    #: Hold the tail at neutral. The tail has actuators; leaving it free was an
    #: oversight, and a free 0.3 kg weighted tail contributes ~30% of the
    #: uncommanded yaw drift and essentially all of the late-time drift rate.
    #: (uses kp_tail / kd_tail below for the hold)
    hold_tail: bool = True

    kp_leg: float = 80.0
    kd_leg: float = 2.0
    kp_spine: float = 120.0
    kd_spine: float = 4.0
    kp_tail: float = 20.0
    kd_tail: float = 1.0

    def as_dict(self) -> dict:
        return asdict(self)


class CPGController:
    """
    Phase-driven joint-target generator with a PD torque layer.

    The commanded body velocity sets the stride amplitude geometrically:
    a foot planted through a stance sweep of 2A rad carries the body
    2*A*LEG_REACH per cycle, so A = vx / (2 * LEG_REACH * freq). Turning uses
    differential stride between the left and right legs, which is the mechanism
    a legged robot actually turns with; any spine yaw contribution then shows up
    as turning authority *on top* of that, which is the thing worth measuring.
    """

    def __init__(self, model: mujoco.MjModel, params: GaitParams, command=None) -> None:
        self.model = model
        self.p = params
        if params.gait not in GAITS:
            raise ValueError(f"unknown gait {params.gait!r}, expected one of {list(GAITS)}")
        self.offsets = GAITS[params.gait]

        self.vx = float(getattr(command, "vx", 1.0)) if command is not None else 1.0
        self.yaw_rate = float(getattr(command, "yaw_rate", 0.0)) if command is not None else 0.0

        self.act = actuator_index(model)
        self.qadr = joint_qpos_index(model)
        self.dadr = {
            name: int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in self.qadr
        }
        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

        # Nominal stride amplitude for the commanded speed. Per-leg amplitudes
        # are derived from this each step, because with a heading loop the
        # effective yaw rate changes continuously.
        self.a_nom = self.vx / (
            2.0 * LEG_REACH * max(params.freq, 1e-6) * max(params.stride_efficiency, 1e-6)
        )

        # Which spine joints this variant actually has. The rigid and passive
        # variants have no spine ACTUATORS, so every spine term below simply
        # never fires -- for passive, the spring does the work instead.
        self.has_spine = {
            j: (j in self.act) for j in ("spine_yaw", "spine_pitch", "spine_roll")
        }
        self.has_tail = "tail_yaw" in self.act

        # Gait phase is integrated rather than computed as freq*t, because the
        # speed loop varies the frequency and freq(t)*t is not the phase of a
        # frequency-modulated oscillator.
        self.phase = 0.0
        self._last_t: float | None = None
        self.freq_scale = 1.0
        self._speed_integral = 0.0
        self.yaw_ref = 0.0
        self._yaw_ref_init = False
        self.yaw_rate_eff = self.yaw_rate
        #: Reference path, integrated internally so the controller can steer
        #: back onto it rather than merely parallel to it.
        self.ref_pos = np.zeros(2)
        self.cross_track = 0.0

        self.clip_events = 0
        self.total_commands = 0

    # ------------------------------------------------------------- estimation

    @staticmethod
    def _yaw_of(quat) -> float:
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, quat)
        return float(np.arctan2(m[3], m[0]))

    def _measure(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(forward speed, yaw, yaw rate) in world/body terms."""
        yaw = self._yaw_of(data.qpos[3:7])
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        v_fwd = float(np.dot(data.qvel[0:2], heading))
        # For a freejoint MuJoCo stores angular velocity in the BODY frame, so
        # the world yaw rate is the body rate rotated back out. For moderate
        # roll/pitch the z component dominates and this is close enough for a
        # feedback term; it is never used as a reported metric.
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, data.qpos[3:7])
        yaw_rate = float(m.reshape(3, 3)[2] @ data.qvel[3:6])
        return v_fwd, yaw, yaw_rate

    def _update_feedback(self, data: mujoco.MjData, dt: float, t: float) -> None:
        p = self.p
        if dt <= 0.0:
            return
        v_fwd, yaw, yaw_rate = self._measure(data)

        # Seed the heading reference from the robot's actual initial heading,
        # so a randomised start is not read as a tracking error to fight.
        if not self._yaw_ref_init:
            self.yaw_ref = yaw
            self.ref_pos = np.array(data.qpos[0:2], dtype=float)
            self._yaw_ref_init = True

        active = t >= p.feedback_start

        if p.closed_loop_speed and active:
            err = self.vx - v_fwd
            u_unsat = p.kp_speed * err + p.ki_speed * self._speed_integral
            scale_unsat = 1.0 + u_unsat
            # Conditional integration: stop accumulating when the frequency is
            # already at a limit and the error would push it further out. A
            # plain magnitude clamp on the integral does not prevent this --
            # the robot sits at freq_max with a saturated integral and thrashes.
            saturated_high = scale_unsat >= p.freq_max and err > 0
            saturated_low = scale_unsat <= p.freq_min and err < 0
            if not (saturated_high or saturated_low):
                self._speed_integral += err * dt
            u = p.kp_speed * err + p.ki_speed * self._speed_integral
            self.freq_scale = float(np.clip(1.0 + u, p.freq_min, p.freq_max))

        # Advance the reference pose. Both integrate the COMMANDED motion, so a
        # turn command is tracked rather than fought.
        self.ref_pos = self.ref_pos + self.vx * dt * np.array(
            [np.cos(self.yaw_ref), np.sin(self.yaw_ref)]
        )
        self.yaw_ref += self.yaw_rate * dt

        # Signed lateral offset from the reference path, expressed in the
        # reference frame. Positive means the robot is to the LEFT of the path.
        d = np.asarray(data.qpos[0:2], dtype=float) - self.ref_pos
        self.cross_track = float(-np.sin(self.yaw_ref) * d[0] + np.cos(self.yaw_ref) * d[1])

        if p.closed_loop_heading and active:
            err = (self.yaw_ref - yaw + np.pi) % (2 * np.pi) - np.pi
            corr = (
                p.kp_heading * err
                + p.kd_heading * (self.yaw_rate - yaw_rate)
                # Left of the path -> steer right, hence the minus sign.
                - p.kp_lateral * self.cross_track
            )
            corr = float(np.clip(corr, -p.max_yaw_correction, p.max_yaw_correction))
            self.yaw_rate_eff = self.yaw_rate + corr
        else:
            self.yaw_rate_eff = self.yaw_rate

    def _leg_amp(self, leg: str) -> float:
        """Stride amplitude for one leg under the current effective yaw rate."""
        side = 1.0 if leg in LEFT_LEGS else -1.0
        scale = 1.0 - side * self.p.turn_stride_gain * self.yaw_rate_eff
        return float(np.clip(self.a_nom * scale,
                             -self.p.hip_amp_max, self.p.hip_amp_max))

    # ---------------------------------------------------------------- targets

    def _leg_targets(self, leg: str, phase: float, ramp: float) -> tuple[float, float]:
        """(hip_pitch, knee) targets for one leg at its own phase."""
        p = self.p
        beta = min(max(p.duty, 0.05), 0.95)
        A = self._leg_amp(leg) * ramp
        K = p.knee_amp * ramp

        # Sign convention: hip_pitch rotates about +y, and R_y(theta) maps the
        # downward leg vector (0,0,-L) to x = -L*sin(theta). So INCREASING
        # hip_pitch drives the foot toward -x. To push the body toward +x the
        # planted foot must travel toward -x, i.e. hip_pitch must increase
        # through stance. Getting this backwards produces a gait that runs
        # smoothly in reverse, which the unsigned distance metrics happily
        # score as success.
        if phase < beta:
            # Stance: foot planted, hip sweeps -A -> +A, body is carried forward.
            u = phase / beta
            hip = HOME_HIP_PITCH - A * np.cos(np.pi * u)
            knee = HOME_KNEE
        else:
            # Swing: hip returns +A -> -A while the knee flexes to clear ground.
            u = (phase - beta) / (1.0 - beta)
            hip = HOME_HIP_PITCH + A * np.cos(np.pi * u)
            knee = HOME_KNEE - K * np.sin(np.pi * u)
        return float(hip), float(knee)

    def joint_targets(self, t: float) -> dict[str, float]:
        """
        Desired joint angles now.

        Uses the integrated gait phase, not freq*t: under the speed loop the
        frequency varies, and freq(t)*t is not the phase of a frequency-
        modulated oscillator (it double-counts every past frequency change).
        """
        p = self.p
        ramp = smoothstep(t / p.ramp_time) if p.ramp_time > 0 else 1.0
        base_phase = self.phase
        tgt: dict[str, float] = {}

        for leg in LEGS:
            ph = (base_phase + self.offsets[leg]) % 1.0
            hip, knee = self._leg_targets(leg, ph, ramp)
            tgt[f"{leg}_hip_pitch"] = hip
            tgt[f"{leg}_knee"] = knee
            side = 1.0 if leg in LEFT_LEGS else -1.0
            tgt[f"{leg}_abduct"] = p.turn_abduct_gain * self.yaw_rate_eff * side * ramp

        if self.has_spine["spine_yaw"]:
            ph = (base_phase * p.spine_freq_mult + p.spine_phase) % 1.0
            und = p.spine_yaw_amp * np.sin(2.0 * np.pi * ph)
            # Spine steering follows the COMMANDED yaw rate, not the effective
            # one. Routing the heading loop's correction through the spine too
            # double-counts it: the legs already act on that correction, and at
            # 2 m/s the combination over-steers and the robot falls on every
            # seed -- even on a straight command, where the correction is only
            # supposed to be a small trim. The spine articulates for intended
            # turns; the legs handle heading trim.
            tgt["spine_yaw"] = (und + p.turn_spine_gain * self.yaw_rate) * ramp

        if self.has_spine["spine_pitch"]:
            ph = (base_phase * p.spine_freq_mult + p.spine_phase) % 1.0
            # asymmetric_wave IS the target shape: negative is flexion. No sign
            # flip here -- one existed, defaulted to -1.0, and silently ran the
            # asymmetry backwards for the whole first phase of this study.
            w = float(asymmetric_wave(ph, p.flexion_ratio))
            tgt["spine_pitch"] = p.spine_pitch_amp * w * ramp

        if self.has_spine["spine_roll"]:
            ph = (base_phase * p.spine_freq_mult + p.spine_phase + 0.25) % 1.0
            tgt["spine_roll"] = p.spine_roll_amp * np.sin(2.0 * np.pi * ph) * ramp

        if self.has_tail:
            if p.tail_yaw_gain:
                tgt["tail_yaw"] = p.tail_yaw_gain * self.yaw_rate_eff * ramp
            elif p.hold_tail:
                # Actively held rather than left swinging. See hold_tail.
                tgt["tail_yaw"] = 0.0
            if p.hold_tail:
                tgt["tail_pitch"] = 0.0

        return tgt

    # ------------------------------------------------------------------ torque

    def _pd(self, data: mujoco.MjData, targets: dict[str, float]) -> np.ndarray:
        p = self.p
        ctrl = np.zeros(self.model.nu)
        for name, q_des in targets.items():
            ai = self.act.get(name)
            if ai is None:
                continue  # joint absent in this variant (rigid trunk)
            if name.startswith("spine_"):
                kp, kd = p.kp_spine, p.kd_spine
            elif name.startswith("tail_"):
                kp, kd = p.kp_tail, p.kd_tail
            else:
                kp, kd = p.kp_leg, p.kd_leg
            q = data.qpos[self.qadr[name]]
            v = data.qvel[self.dadr[name]]
            ctrl[ai] = kp * (q_des - q) - kd * v
        return ctrl

    def __call__(self, data: mujoco.MjData, t: float) -> np.ndarray:
        """Clipped torque vector for this instant. Advances the gait phase."""
        dt = self.model.opt.timestep if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        self._update_feedback(data, dt, t)
        self.phase = (self.phase + self.p.freq * self.freq_scale * dt) % 1.0

        ctrl = self._pd(data, self.joint_targets(t))
        clipped = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        self.clip_events += int(np.count_nonzero(~np.isclose(clipped, ctrl)))
        self.total_commands += self.model.nu
        return clipped

    def hold(self, data: mujoco.MjData) -> np.ndarray:
        """
        PD toward the neutral standing pose, no gait.

        Used during settling so the robot is measured while locomoting rather
        than mid-drop. Torque here is deliberately not counted toward the
        clip statistics, since it is not part of the gait.
        """
        targets = {f"{leg}_hip_pitch": HOME_HIP_PITCH for leg in LEGS}
        targets.update({f"{leg}_knee": HOME_KNEE for leg in LEGS})
        targets.update({f"{leg}_abduct": 0.0 for leg in LEGS})
        for j, present in self.has_spine.items():
            if present:
                targets[j] = 0.0
        if self.has_tail and self.p.hold_tail:
            targets["tail_yaw"] = 0.0
            targets["tail_pitch"] = 0.0
        return np.clip(self._pd(data, targets), self.ctrl_lo, self.ctrl_hi)

    @property
    def clip_fraction(self) -> float:
        """Fraction of actuator commands that hit their torque limit."""
        if not self.total_commands:
            return 0.0
        return self.clip_events / self.total_commands
