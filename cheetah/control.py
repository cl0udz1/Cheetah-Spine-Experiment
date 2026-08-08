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
    Unit-amplitude oscillation whose negative (flexion) excursion is `ratio`
    times its positive (extension) excursion.

    A real cheetah's spine flexes roughly twice as far as it extends, so a
    symmetric sinusoid is the wrong prior. At ratio=2 this peaks at -1.0 in
    flexion and +0.5 in extension. At ratio=1 it degenerates to sin(), which
    is the control condition for testing whether the asymmetry matters at all.
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
    flexion_ratio: float = 2.0   # flexion : extension excursion
    flexion_sign: float = -1.0   # which sign of spine_pitch counts as flexion
    spine_freq_mult: float = 1.0 # sagittal spine often runs at 1x gait freq

    #: Differential stride between left and right legs, as a fraction of
    #: nominal stride amplitude traded per rad/s of commanded yaw. Deliberately
    #: NOT derived from the physical track width: at a 0.18 m track that yields
    #: a 7% stride difference, which produces no measurable yaw on point feet.
    turn_stride_gain: float = 0.6
    turn_spine_gain: float = 0.35   # spine yaw bias per rad/s of turn command
    turn_abduct_gain: float = 0.0   # differential abduction per rad/s
    tail_yaw_gain: float = 0.0      # tail yaw bias per rad/s

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

        # Per-leg stride amplitude from the speed and turn command. Positive
        # yaw_rate is a left turn (counter-clockwise about +z), so the LEFT
        # legs must shorten their stride and the right legs lengthen it.
        a_nom = self.vx / (2.0 * LEG_REACH * max(params.freq, 1e-6))
        self.leg_amp = {}
        for leg in LEGS:
            side = 1.0 if leg in LEFT_LEGS else -1.0
            scale = 1.0 - side * params.turn_stride_gain * self.yaw_rate
            a = a_nom * scale
            self.leg_amp[leg] = float(np.clip(a, -params.hip_amp_max, params.hip_amp_max))

        # Which spine joints this variant actually has. The rigid variant has
        # none, and every spine term below simply never fires.
        self.has_spine = {
            j: (j in self.act) for j in ("spine_yaw", "spine_pitch", "spine_roll")
        }
        self.has_tail = "tail_yaw" in self.act

        self.clip_events = 0
        self.total_commands = 0

    # ---------------------------------------------------------------- targets

    def _leg_targets(self, leg: str, phase: float, ramp: float) -> tuple[float, float]:
        """(hip_pitch, knee) targets for one leg at its own phase."""
        p = self.p
        beta = min(max(p.duty, 0.05), 0.95)
        A = self.leg_amp[leg] * ramp
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
        """Desired joint angles at time t."""
        p = self.p
        ramp = smoothstep(t / p.ramp_time) if p.ramp_time > 0 else 1.0
        base_phase = p.freq * t
        tgt: dict[str, float] = {}

        for leg in LEGS:
            ph = (base_phase + self.offsets[leg]) % 1.0
            hip, knee = self._leg_targets(leg, ph, ramp)
            tgt[f"{leg}_hip_pitch"] = hip
            tgt[f"{leg}_knee"] = knee
            side = 1.0 if leg in LEFT_LEGS else -1.0
            tgt[f"{leg}_abduct"] = p.turn_abduct_gain * self.yaw_rate * side * ramp

        if self.has_spine["spine_yaw"]:
            ph = (base_phase * p.spine_freq_mult + p.spine_phase) % 1.0
            und = p.spine_yaw_amp * np.sin(2.0 * np.pi * ph)
            tgt["spine_yaw"] = (und + p.turn_spine_gain * self.yaw_rate) * ramp

        if self.has_spine["spine_pitch"]:
            ph = (base_phase * p.spine_freq_mult + p.spine_phase) % 1.0
            # The asymmetric part: flexion excursion is `flexion_ratio` times
            # extension. flexion_sign picks which direction counts as flexion.
            w = float(asymmetric_wave(ph, p.flexion_ratio))
            tgt["spine_pitch"] = p.flexion_sign * p.spine_pitch_amp * w * ramp

        if self.has_spine["spine_roll"]:
            ph = (base_phase * p.spine_freq_mult + p.spine_phase + 0.25) % 1.0
            tgt["spine_roll"] = p.spine_roll_amp * np.sin(2.0 * np.pi * ph) * ramp

        if self.has_tail and p.tail_yaw_gain:
            tgt["tail_yaw"] = p.tail_yaw_gain * self.yaw_rate * ramp

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
        """Clipped torque vector for this instant."""
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
        return np.clip(self._pd(data, targets), self.ctrl_lo, self.ctrl_hi)

    @property
    def clip_fraction(self) -> float:
        """Fraction of actuator commands that hit their torque limit."""
        if not self.total_commands:
            return 0.0
        return self.clip_events / self.total_commands
