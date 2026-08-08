"""
Open-loop CPG gait controller.

This exists so the harness can produce spine-vs-rigid numbers *before* any
policy is trained. It is a fixed, hand-written trot with optional spine
undulation, applied identically to both variants. It is deliberately not
optimised for either -- an untuned controller is a weak locomotion result but
an honest A/B, because neither variant gets to tune against it.

The asymmetric flexion waveform defined here is reused by the RL reward in
step 2, so the "phase-correct flexion" notion has exactly one definition.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import mujoco
import numpy as np

from .model import HOME_HIP_PITCH, HOME_KNEE, actuator_index, joint_qpos_index

LEGS = ("fl", "fr", "rl", "rr")

#: Trot: diagonal pairs move together. FL+RR at phase 0, FR+RL at phase 0.5.
TROT_OFFSETS = {"fl": 0.0, "fr": 0.5, "rl": 0.5, "rr": 0.0}
#: Bound: front pair together, rear pair together. This is the gait a sagittal
#: spine is supposed to help, so it matters that the harness can select it.
BOUND_OFFSETS = {"fl": 0.0, "fr": 0.0, "rl": 0.5, "rr": 0.5}

GAITS = {"trot": TROT_OFFSETS, "bound": BOUND_OFFSETS}


def asymmetric_wave(phase: np.ndarray | float, ratio: float = 2.0) -> np.ndarray | float:
    """
    Unit-amplitude oscillation whose negative (flexion) excursion is `ratio`
    times its positive (extension) excursion.

    A real cheetah's spine flexes roughly twice as far as it extends, so a
    symmetric sinusoid is the wrong prior. At ratio=2 this peaks at -1.0 in
    flexion and +0.5 in extension. At ratio=1 it degenerates to sin(), which
    is the control condition for testing whether the asymmetry matters at all.

    Continuous everywhere; the kink is only in the second derivative at the
    zero crossings, which the PD layer smooths out.
    """
    s = np.sin(2.0 * np.pi * np.asarray(phase, dtype=float))
    return np.where(s < 0.0, s, s / ratio)


@dataclass
class GaitParams:
    """Everything the open-loop controller needs. Serialised into results."""

    gait: str = "trot"
    freq: float = 2.0               # gait cycles per second
    hip_amp: float = 0.45           # rad, fore-aft hip swing
    knee_amp: float = 0.55          # rad, extra knee flexion during swing
    abduct_amp: float = 0.0         # rad, lateral

    spine_yaw_amp: float = 0.0      # rad, lateral undulation
    spine_pitch_amp: float = 0.0    # rad, sagittal flexion/extension
    spine_roll_amp: float = 0.0     # rad
    spine_phase: float = 0.0        # phase lead of spine over the front legs
    flexion_ratio: float = 2.0      # flexion : extension excursion
    flexion_sign: float = -1.0      # which sign of spine_pitch counts as flexion
    spine_freq_mult: float = 1.0    # sagittal spine often runs at 1x gait freq

    turn_rate: float = 0.0          # commanded yaw rate, rad/s
    turn_abduct_gain: float = 0.15  # differential abduction per rad/s
    turn_spine_gain: float = 0.35   # spine yaw bias per rad/s

    tail_yaw_gain: float = 0.0      # tail yaw bias per rad/s of turn command

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

    Targets are computed in joint space and converted to torque by PD, then
    clipped to each actuator's ctrlrange. Clipping is counted, because a
    controller that spends its life saturated is not the controller you think
    you specified.
    """

    def __init__(self, model: mujoco.MjModel, params: GaitParams) -> None:
        self.model = model
        self.p = params
        if params.gait not in GAITS:
            raise ValueError(f"unknown gait {params.gait!r}, expected one of {list(GAITS)}")
        self.offsets = GAITS[params.gait]

        self.act = actuator_index(model)
        self.qadr = joint_qpos_index(model)
        self.dadr = {
            name: int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in self.qadr
        }
        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

        # Which spine joints this variant actually has. The rigid variant has
        # none, and every spine term below simply never fires.
        self.has_spine = {
            j: (j in self.act) for j in ("spine_yaw", "spine_pitch", "spine_roll")
        }
        self.has_tail = "tail_yaw" in self.act

        self.clip_events = 0
        self.total_commands = 0

    # ---------------------------------------------------------------- targets

    def joint_targets(self, t: float) -> dict[str, float]:
        """Desired joint angles at time t."""
        p = self.p
        base_phase = p.freq * t
        tgt: dict[str, float] = {}

        for leg in LEGS:
            ph = (base_phase + self.offsets[leg]) % 1.0
            s = np.sin(2.0 * np.pi * ph)
            # Swing is the half-cycle where sin > 0: knee flexes, foot lifts.
            swing_lift = max(0.0, s)
            tgt[f"{leg}_hip_pitch"] = HOME_HIP_PITCH - p.hip_amp * np.cos(2.0 * np.pi * ph)
            tgt[f"{leg}_knee"] = HOME_KNEE - p.knee_amp * swing_lift

            side = 1.0 if leg in ("fl", "rl") else -1.0
            abd = p.abduct_amp * s * side
            # Turning: push the outside legs out, tuck the inside legs in.
            abd += p.turn_abduct_gain * p.turn_rate * side
            tgt[f"{leg}_abduct"] = abd

        if self.has_spine["spine_yaw"]:
            ph = (base_phase * p.spine_freq_mult + p.spine_phase) % 1.0
            und = p.spine_yaw_amp * np.sin(2.0 * np.pi * ph)
            tgt["spine_yaw"] = und + p.turn_spine_gain * p.turn_rate

        if self.has_spine["spine_pitch"]:
            ph = (base_phase * p.spine_freq_mult + p.spine_phase) % 1.0
            # The asymmetric part: flexion excursion is `flexion_ratio` times
            # extension. flexion_sign picks which direction is flexion.
            w = float(asymmetric_wave(ph, p.flexion_ratio))
            tgt["spine_pitch"] = p.flexion_sign * p.spine_pitch_amp * w

        if self.has_spine["spine_roll"]:
            ph = (base_phase * p.spine_freq_mult + p.spine_phase + 0.25) % 1.0
            tgt["spine_roll"] = p.spine_roll_amp * np.sin(2.0 * np.pi * ph)

        if self.has_tail and p.tail_yaw_gain:
            tgt["tail_yaw"] = p.tail_yaw_gain * p.turn_rate

        return tgt

    # ------------------------------------------------------------------ torque

    def __call__(self, data: mujoco.MjData, t: float) -> np.ndarray:
        """Compute the clipped torque vector for this instant."""
        p = self.p
        tgt = self.joint_targets(t)
        ctrl = np.zeros(self.model.nu)

        for name, q_des in tgt.items():
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

        clipped = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        self.clip_events += int(np.count_nonzero(~np.isclose(clipped, ctrl)))
        self.total_commands += self.model.nu
        return clipped

    def hold(self, data: mujoco.MjData) -> np.ndarray:
        """
        PD toward the neutral standing pose, no gait.

        Used during the settling window so the robot lands on its feet before
        the metrics window opens, instead of being measured mid-drop.
        """
        p = self.p
        ctrl = np.zeros(self.model.nu)
        targets = {f"{leg}_hip_pitch": HOME_HIP_PITCH for leg in LEGS}
        targets.update({f"{leg}_knee": HOME_KNEE for leg in LEGS})
        targets.update({f"{leg}_abduct": 0.0 for leg in LEGS})
        for j, present in self.has_spine.items():
            if present:
                targets[j] = 0.0
        for name, q_des in targets.items():
            ai = self.act.get(name)
            if ai is None:
                continue
            kp, kd = (p.kp_spine, p.kd_spine) if name.startswith("spine_") else (p.kp_leg, p.kd_leg)
            ctrl[ai] = kp * (q_des - data.qpos[self.qadr[name]]) - kd * data.qvel[self.dadr[name]]
        return np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

    @property
    def clip_fraction(self) -> float:
        """Fraction of actuator commands that hit their torque limit."""
        if not self.total_commands:
            return 0.0
        return self.clip_events / self.total_commands
