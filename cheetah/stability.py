"""
Numerical stability monitoring.

The failure this exists to prevent: a sim quietly diverges, the metrics code
happily averages over the garbage, and you report a peak speed of 40 m/s for
a robot that actually exploded at t=1.2 s. A diverged rollout must never
produce a number that looks like a result.

Policy here: any trip marks the rollout `diverged`, and every downstream
metric for that rollout is forced to NaN rather than computed. NaN propagates
and is visible in a CSV; a plausible-looking float is not.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import mujoco
import numpy as np

#: Nothing on this robot should ever move this fast (rad/s or m/s). The
#: random-torque shakedown peaks near 29, so this leaves a wide margin
#: before we call divergence.
MAX_ABS_QVEL = 500.0

#: The arena floor is 20 m half-width. Beyond this the robot has been
#: launched, not locomoted.
MAX_ABS_POS = 100.0


@dataclass
class StabilityReport:
    """Outcome of monitoring one rollout."""

    diverged: bool = False
    reasons: list[str] = field(default_factory=list)
    first_bad_step: int | None = None
    first_bad_time: float | None = None
    max_abs_qvel: float = 0.0
    max_abs_qacc: float = 0.0
    mujoco_warnings: dict[str, int] = field(default_factory=dict)
    steps_completed: int = 0

    @property
    def ok(self) -> bool:
        return not self.diverged

    def as_dict(self) -> dict:
        return {
            "diverged": self.diverged,
            "reasons": list(self.reasons),
            "first_bad_step": self.first_bad_step,
            "first_bad_time": self.first_bad_time,
            "max_abs_qvel": float(self.max_abs_qvel),
            "max_abs_qacc": float(self.max_abs_qacc),
            "mujoco_warnings": dict(self.mujoco_warnings),
            "steps_completed": self.steps_completed,
        }

    def summary(self) -> str:
        if not self.diverged:
            w = (
                f", mujoco warnings: {self.mujoco_warnings}"
                if self.mujoco_warnings
                else ""
            )
            return (
                f"stable over {self.steps_completed} steps "
                f"(max|qvel|={self.max_abs_qvel:.2f}, max|qacc|={self.max_abs_qacc:.3g}){w}"
            )
        return (
            f"DIVERGED at step {self.first_bad_step} (t={self.first_bad_time:.4f}s): "
            + "; ".join(self.reasons)
        )


class StabilityMonitor:
    """
    Step-by-step divergence detector.

    Usage:
        mon = StabilityMonitor(model, data)
        for i in range(n):
            mujoco.mj_step(model, data)
            if not mon.check(i):
                break
        report = mon.report()
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        max_abs_qvel: float = MAX_ABS_QVEL,
        max_abs_pos: float = MAX_ABS_POS,
    ) -> None:
        self.model = model
        self.data = data
        self.max_abs_qvel = max_abs_qvel
        self.max_abs_pos = max_abs_pos
        self._rep = StabilityReport()
        # MuJoCo accumulates warning counts in data.warning; snapshot the
        # baseline so we only report warnings raised during *this* rollout.
        self._warn0 = np.array(data.warning.number, dtype=np.int64).copy()

    def check(self, step: int) -> bool:
        """Inspect state after one mj_step. Returns False once diverged."""
        rep = self._rep
        if rep.diverged:
            return False
        rep.steps_completed = step + 1
        d = self.data
        dt = self.model.opt.timestep

        reasons: list[str] = []

        qpos = np.asarray(d.qpos)
        qvel = np.asarray(d.qvel)
        qacc = np.asarray(d.qacc)

        if not np.isfinite(qpos).all():
            reasons.append("non-finite qpos")
        if not np.isfinite(qvel).all():
            reasons.append("non-finite qvel")
        if not np.isfinite(qacc).all():
            reasons.append("non-finite qacc")

        if not reasons:
            # Only meaningful to take maxima of finite arrays.
            v = float(np.abs(qvel).max()) if qvel.size else 0.0
            a = float(np.abs(qacc).max()) if qacc.size else 0.0
            rep.max_abs_qvel = max(rep.max_abs_qvel, v)
            rep.max_abs_qacc = max(rep.max_abs_qacc, a)
            if v > self.max_abs_qvel:
                reasons.append(f"|qvel|={v:.3g} exceeds {self.max_abs_qvel:g}")
            p = float(np.abs(qpos[:3]).max()) if qpos.size >= 3 else 0.0
            if p > self.max_abs_pos:
                reasons.append(f"|root pos|={p:.3g} m exceeds {self.max_abs_pos:g}")

        # MuJoCo's own solver complaints. mjWARN_BADQACC in particular means
        # the integrator produced a bad acceleration and reset it -- the state
        # is no longer the physics you asked for.
        wnow = np.array(d.warning.number, dtype=np.int64)
        delta = wnow - self._warn0
        for k in range(len(delta)):
            if delta[k] > 0:
                nm = mujoco.mjtWarning(k).name
                rep.mujoco_warnings[nm] = int(delta[k])
                if nm in ("mjWARN_BADQACC", "mjWARN_BADQVEL", "mjWARN_BADQPOS"):
                    reasons.append(f"{nm} x{int(delta[k])}")

        if reasons:
            rep.diverged = True
            rep.reasons = reasons
            rep.first_bad_step = step
            rep.first_bad_time = step * dt
            return False
        return True

    def report(self) -> StabilityReport:
        return self._rep


def warn_loudly(label: str, report: StabilityReport) -> None:
    """
    Print an unmissable banner for a diverged rollout, and raise a Python
    warning so it also shows up in captured logs and test output.
    """
    if report.ok:
        return
    bar = "!" * 78
    msg = (
        f"\n{bar}\n"
        f"!!  NUMERICAL DIVERGENCE: {label}\n"
        f"!!  {report.summary()}\n"
        f"!!  All metrics for this rollout are reported as NaN. Do not use them.\n"
        f"{bar}\n"
    )
    print(msg, flush=True)
    warnings.warn(f"diverged: {label}: {report.summary()}", RuntimeWarning, stacklevel=2)


def check_model_sanity(model: mujoco.MjModel, label: str = "") -> list[str]:
    """
    Static checks on a freshly compiled model, before anything is simulated.

    Catches the classic own-goals: a timestep too large for the stiffest
    joint, non-positive masses, degenerate inertias.
    """
    issues: list[str] = []
    dt = model.opt.timestep

    # Explicit stability bound for each spring-loaded joint: dt < 2*sqrt(I/k).
    #
    # Two corrections over the naive version. First, the effective inertia is
    # the joint's OWN armature plus the inertia of the subtree it swings, not
    # the global minimum armature -- pairing a stiff spine spring with a leg's
    # armature understates the bound by orders of magnitude and produces a
    # warning on every healthy run. Second, implicit and implicitfast integrate
    # joint stiffness and damping implicitly, so the explicit bound does not
    # apply; there we only flag the far looser case of a spring period shorter
    # than a single timestep, which no integrator can resolve.
    implicit = model.opt.integrator in (
        mujoco.mjtIntegrator.mjINT_IMPLICIT,
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    )
    for j in range(model.njnt):
        k = float(model.jnt_stiffness[j])
        if k <= 0:
            continue
        dofadr = int(model.jnt_dofadr[j])
        bid = int(model.jnt_bodyid[j])
        # Subtree inertia about the joint, roughly: subtree mass times the
        # square of its extent. body_inertia's largest principal component is a
        # serviceable stand-in and errs on the small side, so the bound is
        # conservative.
        inertia = float(np.max(model.body_inertia[bid])) if bid < model.nbody else 0.0
        subtree = float(model.body_subtreemass[bid]) if bid < model.nbody else 0.0
        i_eff = max(float(model.dof_armature[dofadr]) + inertia + 0.01 * subtree, 1e-9)
        bound = 2.0 * np.sqrt(i_eff / k)
        limit = bound if not implicit else bound / 10.0
        if dt > limit:
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
            issues.append(
                f"timestep {dt:g}s vs spring period bound {bound:.4g}s on {nm!r} "
                f"(k={k:g}, I_eff~{i_eff:.4g}"
                + (", implicit integrator" if implicit else "") + ")"
            )

    bad_mass = np.where(model.body_mass[1:] <= 0)[0]
    if bad_mass.size:
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i) + 1)
            for i in bad_mass
        ]
        issues.append(f"non-positive body mass: {names}")

    if np.any(model.body_inertia[1:] <= 0):
        rows = np.where((model.body_inertia[1:] <= 0).any(axis=1))[0]
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i) + 1) for i in rows
        ]
        issues.append(f"non-positive principal inertia: {names}")

    if issues and label:
        print(f"[model sanity] {label}: " + "; ".join(issues), flush=True)
    return issues
