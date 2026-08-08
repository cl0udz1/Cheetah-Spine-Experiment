"""
The simulation loop.

One rollout = one variant, one controller, one command, for a fixed duration.
Everything downstream (metrics, plots, video) reads the RolloutLog this
produces, so there is exactly one place where physics is stepped and exactly
one place where divergence is detected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .model import foot_geom_ids, set_home_pose
from .render import Recorder
from .stability import StabilityMonitor, StabilityReport, warn_loudly

LEGS = ("fl", "fr", "rl", "rr")


@dataclass
class Command:
    """What the robot was asked to do. Also defines the reference path."""

    vx: float = 1.0        # desired forward speed, m/s
    yaw_rate: float = 0.0  # desired turn rate, rad/s

    def as_dict(self) -> dict:
        return {"cmd_vx": self.vx, "cmd_yaw_rate": self.yaw_rate}


@dataclass
class RolloutLog:
    """Per-step time series. All arrays share axis 0 = time."""

    dt: float
    t: np.ndarray
    pos: np.ndarray            # (N,3) world position of the trunk
    quat: np.ndarray           # (N,4)
    yaw: np.ndarray            # (N,) unwrapped
    pitch: np.ndarray
    roll: np.ndarray
    linvel_world: np.ndarray   # (N,3)
    fwd_speed: np.ndarray      # (N,) velocity projected on the body heading
    yaw_rate: np.ndarray       # (N,)
    force: np.ndarray          # (N,nu) actuator force
    jvel: np.ndarray           # (N,nu) velocity of each actuated joint
    power: np.ndarray          # (N,) sum |tau * omega|
    contacts: np.ndarray       # (N,4) bool, order = LEGS
    spine_angles: np.ndarray   # (N,3) NaN where the joint does not exist
    ref_pos: np.ndarray        # (N,2) commanded path
    ref_yaw: np.ndarray        # (N,)
    command: Command
    stability: StabilityReport
    variant: str = ""
    actuator_names: tuple[str, ...] = ()
    clip_fraction: float = 0.0
    frames: list = field(default_factory=list)

    @property
    def diverged(self) -> bool:
        return self.stability.diverged

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0


SPINE_ORDER = ("spine_yaw", "spine_pitch", "spine_roll")


def _rpy(quat: np.ndarray) -> tuple[float, float, float]:
    """Roll/pitch/yaw from a wxyz quaternion, via the rotation matrix."""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, quat)
    m = m.reshape(3, 3)
    yaw = float(np.arctan2(m[1, 0], m[0, 0]))
    pitch = float(np.arcsin(np.clip(-m[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(m[2, 1], m[2, 2]))
    return roll, pitch, yaw


def run_rollout(
    model: mujoco.MjModel,
    controller,
    command: Command,
    duration: float = 6.0,
    settle: float = 0.5,
    variant: str = "",
    recorder: Recorder | None = None,
    seed: int = 0,
) -> RolloutLog:
    """
    Simulate one episode.

    Args:
        settle: seconds of zero-command settling before logging starts, so the
            robot drops onto its feet and the metrics window contains
            locomotion rather than the initial transient.

    Divergence stops the loop immediately; the log is truncated to whatever
    was valid, and every metric computed from it will be NaN.
    """
    data = mujoco.MjData(model)
    set_home_pose(model, data)

    dt = model.opt.timestep
    n_settle = int(round(settle / dt))
    n_steps = int(round(duration / dt))

    feet = foot_geom_ids(model)
    foot_gids = np.array([feet[l] for l in LEGS])
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # Map each actuator to the dof of the joint it drives, for power accounting.
    act_dof = np.full(model.nu, -1, dtype=int)
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            jid = int(model.actuator_trnid[i, 0])
            act_dof[i] = int(model.jnt_dofadr[jid])

    act_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    )
    spine_qadr = []
    for nm in SPINE_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        spine_qadr.append(int(model.jnt_qposadr[jid]) if jid >= 0 else -1)

    monitor = StabilityMonitor(model, data)

    # --- settle: hold the home pose with the controller's PD, no gait ---------
    for i in range(n_settle):
        if hasattr(controller, "hold"):
            data.ctrl[:] = controller.hold(data)
        else:
            data.ctrl[:] = controller(data, 0.0)
        mujoco.mj_step(model, data)
        if not monitor.check(i):
            warn_loudly(f"{variant} (settling)", monitor.report())
            break

    rep = monitor.report()

    # --- logged phase --------------------------------------------------------
    T, P, Q, LV, F, JV, C, SA = [], [], [], [], [], [], [], []
    capture_every = recorder.steps_per_frame() if recorder is not None else 0
    yaw_prev = None
    yaw_acc = 0.0
    yaws, pitches, rolls, yrates = [], [], [], []

    if not rep.diverged:
        for i in range(n_steps):
            t = i * dt
            data.ctrl[:] = controller(data, t)
            mujoco.mj_step(model, data)
            if not monitor.check(n_settle + i):
                warn_loudly(f"{variant} @ command {command.as_dict()}", monitor.report())
                break

            roll, pitch, yaw = _rpy(data.qpos[3:7])
            if yaw_prev is None:
                yaw_acc = yaw
            else:
                d = yaw - yaw_prev
                d = (d + np.pi) % (2 * np.pi) - np.pi  # unwrap
                yaw_acc += d
            yaw_prev = yaw

            T.append(t)
            P.append(data.qpos[0:3].copy())
            Q.append(data.qpos[3:7].copy())
            LV.append(data.qvel[0:3].copy())
            yaws.append(yaw_acc)
            pitches.append(pitch)
            rolls.append(roll)

            f = data.actuator_force.copy()
            jv = np.array([data.qvel[a] if a >= 0 else 0.0 for a in act_dof])
            F.append(f)
            JV.append(jv)

            touch = np.zeros(4, dtype=bool)
            for c in range(data.ncon):
                g1 = data.contact.geom1[c]
                g2 = data.contact.geom2[c]
                for k, gid in enumerate(foot_gids):
                    if (g1 == gid and g2 == floor_gid) or (g2 == gid and g1 == floor_gid):
                        touch[k] = True
            C.append(touch)

            SA.append(
                np.array([data.qpos[a] if a >= 0 else np.nan for a in spine_qadr])
            )

            if capture_every and i % capture_every == 0:
                recorder.capture(data)

    n = len(T)
    if n == 0:
        # Diverged during settling: emit an empty, explicitly-diverged log.
        empty = np.zeros((0,))
        return RolloutLog(
            dt=dt, t=empty, pos=np.zeros((0, 3)), quat=np.zeros((0, 4)),
            yaw=empty, pitch=empty, roll=empty, linvel_world=np.zeros((0, 3)),
            fwd_speed=empty, yaw_rate=empty, force=np.zeros((0, model.nu)),
            jvel=np.zeros((0, model.nu)), power=empty, contacts=np.zeros((0, 4), bool),
            spine_angles=np.zeros((0, 3)), ref_pos=np.zeros((0, 2)), ref_yaw=empty,
            command=command, stability=monitor.report(), variant=variant,
            actuator_names=act_names,
            clip_fraction=getattr(controller, "clip_fraction", 0.0),
            frames=list(recorder.frames) if recorder else [],
        )

    t_arr = np.asarray(T)
    pos = np.asarray(P)
    quat = np.asarray(Q)
    lv = np.asarray(LV)
    yaw = np.asarray(yaws)
    force = np.asarray(F)
    jvel = np.asarray(JV)

    # Forward speed = world velocity projected onto the current heading. Using
    # the heading rather than the fixed +x axis keeps this meaningful when the
    # robot is commanded to turn.
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    fwd = np.einsum("ij,ij->i", lv[:, :2], heading)

    yaw_rate = np.gradient(yaw, dt) if n > 1 else np.zeros(n)
    power = np.abs(force * jvel).sum(axis=1)

    # Reference path: integrate the commanded body velocity from the robot's
    # actual starting pose.
    ref_yaw = np.zeros(n)
    ref_pos = np.zeros((n, 2))
    ref_yaw[0] = yaw[0]
    ref_pos[0] = pos[0, :2]
    for i in range(1, n):
        ref_yaw[i] = ref_yaw[i - 1] + command.yaw_rate * dt
        ref_pos[i] = ref_pos[i - 1] + command.vx * dt * np.array(
            [np.cos(ref_yaw[i - 1]), np.sin(ref_yaw[i - 1])]
        )

    log = RolloutLog(
        dt=dt,
        t=t_arr,
        pos=pos,
        quat=quat,
        yaw=yaw,
        pitch=np.asarray(pitches),
        roll=np.asarray(rolls),
        linvel_world=lv,
        fwd_speed=fwd,
        yaw_rate=yaw_rate,
        force=force,
        jvel=jvel,
        power=power,
        contacts=np.asarray(C),
        spine_angles=np.asarray(SA),
        ref_pos=ref_pos,
        ref_yaw=ref_yaw,
        command=command,
        stability=monitor.report(),
        variant=variant,
        actuator_names=act_names,
        clip_fraction=getattr(controller, "clip_fraction", 0.0),
        frames=list(recorder.frames) if recorder else [],
    )
    return log
