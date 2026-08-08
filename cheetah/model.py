"""
Model variant construction.

The rigid variant *deletes* the spine joints and their motors. It does not
stiffen them. Stiffening a hinge to fake a weld drives the mass matrix
condition number up and MuJoCo's solver falls over -- you get a diverged sim
that still returns finite-looking numbers for a while.

With the joint element gone, MuJoCo welds the child body to its parent at
compile time: same bodies, same geoms, same total mass, three fewer DOF.
That is a real rigid trunk, and the mass-matching is what makes the
spine-vs-rigid comparison mean anything.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_XML = "spine_quadruped.xml"

#: Spine joint names in proximal->distal order, matching the S-Cheetah ordering.
SPINE_JOINTS = ("spine_yaw", "spine_pitch", "spine_roll")

#: Standing pose. Thigh and calf are both 0.22 m; hip_pitch=+0.6, knee=-1.2
#: puts each foot directly under its hip (the +x and -x offsets cancel) at
#: 0.363 m below the hip, so the trunk sits at 0.363 + 0.025 (foot radius).
HOME_HIP_PITCH = 0.6
HOME_KNEE = -1.2
HOME_HEIGHT = 0.388


@dataclass
class VariantInfo:
    """What actually got built, verified against the compiled model."""

    name: str
    spine_joints: tuple[str, ...]
    nq: int
    nv: int
    nu: int
    njnt: int
    total_mass: float
    removed_joints: tuple[str, ...] = ()
    removed_motors: tuple[str, ...] = ()
    actuator_names: tuple[str, ...] = field(default_factory=tuple)
    joint_names: tuple[str, ...] = field(default_factory=tuple)
    #: Spring-damper on the spine joints. Zero for "spine" and "rigid".
    spine_stiffness: float = 0.0
    spine_damping: float = 0.0
    #: True when no actuator can reach any spine joint.
    spine_unactuated: bool = False

    def as_dict(self) -> dict:
        return {
            "variant": self.name,
            "spine_joints": list(self.spine_joints),
            "nq": self.nq,
            "nv": self.nv,
            "nu": self.nu,
            "njnt": self.njnt,
            "total_mass_kg": round(self.total_mass, 6),
            "removed_joints": list(self.removed_joints),
            "removed_motors": list(self.removed_motors),
            "spine_stiffness": self.spine_stiffness,
            "spine_damping": self.spine_damping,
            "spine_unactuated": self.spine_unactuated,
        }


class ModelBuildError(RuntimeError):
    """Raised when the compiled model does not match what we asked for."""


def _strip_spine(root: ET.Element, keep: set[str]) -> tuple[list[str], list[str]]:
    """
    Delete <joint> and <motor> elements for spine joints not in `keep`.

    ElementTree has no parent pointers, so we walk parents explicitly.
    Returns (removed_joint_names, removed_motor_names).
    """
    removed_j: list[str] = []
    removed_m: list[str] = []

    for parent in root.iter():
        for child in list(parent):
            if child.tag == "joint":
                name = child.get("name", "")
                if name in SPINE_JOINTS and name not in keep:
                    parent.remove(child)
                    removed_j.append(name)
            elif child.tag == "motor":
                # A motor is orphaned if the joint it drives is gone.
                jnt = child.get("joint", "")
                if jnt in SPINE_JOINTS and jnt not in keep:
                    parent.remove(child)
                    removed_m.append(child.get("name", jnt))

    return removed_j, removed_m


def _make_spine_passive(
    root: ET.Element, keep: set[str], stiffness: float, damping: float
) -> list[str]:
    """
    Turn the spine into a passive spring-damper: keep the joints, delete their
    motors, and give each joint stiffness and damping.

    A PD controller holding a joint at zero with gains (kp, kd) is *exactly* a
    spring-damper of stiffness kp and damping kd, up to actuator saturation. So
    setting stiffness=kp_spine and damping=kd_spine reproduces the actuated
    variant's neutral-hold behaviour with zero motor torque, which is what makes
    "is the benefit compliance or control?" a clean question.

    Returns the names of the motors removed.
    """
    removed_m: list[str] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "joint" and child.get("name", "") in keep:
                child.set("stiffness", f"{stiffness:g}")
                child.set("damping", f"{damping:g}")
                # springref defaults to 0, i.e. the spring's rest position is
                # the neutral spine. Stated explicitly so it cannot drift with
                # a future edit to the default class.
                child.set("springref", "0")
            elif child.tag == "motor" and child.get("joint", "") in keep:
                parent.remove(child)
                removed_m.append(child.get("name", child.get("joint", "")))
    return removed_m


def build_model(
    variant: str = "spine",
    xml_path: str | Path = DEFAULT_XML,
    keep_spine_joints: tuple[str, ...] | None = None,
    timestep: float | None = None,
    passive_stiffness: float = 400.0,
    passive_damping: float = 12.0,
    spine: bool | None = None,
) -> tuple[mujoco.MjModel, VariantInfo]:
    """
    Compile one of the three trunk variants.

        "spine"   -- joints present, motor-driven (active spine)
        "rigid"   -- joints deleted, child body welded to parent (rigid trunk)
        "passive" -- joints present with spring+damper, NO motors (compliant
                     trunk). Isolates compliance from control.

    Args:
        variant: one of the three names above.
        xml_path: source MJCF.
        keep_spine_joints: explicit subset to keep. Lets a sweep ask for e.g.
            pitch-only without hand-editing XML.
        timestep: override the integrator timestep.
        passive_stiffness/passive_damping: spring-damper gains for the passive
            variant. Defaults match kp_spine/kd_spine so a passive spine and a
            held actuated spine are the same mechanical system.
        spine: deprecated bool form of `variant`, kept so older call sites work.

    Returns (model, info). Raises ModelBuildError if the compiled model does
    not match the request -- a silent no-op edit is exactly the failure mode
    that produces a fake "rigid" result.
    """
    if spine is not None:
        variant = "spine" if spine else "rigid"
    if variant not in ("spine", "rigid", "passive"):
        raise ValueError(f"unknown variant {variant!r}")

    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    if keep_spine_joints is None:
        keep = set() if variant == "rigid" else set(SPINE_JOINTS)
    else:
        keep = set(keep_spine_joints)
        unknown = keep - set(SPINE_JOINTS)
        if unknown:
            raise ValueError(f"unknown spine joints: {sorted(unknown)}")

    removed_j, removed_m = _strip_spine(root, keep)
    if variant == "passive":
        removed_m += _make_spine_passive(root, keep, passive_stiffness, passive_damping)

    if timestep is not None:
        opt = root.find("option")
        if opt is None:
            opt = ET.SubElement(root, "option")
        opt.set("timestep", str(timestep))

    xml_str = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_str)

    joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"<{i}>"
        for i in range(model.njnt)
    )
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"<{i}>"
        for i in range(model.nu)
    )

    # Verify the surgery landed, rather than trusting the string edit.
    survivors = tuple(j for j in SPINE_JOINTS if j in joint_names)
    if set(survivors) != keep:
        raise ModelBuildError(
            f"spine joint removal failed: asked to keep {sorted(keep)}, "
            f"compiled model has {sorted(survivors)}"
        )
    for name in removed_m:
        if name in actuator_names:
            raise ModelBuildError(f"motor {name!r} survived removal")

    # The passive variant's defining property is that no actuator can reach the
    # spine. Verify against the compiled model rather than trusting the edit.
    if variant == "passive":
        for i in range(model.nu):
            if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = int(model.actuator_trnid[i, 0])
                jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if jn in SPINE_JOINTS:
                    raise ModelBuildError(
                        f"passive variant still has an actuator driving {jn!r}"
                    )
        for jn in keep:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if model.jnt_stiffness[jid] <= 0:
                raise ModelBuildError(
                    f"passive variant has no spring on {jn!r} "
                    f"(stiffness={model.jnt_stiffness[jid]})"
                )

    if keep == set(SPINE_JOINTS):
        name = variant  # "spine" or "passive"
    elif not keep:
        name = "rigid"
    else:
        name = f"partial_{variant}"
    info = VariantInfo(
        name=name,
        spine_joints=tuple(survivors),
        nq=model.nq,
        nv=model.nv,
        nu=model.nu,
        njnt=model.njnt,
        total_mass=float(model.body_mass.sum()),
        removed_joints=tuple(removed_j),
        removed_motors=tuple(removed_m),
        actuator_names=actuator_names,
        joint_names=joint_names,
        spine_stiffness=passive_stiffness if variant == "passive" else 0.0,
        spine_damping=passive_damping if variant == "passive" else 0.0,
        spine_unactuated=(variant == "passive" or not keep),
    )
    return model, info


def actuator_index(model: mujoco.MjModel) -> dict[str, int]:
    """name -> ctrl index. Indices shift between variants; never hardcode them."""
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }


def joint_qpos_index(model: mujoco.MjModel) -> dict[str, int]:
    """name -> qpos address, for hinge joints."""
    out = {}
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm:
            out[nm] = int(model.jnt_qposadr[i])
    return out


def joint_dof_index(model: mujoco.MjModel) -> dict[str, int]:
    """name -> qvel/dof address, for hinge joints."""
    out = {}
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm:
            out[nm] = int(model.jnt_dofadr[i])
    return out


def set_home_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the robot in a crouched stance with feet on the ground, at rest."""
    mujoco.mj_resetData(model, data)
    qadr = joint_qpos_index(model)
    data.qpos[0:3] = [0.0, 0.0, HOME_HEIGHT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for leg in ("fl", "fr", "rl", "rr"):
        data.qpos[qadr[f"{leg}_hip_pitch"]] = HOME_HIP_PITCH
        data.qpos[qadr[f"{leg}_knee"]] = HOME_KNEE
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def home_qpos(model: mujoco.MjModel) -> np.ndarray:
    """The standing pose as a bare qpos vector."""
    d = mujoco.MjData(model)
    set_home_pose(model, d)
    return d.qpos.copy()


def foot_geom_ids(model: mujoco.MjModel) -> dict[str, int]:
    """leg -> geom id for the four foot spheres, used for contact detection."""
    out = {}
    for leg in ("fl", "fr", "rl", "rr"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot")
        if gid < 0:
            raise ModelBuildError(f"foot geom {leg}_foot not found")
        out[leg] = gid
    return out
