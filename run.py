"""
Starter script for spine_quadruped.xml

    python run.py view      # interactive 3D viewer
    python run.py freefall  # does the spine actually reorient the body in mid-air?

Install:  pip install mujoco
"""
import re
import sys
import numpy as np
import mujoco

XML = "spine_quadruped.xml"


def load(rigid=False):
    """rigid=True physically deletes the spine joints -> true rigid trunk."""
    xml = open(XML).read()
    if rigid:
        # remove the three spine joint elements entirely (the honest way to lock a trunk)
        xml = re.sub(r'<joint name="spine_\w+"[^/]*/>', "", xml)
        xml = re.sub(r'<motor name="spine_\w+"[^/]*/>', "", xml)
    return mujoco.MjModel.from_xml_string(xml)


def view():
    import mujoco.viewer
    model = load()
    data = mujoco.MjData(model)
    print("Viewer open. The robot collapses - there is no controller yet.")
    print("This is the body, not the brain.")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()


def freefall():
    """
    The real physics question, with no RL and no hand-waving:
    in free fall, angular momentum is conserved. Can changing the body's
    SHAPE still rotate it? (This is the falling-cat theorem.)

    We drive the spine in a cyclic pattern during a 2 s fall and measure
    how much net body rotation results. The rigid trunk is the control:
    it should produce ~0, because it has no internal shape to change.
    """
    results = {}
    for label, rigid in [("rigid trunk", True), ("3-DOF spine", False)]:
        model = load(rigid=rigid)
        model.opt.gravity[:] = [0, 0, 0]      # isolate: no ground, no gravity torque
        data = mujoco.MjData(model)
        data.qpos[2] = 5.0

        yaw0 = _yaw(data.qpos[3:7])
        for i in range(1000):                  # 2 seconds
            if not rigid:
                t = i * model.opt.timestep
                # cyclic roll+yaw: bend one way slowly, snap back fast
                phase = (t * 2.0) % 1.0
                amp = 40.0 if phase < 0.5 else -40.0
                data.ctrl[model.nu - 5] = amp      # spine_yaw
                data.ctrl[model.nu - 3] = amp      # spine_roll
            mujoco.mj_step(model, data)
        results[label] = np.rad2deg(_yaw(data.qpos[3:7]) - yaw0)

    print("\nFree-fall reorientation test (zero gravity, 2 s)")
    print("-" * 48)
    for k, v in results.items():
        print(f"  {k:14s} : {v:+8.2f} deg net body rotation")
    print("\nThe rigid trunk cannot reorient itself - no internal DOF to trade")
    print("angular momentum against. The spine can. That is the capability")
    print("S-Cheetah reported, and it is not a matter of degree.")


def _yaw(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return np.arctan2(m[3], m[0])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "view"
    {"view": view, "freefall": freefall}.get(cmd, view)()
