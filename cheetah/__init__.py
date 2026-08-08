"""
Spine-vs-rigid quadruped experiment harness.

IMPORTANT: importing this package sets MUJOCO_GL before mujoco is imported
anywhere. Every submodule lives under this package, so Python guarantees this
file runs first. Do not `import mujoco` above this line in any module here.
"""
from .glbackend import configure_gl

configure_gl()

__all__ = ["configure_gl"]
