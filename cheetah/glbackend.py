"""
OpenGL backend selection for offscreen rendering.

Must run before `import mujoco`. MuJoCo reads MUJOCO_GL once, at import time,
to pick its rendering context; setting it afterwards silently does nothing.

Platform map:
    darwin  -> glfw    (the only offscreen path that works on macOS)
    linux   -> egl     (headless GPU), falling back to osmesa (CPU)
    win32   -> glfw    (WGL under the hood; osmesa needs an OSMesa DLL that
                        is not present on a stock Windows box)

An explicit MUJOCO_GL in the environment always wins -- we never override a
deliberate choice by the user.
"""
from __future__ import annotations

import os
import sys

#: Ordered candidates per platform. First entry is the default, rest are fallbacks.
_CANDIDATES = {
    "darwin": ("glfw",),
    "linux": ("egl", "osmesa", "glfw"),
    "win32": ("glfw", "osmesa"),
}

_configured: str | None = None
_was_explicit = False


def candidates() -> tuple[str, ...]:
    """Backends to try on this platform, best first."""
    return _CANDIDATES.get(sys.platform, ("osmesa",))


def configure_gl(backend: str | None = None) -> str:
    """
    Set MUJOCO_GL and return the backend chosen.

    This only *selects* a backend; it does not prove rendering works. Call
    `probe()` for that -- it is the honest check, because a backend can be
    set successfully and still fail to create a context.
    """
    global _configured, _was_explicit
    if _configured is not None and backend is None:
        return _configured

    if backend is None:
        explicit = os.environ.get("MUJOCO_GL")
        if explicit:
            _was_explicit = True
            backend = explicit
        else:
            backend = candidates()[0]

    os.environ["MUJOCO_GL"] = backend
    # PyOpenGL honours this separately on some platforms.
    if backend == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    elif backend == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    _configured = backend
    return backend


def current() -> str | None:
    """The backend configure_gl() settled on, or None if not yet called."""
    return _configured


def was_explicit() -> bool:
    """True if the backend came from the caller's environment, not our defaults."""
    return _was_explicit


def probe(model=None) -> tuple[bool, str]:
    """
    Actually try to build a Renderer and pull one frame.

    Returns (ok, message). Never raises -- rendering is optional, and a
    headless numeric experiment must not die because a GL context did not
    come up. Callers are expected to report the message loudly and continue.
    """
    backend = current() or configure_gl()
    try:
        import mujoco

        m = model
        if m is None:
            m = mujoco.MjModel.from_xml_string(
                "<mujoco><worldbody><geom type='plane' size='1 1 .1'/>"
                "<body pos='0 0 .3'><freejoint/><geom type='sphere' size='.1'/>"
                "</body></worldbody></mujoco>"
            )
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, height=64, width=64)
        try:
            r.update_scene(d)
            px = r.render()
        finally:
            r.close()
        if px is None or px.size == 0:
            return False, f"MUJOCO_GL={backend}: renderer returned an empty frame"
        return True, f"MUJOCO_GL={backend}: offscreen rendering OK ({px.shape[1]}x{px.shape[0]})"
    except Exception as exc:  # noqa: BLE001 - we genuinely want every failure mode
        return False, f"MUJOCO_GL={backend}: rendering unavailable ({type(exc).__name__}: {exc})"
