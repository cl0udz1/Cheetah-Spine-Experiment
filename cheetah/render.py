"""
Offscreen rendering: PNG stills, MP4 clips, and side-by-side comparisons.

Rendering is strictly optional. Every entry point here degrades to a clear
message and a no-op rather than taking the numeric experiment down with it --
a missing GL context is not a reason to lose a night of results.

There is no interactive viewer anywhere in this package. `mujoco.viewer`
cannot be verified from a headless run, so nothing that needs verifying is
allowed to depend on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import glbackend

#: Divisible by 16, which keeps libx264 from silently rescaling the frames.
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 50

_render_state: dict = {"probed": False, "ok": False, "message": ""}


def rendering_available(model: mujoco.MjModel | None = None) -> tuple[bool, str]:
    """Probe once per process and cache. Returns (ok, message)."""
    if not _render_state["probed"]:
        ok, msg = glbackend.probe(model)
        _render_state.update(probed=True, ok=ok, message=msg)
        if not ok:
            print(f"\n[render] WARNING: {msg}")
            print("[render] Continuing with numeric experiments; no images or video "
                  "will be written.\n", flush=True)
    return _render_state["ok"], _render_state["message"]


@dataclass
class CameraSpec:
    """Free camera that tracks a body."""

    track_body: str = "front_body"
    distance: float = 2.4
    azimuth: float = 120.0
    elevation: float = -18.0
    height_offset: float = 0.15
    #: Follow only along the travel axis, so the robot does not slide around
    #: the frame when it drifts laterally.
    follow_lateral: bool = True


class Recorder:
    """
    Captures frames from a rollout at a fixed wall-clock frame rate.

    Constructed even when rendering is unavailable -- in that case `enabled`
    is False and capture() does nothing, so callers need no branching.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        camera: CameraSpec | None = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.width = width
        self.height = height
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self.cam_spec = camera or CameraSpec()
        self._renderer: mujoco.Renderer | None = None
        self.enabled = False
        self.message = "rendering disabled by caller"

        if not enabled:
            return
        ok, msg = rendering_available(model)
        self.message = msg
        if not ok:
            return
        try:
            self._renderer = mujoco.Renderer(model, height=height, width=width)
        except Exception as exc:  # noqa: BLE001
            self.message = f"Renderer construction failed ({type(exc).__name__}: {exc})"
            print(f"[render] WARNING: {self.message}", flush=True)
            return

        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._cam)
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.distance = self.cam_spec.distance
        self._cam.azimuth = self.cam_spec.azimuth
        self._cam.elevation = self.cam_spec.elevation

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.cam_spec.track_body)
        self._track_bid = bid if bid >= 0 else 1
        self.enabled = True

    def steps_per_frame(self) -> int:
        dt = self.model.opt.timestep
        return max(1, int(round(1.0 / (self.fps * dt))))

    def capture(self, data: mujoco.MjData) -> None:
        """Render one frame now. Silently no-ops when disabled."""
        if not self.enabled or self._renderer is None:
            return
        pos = np.asarray(data.xpos[self._track_bid], dtype=float)
        if not np.isfinite(pos).all():
            return  # never try to point a camera at NaN
        lookat = pos.copy()
        if not self.cam_spec.follow_lateral:
            lookat[1] = 0.0
        lookat[2] += self.cam_spec.height_offset
        self._cam.lookat[:] = lookat
        try:
            self._renderer.update_scene(data, camera=self._cam)
            self.frames.append(self._renderer.render().copy())
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            self.message = f"render failed mid-rollout ({type(exc).__name__}: {exc})"
            print(f"[render] WARNING: {self.message}", flush=True)

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001
                pass
            self._renderer = None


# --------------------------------------------------------------------- output


def _label_frame(frame: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Burn a caption into the top-left of a frame. Pillow ships with matplotlib."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    pad = 6
    draw.rectangle([0, 0, img.width, 34 if sub else 22], fill=(0, 0, 0))
    draw.text((pad, 3), text, fill=(255, 255, 255))
    if sub:
        draw.text((pad, 17), sub, fill=(200, 200, 200))
    return np.asarray(img)


def save_png(frame: np.ndarray, path: str | Path) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(frame).save(path)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[render] WARNING: could not write {path}: {exc}", flush=True)
        return False


def write_video(frames: list[np.ndarray], path: str | Path, fps: int = DEFAULT_FPS) -> bool:
    """
    Encode frames to MP4. Returns True on success.

    Falls back to writing a PNG sequence plus the ffmpeg command line if the
    encoder is unavailable, so the run still leaves usable visual output.
    """
    path = Path(path)
    if not frames:
        print(f"[render] WARNING: no frames captured, skipping {path.name}", flush=True)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            path, fps=fps, codec="libx264", quality=8, macro_block_size=16
        ) as w:
            for f in frames:
                w.append_data(f)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[render] WARNING: MP4 encoding failed ({type(exc).__name__}: {exc})",
              flush=True)
        seq_dir = path.with_suffix("")
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            save_png(f, seq_dir / f"frame_{i:05d}.png")
        print(f"[render] Wrote {len(frames)} PNGs to {seq_dir}. Encode with:\n"
              f"  ffmpeg -framerate {fps} -i \"{seq_dir}/frame_%05d.png\" "
              f"-c:v libx264 -pix_fmt yuv420p \"{path}\"", flush=True)
        return False


def side_by_side(
    left: list[np.ndarray],
    right: list[np.ndarray],
    path: str | Path,
    left_label: str = "spine",
    right_label: str = "rigid",
    left_sub: str = "",
    right_sub: str = "",
    fps: int = DEFAULT_FPS,
) -> bool:
    """
    Compose two frame sequences into one MP4, left | right.

    Sequences are truncated to the shorter length -- if one variant diverged
    early its clip is short, and padding it would misrepresent the run as
    having lasted longer than it did.
    """
    if not left or not right:
        missing = [n for n, f in ((left_label, left), (right_label, right)) if not f]
        print(f"[render] WARNING: no frames for {missing}, skipping side-by-side",
              flush=True)
        return False

    n = min(len(left), len(right))
    if len(left) != len(right):
        print(f"[render] note: clip lengths differ ({left_label}={len(left)}, "
              f"{right_label}={len(right)}); truncating to {n} frames", flush=True)

    combined = []
    sep = None
    for i in range(n):
        a = _label_frame(left[i], left_label, left_sub)
        b = _label_frame(right[i], right_label, right_sub)
        if a.shape != b.shape:
            print("[render] WARNING: frame sizes differ, skipping side-by-side",
                  flush=True)
            return False
        if sep is None:
            sep = np.full((a.shape[0], 4, 3), 90, dtype=a.dtype)
        combined.append(np.concatenate([a, sep, b], axis=1))
    return write_video(combined, path, fps=fps)
