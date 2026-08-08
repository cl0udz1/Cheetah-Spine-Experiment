"""
Build the committed media set the README embeds.

Everything under media/ and results/ is gitignored because a full render run is
~17 MB and regenerating it is cheap. The README needs its figures to exist in
the repo, so this copies a curated, size-budgeted subset into docs/media/ and
renders the autoplaying GIF used at the top (GitHub autoplays GIFs in a README
preview but not MP4s).

Run `python harness.py run --name closed_loop --variants spine,rigid,passive
--seeds 0,1,2,3,4,5,6,7 --duration 8.0 --render` first, then:

    python tools/make_readme_media.py

Total budget is ~5 MB so the page loads comfortably on GitHub.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

import cheetah  # noqa: F401  -- sets MUJOCO_GL before mujoco is imported
from cheetah import render
from cheetah.control import CPGController, GaitParams
from cheetah.model import build_model
from cheetah.rollout import Command, run_rollout
from harness import DEFAULT_GAIT

SRC = Path("media/closed_loop")
DST = Path("docs/media")

#: Figures the README embeds, with the max width each is downsampled to.
FIGURES = {
    "trajectory_straight_1.0.png": 1100,
    "trajectory_turn_0.8.png": 900,
    "spine_angles_turn_0.8.png": 1200,
    "gait_straight_1.0.png": 1200,
    "speed_straight_2.0.png": 1200,
}

GIF_COMMAND = ("turn", 1.0, 0.8)
GIF_SECONDS = 3.0
GIF_FPS = 12             # enough to read the gait; 20 fps cost 2.9 MB
GIF_WIDTH = 600          # composed width after downscale
GIF_COLORS = 48          # the scene is nearly monochrome, so this is generous


def copy_figures() -> list[tuple[str, int]]:
    out = []
    for name, max_w in FIGURES.items():
        src = SRC / name
        if not src.exists():
            print(f"  MISSING {src} -- run the render pass first")
            continue
        img = Image.open(src)
        if img.width > max_w:
            h = round(img.height * max_w / img.width)
            img = img.resize((max_w, h), Image.LANCZOS)
        dst = DST / name
        img.save(dst, optimize=True)
        out.append((name, dst.stat().st_size))
        print(f"  {name:<34} {dst.stat().st_size/1024:8.1f} KB")
    return out


def copy_video() -> None:
    """Ship one MP4: the turn, which is the only command with a real effect."""
    src = SRC / "sidebyside_turn_0.8.mp4"
    if not src.exists():
        print(f"  MISSING {src} -- run the render pass first")
        return
    dst = DST / "sidebyside_turn_0.8.mp4"
    shutil.copy2(src, dst)
    print(f"  {dst.name:<34} {dst.stat().st_size/1024:8.1f} KB")


def render_gif() -> None:
    """Short spine-vs-rigid loop for the top of the README."""
    ok, msg = render.rendering_available()
    if not ok:
        print(f"  GIF skipped: {msg}")
        return

    _, vx, yaw_rate = GIF_COMMAND
    command = Command(vx=vx, yaw_rate=yaw_rate)
    frames_by_variant = {}
    for variant in ("spine", "rigid"):
        model, info = build_model(variant=variant)
        rec = render.Recorder(model, width=480, height=360, fps=GIF_FPS)
        ctrl = CPGController(model, GaitParams(**DEFAULT_GAIT), command=command)
        run_rollout(model, ctrl, command, duration=GIF_SECONDS, settle=0.5,
                    variant=variant, recorder=rec, seed=0)
        rec.close()
        frames_by_variant[variant] = rec.frames
        if not rec.frames:
            print("  GIF skipped: no frames captured")
            return

    left, right = frames_by_variant["spine"], frames_by_variant["rigid"]
    n = min(len(left), len(right))
    sep = np.full((left[0].shape[0], 6, 3), 90, dtype=left[0].dtype)

    out = []
    for i in range(n):
        a = render._label_frame(left[i], "active spine", "turning, 0.8 rad/s cmd")
        b = render._label_frame(right[i], "rigid trunk", "turning, 0.8 rad/s cmd")
        comp = np.concatenate([a, sep, b], axis=1)
        img = Image.fromarray(comp)
        h = round(img.height * GIF_WIDTH / img.width)
        img = img.resize((GIF_WIDTH, h), Image.LANCZOS)
        out.append(img.quantize(colors=GIF_COLORS, method=Image.MEDIANCUT))

    dst = DST / "hero_turn.gif"
    out[0].save(dst, save_all=True, append_images=out[1:],
                duration=int(1000 / GIF_FPS), loop=0, optimize=True)
    print(f"  {dst.name:<34} {dst.stat().st_size/1024:8.1f} KB  "
          f"({len(out)} frames, {GIF_WIDTH}px wide)")


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    print(f"writing README media to {DST}/\n")
    copy_figures()
    copy_video()
    render_gif()
    total = sum(p.stat().st_size for p in DST.iterdir() if p.is_file())
    print(f"\ntotal committed media: {total/1024/1024:.2f} MB")
    if total > 6 * 1024 * 1024:
        print("WARNING: over the ~5 MB budget; reduce GIF_SECONDS or GIF_WIDTH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
