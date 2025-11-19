#!/usr/bin/env python3
"""
Resumable GLB -> multi-zoom PNG renderer (trimesh + pyrender), with:
- Safe batching via subprocesses (GL context fully torn down per batch).
- Multiple zoom levels per object.
- Resume support (skips already-rendered zoom PNGs).
- Global tqdm progress bar over all zoom renders.
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
import subprocess
import gc
import time
from pathlib import Path
from glob import glob

from tqdm import tqdm

# ---------------- EDIT THESE ----------------
IN_DIR   = Path("/Users/mac/PycharmProjects/rotation_image_generation/data/3d_glb_30mb")
OUT_ROOT = Path("/Users/mac/PycharmProjects/rotation_image_generation/data_zoom/dataset_zoom_30mb_notfiltered_10")

WIDTH, HEIGHT = 1024, 1024
FOV_Y_DEG = 50.0
LOOK_DIR = (1, 1, 1)      # isometric-ish
PAD_RATIO = 1.15

# Number of zoom levels per object.
# Extremes are fixed: farthest = 1.0, closest = ZOOM_MIN_MULT.
# For ZOOM_PER_OBJECT = 6 this matches the old levels 0..5 exactly.
ZOOM_PER_OBJECT = 10

ZOOM_MIN_MULT = 0.25   # how close level max gets (fraction of base distance)

BATCH_SIZE = 400        # number of (object, zoom) tasks per worker before full reset
SLEEP_BETWEEN_BATCHES = 0.1
# -------------------------------------------


# ---------------- TASK / RESUME UTILS ----------------
def _build_all_tasks(in_dir: Path, out_root: Path):
    """
    Build a list of tasks:
        {"glb": <glb_path>, "zoom": <float>, "out": <png_path>}
    One task per (object, zoom_level).
    """
    all_glbs = sorted(glob(str(in_dir / "*.glb")))
    tasks = []

    for glb_fp in all_glbs:
        glb_fp = Path(glb_fp)
        name = glb_fp.stem

        obj_out_dir = out_root / name
        obj_out_dir.mkdir(parents=True, exist_ok=True)

        # Now generate zoom indexes 0 .. ZOOM_PER_OBJECT-1
        for z in range(ZOOM_PER_OBJECT):
            out_png = obj_out_dir / f"zoom_{int(z)}.png"
            tasks.append(
                {
                    "glb": str(glb_fp),
                    "zoom": float(z),  # zoom index; mapping handled in worker
                    "out": str(out_png),
                }
            )
    return tasks


def _task_done(task: dict) -> bool:
    """
    Check if this zoom PNG already exists and is non-empty.
    """
    out_png = Path(task["out"])
    return out_png.exists() and out_png.stat().st_size > 0


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _run_worker(batch_tasks, width, height, fov_y_deg, look_dir, pad_ratio, zoom_min_mult, zoom_per_object):
    """
    Spawn a fresh worker subprocess that will render this batch of tasks.
    Args passed via temp JSON file to avoid huge command lines.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(
            {
                "tasks": batch_tasks,
                "width": int(width),
                "height": int(height),
                "fov_y_deg": float(fov_y_deg),
                "look_dir": tuple(look_dir),
                "pad_ratio": float(pad_ratio),
                "zoom_min_mult": float(zoom_min_mult),
                "zoom_per_object": int(zoom_per_object),
            },
            tf,
        )
        tf_path = tf.name

    try:
        cmd = [sys.executable, __file__, "--worker", tf_path]
        res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        return res.returncode == 0
    finally:
        try:
            os.remove(tf_path)
        except Exception:
            pass


# ---------------- WORKER CODE ----------------
def _worker_main(json_path: str):
    """
    Worker process:
    - Reads config JSON.
    - Renders all (glb, zoom) tasks in this batch.
    - Uses trimesh + pyrender offscreen rendering.
    - Deletes GL resources at the end of each render.
    """
    import numpy as np
    import trimesh
    import pyrender
    import imageio.v3 as iio
    from pathlib import Path
    from collections import defaultdict

    with open(json_path, "r") as f:
        cfg = json.load(f)

    tasks = cfg["tasks"]
    width = int(cfg["width"])
    height = int(cfg["height"])
    fov_y_deg = float(cfg["fov_y_deg"])
    look_dir = tuple(cfg["look_dir"])
    pad_ratio = float(cfg["pad_ratio"])
    zoom_min_mult = float(cfg["zoom_min_mult"])
    zoom_per_object = int(cfg["zoom_per_object"])

    def _look_at(eye, target, up=(0, 1, 0)):
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)

        z = eye - target
        z /= np.linalg.norm(z) + 1e-9
        x = np.cross(up, z)
        x /= np.linalg.norm(x) + 1e-9
        y = np.cross(z, x)

        M = np.eye(4, dtype=np.float32)
        M[:3, 0] = x
        M[:3, 1] = y
        M[:3, 2] = z
        M[:3, 3] = eye
        return M

    def _scene_bounds(obj):
        # Works for both trimesh.Scene and trimesh.Trimesh
        b = obj.bounds
        c = (b[0] + b[1]) * 0.5
        ext = b[1] - b[0]
        # radius of sphere that encloses the AABB from its center (conservative fit)
        r = 0.5 * np.linalg.norm(ext)
        # handle degenerate tiny models
        r = float(max(r, 1e-3))
        return c, ext, r

    def _composite_on_white(rgba):
        if rgba.ndim == 3 and rgba.shape[-1] == 3:
            return rgba
        rgb = rgba[..., :3].astype(np.float32)
        a = (rgba[..., 3:4].astype(np.float32)) / 255.0
        out = rgb * a + 255.0 * (1.0 - a)
        return out.astype(np.uint8)

    def _zoom_multiplier(level, zoom_per_object, min_mult=0.25):
        """
        Map zoom level in [0 .. zoom_per_object-1] to a distance multiplier.
        - Level 0           -> 1.0 (original distance)
        - Level max (N-1)   -> min_mult (tight close-up)
        Extremes are fixed regardless of zoom_per_object.
        For zoom_per_object = 6 this matches the old 0..5 mapping.
        """
        if zoom_per_object <= 1:
            # Single zoom level: no zoom variation
            return 1.0
        max_level = float(zoom_per_object - 1)
        lvl = float(np.clip(level, 0.0, max_level))
        t = lvl / max_level  # 0 .. 1
        return 1.0 - t * (1.0 - min_mult)

    # Group tasks by GLB path so we can load each file once per worker
    from collections import defaultdict
    by_glb = defaultdict(list)
    for t in tasks:
        by_glb[t["glb"]].append(t)

    rendered = 0
    errors = 0

    for glb_path, glb_tasks in by_glb.items():
        glb_path = Path(glb_path)

        tm = None
        try:
            # Load GLB as a trimesh.Scene (or Trimesh)
            tm = trimesh.load(glb_path, force="scene")
        except Exception:
            errors += len(glb_tasks)
            continue

        try:
            center, extents, radius = _scene_bounds(tm)

            aspect = float(width) / float(height)
            fovy = np.deg2rad(fov_y_deg)
            fovx = 2.0 * np.arctan(np.tan(fovy / 2.0) * aspect)

            dist_y = radius / np.tan(fovy / 2.0)
            dist_x = radius / np.tan(fovx / 2.0)
            base_dist = max(dist_x, dist_y) * pad_ratio

            # Shared normalized look_dir
            look = np.array(look_dir, dtype=np.float32)
            look /= (np.linalg.norm(look) + 1e-9)

            for task in glb_tasks:
                out_png = Path(task["out"])
                zoom_level = float(task["zoom"])  # index 0 .. zoom_per_object-1

                # Resume skip: if this zoom PNG exists and non-empty, skip
                if out_png.exists() and out_png.stat().st_size > 0:
                    continue

                scene = None
                renderer = None
                try:
                    # Distance for this zoom level
                    dist = base_dist * _zoom_multiplier(
                        zoom_level,
                        zoom_per_object=zoom_per_object,
                        min_mult=zoom_min_mult,
                    )
                    eye = center + look * dist
                    cam_pose = _look_at(eye, center, up=(0, 1, 0))

                    # Near / far
                    znear = max(1e-3, dist - 2.5 * radius)
                    zfar = dist + 2.5 * radius

                    # Build pyrender scene
                    try:
                        scene = pyrender.Scene.from_trimesh_scene(
                            tm,
                            ambient_light=[0.35, 0.35, 0.35],
                            bg_color=[1.0, 1.0, 1.0, 0.0],
                        )
                    except Exception:
                        # Fallback: single-mesh without materials
                        if not isinstance(tm, trimesh.Scene):
                            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
                            scene = pyrender.Scene(
                                ambient_light=[0.35, 0.35, 0.35],
                                bg_color=[1, 1, 1, 0],
                            )
                            scene.add(mesh)
                        else:
                            errors += 1
                            continue

                    cam = pyrender.PerspectiveCamera(
                        yfov=fovy,
                        aspectRatio=aspect,
                        znear=znear,
                        zfar=zfar,
                    )
                    scene.add(cam, pose=cam_pose)

                    # Lights (use base_dist so they stay roughly consistent)
                    key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
                    fill = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
                    rim = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

                    scene.add(key, pose=_look_at(center + np.array([base_dist,  base_dist,  base_dist]), center))
                    scene.add(fill, pose=_look_at(center + np.array([-base_dist,  base_dist,  base_dist]), center))
                    scene.add(rim,  pose=_look_at(center + np.array([base_dist, -base_dist, -base_dist]), center))

                    renderer = pyrender.OffscreenRenderer(
                        viewport_width=width,
                        viewport_height=height,
                    )
                    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                    rgb = _composite_on_white(rgba)

                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    iio.imwrite(out_png, rgb)
                    rendered += 1

                except Exception:
                    errors += 1
                finally:
                    try:
                        if renderer is not None:
                            renderer.delete()
                    except Exception:
                        pass
                    del renderer, scene
                    gc.collect()
                    time.sleep(0.005)

        finally:
            del tm
            gc.collect()

    print(f"[worker] rendered={rendered} errors={errors}")
    # Always exit 0 so main keeps going
    return 0


# ---------------- MAIN ----------------
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Build all (object, zoom) tasks
    all_tasks = _build_all_tasks(IN_DIR, OUT_ROOT)
    total_tasks = len(all_tasks)

    # Count already-done
    already_done = sum(1 for t in all_tasks if _task_done(t))
    pending_tasks = [t for t in all_tasks if not _task_done(t)]

    print(f"Found {total_tasks} zoom tasks "
          f"({len(sorted(set(t['glb'] for t in all_tasks)))} GLBs)")
    print(f"Already rendered (skipping): {already_done}")
    print(f"To render now: {len(pending_tasks)}")

    if not pending_tasks:
        print("Nothing to do. Exiting.")
        return

    total_rendered_this_run = 0

    pbar = tqdm(
        total=len(pending_tasks),
        desc="🔥 Rendering zooms",
        unit="img",
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| "
            "{n_fmt}/{total_fmt} "
            "[elapsed {elapsed} • left {remaining} • {rate_fmt}]"
        ),
        colour="cyan",
    )

    for batch_idx, batch in enumerate(_chunked(pending_tasks, BATCH_SIZE), start=1):
        print(f"\n[main] Batch {batch_idx}: {len(batch)} tasks")
        ok = _run_worker(
            batch,
            width=WIDTH,
            height=HEIGHT,
            fov_y_deg=FOV_Y_DEG,
            look_dir=LOOK_DIR,
            pad_ratio=PAD_RATIO,
            zoom_min_mult=ZOOM_MIN_MULT,
            zoom_per_object=ZOOM_PER_OBJECT,
        )
        if not ok:
            print("[main] Worker returned non-zero exit. Continuing.")

        # Count how many zoom PNGs became done in this batch
        new_done = sum(1 for t in batch if _task_done(t))
        total_rendered_this_run += new_done
        pbar.update(new_done)
        print(f"[main] Batch {batch_idx} done: +{new_done} zoom images")

        gc.collect()
        time.sleep(SLEEP_BETWEEN_BATCHES)

    pbar.close()

    now_done = sum(1 for t in all_tasks if _task_done(t))
    print("\nSummary:")
    print(f"  Already done before run: {already_done}")
    print(f"  Rendered this run:       {now_done - already_done}")
    print(f"  Total done now:          {now_done} / {total_tasks} zoom images")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(_worker_main(sys.argv[2]))
    else:
        main()