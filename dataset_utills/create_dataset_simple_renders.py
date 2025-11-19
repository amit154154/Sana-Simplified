#!/usr/bin/env python3
"""
Resumable GLB -> PNG renderer (trimesh + pyrender), with hard memory reset every K files.

- Skips existing PNGs (resume).
- Spawns a fresh subprocess for each batch, so the GL context is fully torn down.
- Prints counts before and after.
"""

from __future__ import annotations
import os, sys, json, tempfile, subprocess, gc, time
from pathlib import Path
from glob import glob

# ---------------- EDIT THESE ----------------
IN_DIR  = Path("/Users/mac/PycharmProjects/rotation_image_generation/data/3d_glb_30mb")
OUT_DIR = Path("/Users/mac/PycharmProjects/rotation_image_generation/data/stage1_processed_full_data")
WIDTH, HEIGHT = 1024, 1024
FOV_Y_DEG = 50.0
LOOK_DIR = (1, 1, 1)      # isometric-ish
PAD_RATIO = 1.15
ZOOM_MULTIPLIER = 0.85
BATCH_SIZE = 100          # number of files per worker before full reset
SLEEP_BETWEEN_BATCHES = 0.1
# -------------------------------------------


def _already_done_count(glbs, out_dir: Path) -> int:
    n = 0
    for f in glbs:
        png = out_dir / (Path(f).stem + ".png")
        if png.exists() and png.stat().st_size > 0:
            n += 1
    return n


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _run_worker(batch_paths, out_dir: Path):
    # pass args via a temp JSON file to avoid huge command lines
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({
            "files": batch_paths,
            "out_dir": str(out_dir),
            "width": WIDTH,
            "height": HEIGHT,
            "fov_y_deg": FOV_Y_DEG,
            "look_dir": LOOK_DIR,
            "pad_ratio": PAD_RATIO,
            "zoom_multiplier": ZOOM_MULTIPLIER,
        }, tf)
        tf_path = tf.name

    try:
        # call *this same script* in worker mode
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
    import numpy as np
    import trimesh
    import pyrender
    import imageio.v3 as iio

    with open(json_path, "r") as f:
        cfg = json.load(f)
    files = cfg["files"]
    out_dir = Path(cfg["out_dir"])
    width = int(cfg["width"])
    height = int(cfg["height"])
    fov_y_deg = float(cfg["fov_y_deg"])
    look_dir = tuple(cfg["look_dir"])
    pad_ratio = float(cfg["pad_ratio"])
    zoom_multiplier = float(cfg["zoom_multiplier"])

    out_dir.mkdir(parents=True, exist_ok=True)

    def _look_at(eye, target, up=(0, 1, 0)):
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        z = eye - target; z /= np.linalg.norm(z) + 1e-9
        x = np.cross(up, z); x /= np.linalg.norm(x) + 1e-9
        y = np.cross(z, x)
        M = np.eye(4, dtype=np.float32)
        M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z; M[:3, 3] = eye
        return M

    def _over_white(rgba):
        if rgba.ndim == 3 and rgba.shape[-1] == 3:
            return rgba
        rgb = rgba[..., :3].astype(np.float32)
        a = (rgba[..., 3:4].astype(np.float32)) / 255.0
        out = rgb * a + 255.0 * (1.0 - a)
        return out.astype(np.uint8)

    rendered = 0
    errors = 0

    for f in files:
        in_fp = Path(f)
        out_png = out_dir / f"{in_fp.stem}.png"

        # resume skip
        if out_png.exists() and out_png.stat().st_size > 0:
            continue

        tm = None
        scene = None
        renderer = None
        try:
            # trimesh load as scene (keeps materials if present)
            tm = trimesh.load(in_fp, force="scene")

            # bounds -> center & radius
            bmin, bmax = tm.bounds
            center = ((bmin + bmax) * 0.5).astype(np.float32)
            radius = float(max(0.5 * np.linalg.norm((bmax - bmin)), 1e-3))

            # camera fit
            aspect = width / float(height)
            fovy = np.deg2rad(fov_y_deg)
            fovx = 2.0 * np.arctan(np.tan(fovy / 2.0) * aspect)
            dist_y = radius / np.tan(fovy / 2.0)
            dist_x = radius / np.tan(fovx / 2.0)
            base_dist = max(dist_x, dist_y) * pad_ratio
            dist = base_dist * zoom_multiplier

            # view
            ldir = np.array(look_dir, dtype=np.float32)
            ldir /= (np.linalg.norm(ldir) + 1e-9)
            eye = center + ldir * dist
            cam_pose = _look_at(eye, center, up=(0, 1, 0))

            # scene build
            try:
                scene = pyrender.Scene.from_trimesh_scene(
                    tm, ambient_light=[0.35, 0.35, 0.35], bg_color=[1, 1, 1, 0]
                )
            except Exception:
                # fallback for single mesh
                if isinstance(tm, trimesh.Trimesh):
                    scene = pyrender.Scene(ambient_light=[0.35, 0.35, 0.35], bg_color=[1, 1, 1, 0])
                    scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))
                else:
                    errors += 1
                    continue

            # camera node
            znear = max(1e-3, dist - 2.5 * radius)
            zfar  = dist + 2.5 * radius
            cam = pyrender.PerspectiveCamera(yfov=fovy, aspectRatio=aspect, znear=znear, zfar=zfar)
            scene.add(cam, pose=cam_pose)

            # simple 3-point lights
            key  = pyrender.DirectionalLight([1,1,1], intensity=3.0)
            fill = pyrender.DirectionalLight([1,1,1], intensity=2.0)
            rim  = pyrender.DirectionalLight([1,1,1], intensity=2.0)
            scene.add(key,  pose=_look_at(center + np.array([ dist,  dist,  dist]), center))
            scene.add(fill, pose=_look_at(center + np.array([-dist,  dist,  dist]), center))
            scene.add(rim,  pose=_look_at(center + np.array([ dist, -dist, -dist]), center))

            # offscreen render (created per file; process dies after batch = full cleanup)
            renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            rgb = _over_white(rgba)
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
            del renderer, scene, tm
            gc.collect()
            time.sleep(0.005)

    print(f"[worker] rendered={rendered} errors={errors}")
    # exit code 0 even if some files failed — main will keep going
    return 0


# ---------------- MAIN ----------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_glbs = sorted(glob(str(IN_DIR / "*.glb")))
    already_done = _already_done_count(all_glbs, OUT_DIR)
    to_do = len(all_glbs) - already_done

    print(f"Found {len(all_glbs)} .glb files")
    print(f"Already rendered (skipping): {already_done}")
    print(f"To render now: {to_do}")

    # only process files that still need rendering
    pending = [f for f in all_glbs
               if not (OUT_DIR / (Path(f).stem + ".png")).exists()
               or (OUT_DIR / (Path(f).stem + ".png")).stat().st_size == 0]

    total_rendered_this_run = 0
    total_errors = 0

    for batch_idx, batch in enumerate(_chunked(pending, BATCH_SIZE), start=1):
        print(f"\n[main] Batch {batch_idx}: {len(batch)} files")
        ok = _run_worker(batch, OUT_DIR)
        if not ok:
            print("[main] Worker returned non-zero exit. Continuing.")

        # count how many new PNGs appeared for this batch
        new_done = sum(1 for f in batch
                       if (OUT_DIR / (Path(f).stem + ".png")).exists()
                       and (OUT_DIR / (Path(f).stem + ".png")).stat().st_size > 0)
        total_rendered_this_run += new_done
        print(f"[main] Batch {batch_idx} done: +{new_done} files")

        # small pause + GC to let OS reclaim GL resources
        gc.collect()
        time.sleep(SLEEP_BETWEEN_BATCHES)

    now_done = _already_done_count(all_glbs, OUT_DIR)
    print("\nSummary:")
    print(f"  Already done before run: {already_done}")
    print(f"  Rendered this run:       {now_done - already_done}")
    print(f"  Total done now:          {now_done} / {len(all_glbs)}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(_worker_main(sys.argv[2]))
    else:
        main()