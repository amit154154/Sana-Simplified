#!/usr/bin/env python3
"""
Simple GLB -> multi-view PNG renderer (one GLB file).

- Input: a single .glb file path
- Output: multiple renders from different distances and angles, saved to OUT_DIR
- Dependencies: trimesh, pyrender, numpy, imageio
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import trimesh
import pyrender
import imageio.v3 as iio


# ========= USER CONFIG =========
GLB_PATH = Path("/Users/mac/PycharmProjects/rotation_image_generation/data/3d_glb_30mb/00c3f6de0c5b494bbf2885ff99ee6b96.glb")
OUT_DIR  = Path("/Users/mac/PycharmProjects/rotation_image_generation/data_dreambooth/puffin")

WIDTH, HEIGHT = 1024, 1024
FOV_Y_DEG     = 50.0
PAD_RATIO     = 1.15  # a bit of padding so the object isn't cropped

# Distances as multipliers of the base distance
DIST_MULTS = [1.0, 0.9,0.8]

# Angles in degrees: (elevation, azimuth)
#   elevation: 0 = horizontal, 90 = top view
#   azimuth: angle around the object in the ground plane
ANGLES = [
    (20,   45),
    (20,  135),
    (20,  225),
    (20,  315),
    (45,   45),
    (45,  135),
    (45,  225),
    (45,  315),
]
# ===============================


def look_at(eye, target, up=(0, 1, 0)):
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


def scene_bounds(obj):
    """
    Works for both trimesh.Scene and trimesh.Trimesh.
    Returns center, extents, and radius of a sphere enclosing the AABB.
    """
    b = obj.bounds  # (min, max)
    c = (b[0] + b[1]) * 0.5
    ext = b[1] - b[0]
    r = 0.5 * np.linalg.norm(ext)
    r = float(max(r, 1e-3))
    return c, ext, r


def composite_on_white(rgba):
    """
    If RGBA, composite over white background.
    If RGB, return as is.
    """
    if rgba.ndim == 3 and rgba.shape[-1] == 3:
        return rgba
    rgb = rgba[..., :3].astype(np.float32)
    a = (rgba[..., 3:4].astype(np.float32)) / 255.0
    out = rgb * a + 255.0 * (1.0 - a)
    return out.astype(np.uint8)


def spherical_to_cartesian(radius, elevation_deg, azimuth_deg):
    """
    Convert spherical coords to Cartesian:
    - radius: distance from origin
    - elevation: angle from the ground plane (0 = horizontal, 90 = top)
    - azimuth: angle around Z axis (0 along +X, 90 along +Y)
    """
    elev = math.radians(elevation_deg)
    az   = math.radians(azimuth_deg)

    x = radius * math.cos(elev) * math.cos(az)
    y = radius * math.cos(elev) * math.sin(az)
    z = radius * math.sin(elev)
    return np.array([x, y, z], dtype=np.float32)


def render_multi_views(
    glb_path: Path,
    out_dir: Path,
    width: int = 1024,
    height: int = 1024,
    fov_y_deg: float = 50.0,
    pad_ratio: float = 1.15,
    dist_mults=None,
    angles=None,
):
    if dist_mults is None:
        dist_mults = [1.0, 0.8, 0.6, 0.4]
    if angles is None:
        angles = [(20, 45), (20, 135), (20, 225), (20, 315)]

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GLB: {glb_path}")
    tm = trimesh.load(glb_path, force="scene")

    center, extents, radius = scene_bounds(tm)

    # Compute base distance so object fits in view
    aspect = float(width) / float(height)
    fovy = math.radians(fov_y_deg)
    fovx = 2.0 * math.atan(math.tan(fovy / 2.0) * aspect)

    dist_y = radius / math.tan(fovy / 2.0)
    dist_x = radius / math.tan(fovx / 2.0)
    base_dist = max(dist_x, dist_y) * pad_ratio

    # Build static scene (we move camera only)
    try:
        scene = pyrender.Scene.from_trimesh_scene(
            tm,
            ambient_light=[0.35, 0.35, 0.35],
            bg_color=[1.0, 1.0, 1.0, 0.0],
        )
    except Exception:
        # Fallback: if it's a single mesh
        if not isinstance(tm, trimesh.Scene):
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            scene = pyrender.Scene(
                ambient_light=[0.35, 0.35, 0.35],
                bg_color=[1.0, 1.0, 1.0, 0.0],
            )
            scene.add(mesh)
        else:
            raise

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    # Pre-add lights
    # (positions are relative to base_dist so they stay roughly reasonable)
    key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    fill = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    rim = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

    center_np = np.array(center, dtype=np.float32)
    def _pose(offset):
        return look_at(center_np + offset, center_np, up=(0, 1, 0))

    scene.add(key,  pose=_pose(np.array([ base_dist,  base_dist,  base_dist], dtype=np.float32)))
    scene.add(fill, pose=_pose(np.array([-base_dist,  base_dist,  base_dist], dtype=np.float32)))
    scene.add(rim,  pose=_pose(np.array([ base_dist, -base_dist, -base_dist], dtype=np.float32)))

    # Render each combination of distance multiplier and angle
    for d_idx, d_mult in enumerate(dist_mults):
        dist = base_dist * float(d_mult)

        # near/far for this distance
        znear = max(1e-3, dist - 2.5 * radius)
        zfar  = dist + 2.5 * radius

        for a_idx, (elev_deg, az_deg) in enumerate(angles):
            eye_offset = spherical_to_cartesian(dist, elev_deg, az_deg)
            eye = center_np + eye_offset
            cam_pose = look_at(eye, center_np, up=(0, 1, 0))

            cam = pyrender.PerspectiveCamera(
                yfov=fovy,
                aspectRatio=aspect,
                znear=znear,
                zfar=zfar,
            )
            cam_node = scene.add(cam, pose=cam_pose)

            rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            scene.remove_node(cam_node)

            rgb = composite_on_white(rgba)
            out_path = out_dir / f"view_d{d_idx}_a{a_idx}_e{int(elev_deg)}_az{int(az_deg)}.png"
            iio.imwrite(out_path, rgb)
            print(f"Saved: {out_path}")

    renderer.delete()
    print("Done.")


if __name__ == "__main__":
    render_multi_views(
        glb_path=GLB_PATH,
        out_dir=OUT_DIR,
        width=WIDTH,
        height=HEIGHT,
        fov_y_deg=FOV_Y_DEG,
        pad_ratio=PAD_RATIO,
        dist_mults=DIST_MULTS,
        angles=ANGLES,
    )