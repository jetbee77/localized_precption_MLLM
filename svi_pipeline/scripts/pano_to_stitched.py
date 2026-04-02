from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from common import ensure_dir, load_config


def _build_perspective_map(width, height, fov_deg, yaw_deg, pitch_deg, pano_w, pano_h):
    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, -y)

    z = np.ones_like(xv) / math.tan(fov / 2.0)

    dirs = np.stack([xv, yv, z], axis=-1)
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs / norm

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)

    rot_y = np.array(
        [[cos_y, 0.0, -sin_y], [0.0, 1.0, 0.0], [sin_y, 0.0, cos_y]],
        dtype=np.float32,
    )
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_p, -sin_p], [0.0, sin_p, cos_p]],
        dtype=np.float32,
    )

    dirs = dirs @ rot_y.T
    dirs = dirs @ rot_x.T

    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))

    map_x = (lon + math.pi) / (2.0 * math.pi) * pano_w
    map_y = (math.pi / 2.0 - lat) / math.pi * pano_h

    return map_x.astype(np.float32), map_y.astype(np.float32)


def pano_to_views(image_bgr, fov_deg=90, out_w=None, out_h=None, pitch_deg=0.0):
    pano_h, pano_w = image_bgr.shape[:2]

    if out_w is None and out_h is None:
        out_h = max(512, pano_h // 2)
        out_w = out_h
    elif out_w is None:
        out_w = out_h
    elif out_h is None:
        out_h = out_w

    yaws = [0, 90, 180, 270]
    results = {}
    for yaw in yaws:
        map_x, map_y = _build_perspective_map(
            out_w, out_h, fov_deg, yaw, pitch_deg, pano_w, pano_h
        )
        view = cv2.remap(
            image_bgr,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        results[yaw] = view

    return results


def stitch_views(views):
    # Order: 0 (front) top-left, 90 (right) top-right, 180 (back) bottom-right, 270 (left) bottom-left
    v0 = views[0]
    v90 = views[90]
    v180 = views[180]
    v270 = views[270]

    top = cv2.hconcat([v0, v90])
    bottom = cv2.hconcat([v270, v180])
    return cv2.vconcat([top, bottom])


def is_image(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Convert panoramas to stitched 4-view images.")
    parser.add_argument("--config", default="../configs/default.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--pitch", type=float, default=None)
    parser.add_argument("--save-views", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_dir = Path(args.input or cfg["paths"]["labeled_pano_dir"])
    output_dir = ensure_dir(args.output or cfg["paths"]["stitched_views_dir"])

    fov = args.fov if args.fov is not None else cfg["pano"]["fov_deg"]
    size = args.size if args.size is not None else cfg["pano"]["out_size"]
    pitch = args.pitch if args.pitch is not None else cfg["pano"]["pitch_deg"]

    images = [p for p in sorted(input_dir.iterdir()) if p.is_file() and is_image(p)]
    if not images:
        print(f"No image files found in {input_dir}")
        return

    for img_path in images:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        views = pano_to_views(image, fov_deg=fov, out_w=size, out_h=size, pitch_deg=pitch)
        stitched = stitch_views(views)

        out_path = output_dir / f"{img_path.stem}_stitched.jpg"
        cv2.imwrite(str(out_path), stitched)

        if args.save_views:
            for yaw, view in views.items():
                view_path = output_dir / f"{img_path.stem}_{yaw:03d}.jpg"
                cv2.imwrite(str(view_path), view)

        print(f"Processed: {img_path.name}")


if __name__ == "__main__":
    main()
