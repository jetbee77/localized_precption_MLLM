from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from openai import OpenAI

from common import ensure_dir, load_config, read_dataset, safe_json_loads
from pano_to_stitched import pano_to_views, stitch_views
from mllm_perception import build_prompt


def encode_image_from_bgr(image_bgr: np.ndarray) -> str:
    # encode to PNG bytes then base64
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")]


def main():
    parser = argparse.ArgumentParser(description="Run MLLM baseline on all SVIs and save XLSX with dataset schema.")
    parser.add_argument("--config", default="../configs/default.yaml")
    parser.add_argument("--output", default="../outputs/mllm_baseline_full.xlsx")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_xlsx = Path(cfg["paths"]["dataset_xlsx"])
    svi_all_dir = Path(cfg["paths"]["svi_all_dir"])

    fov = cfg["pano"]["fov_deg"]
    size = cfg["pano"]["out_size"]
    pitch = cfg["pano"]["pitch_deg"]

    out_path = Path(args.output)
    ensure_dir(out_path.parent)

    # Use dataset schema for output columns
    df_schema = read_dataset(dataset_xlsx)
    columns = df_schema.columns.tolist()

    # Load existing output if resume
    existing = None
    done_ids = set()
    if args.resume and out_path.exists():
        existing = pd.read_excel(out_path, engine="openpyxl")
        if "id" in existing.columns:
            done_ids = set(existing["id"].dropna().astype(int).tolist())

    client = OpenAI(base_url="https://api.openai-proxy.org/v1")
    prompt = build_prompt("baseline")

    rows = []
    images = list_images(svi_all_dir)
    if args.limit:
        images = images[: args.limit]

    for img_path in images:
        img_id_str = img_path.stem
        if not img_id_str.isdigit():
            continue
        img_id = int(img_id_str)
        if img_id in done_ids:
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skip unreadable: {img_path}")
            continue

        views = pano_to_views(image, fov_deg=fov, out_w=size, out_h=size, pitch_deg=pitch)
        stitched = stitch_views(views)
        b64 = encode_image_from_bgr(stitched)

        try:
            response = client.responses.create(
                model=cfg["mllm"]["model"],
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                        ],
                    }
                ],
                max_output_tokens=cfg["mllm"].get("max_output_tokens", 200),
                temperature=cfg["mllm"].get("temperature", 0.2),
            )
            text = response.output_text.strip()
            data = safe_json_loads(text)
            row = {c: None for c in columns}
            if "id" in row:
                row["id"] = img_id
            if "CQ" in row:
                row["CQ"] = data.get("CQ")
            if "AQ" in row:
                row["AQ"] = data.get("AQ")
            if "HQ" in row:
                row["HQ"] = data.get("HQ")
            if "VQ" in row:
                row["VQ"] = data.get("VQ")
            rows.append(row)
            print(f"Scored: {img_path.name}")
        except Exception as e:
            print(f"Failed: {img_path.name} -> {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Merge with existing if resume
    df_new = pd.DataFrame(rows)
    if existing is not None and not existing.empty:
        df_out = pd.concat([existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
