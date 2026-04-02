from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import ensure_dir, id_to_filename, infer_id_width, load_config, read_dataset


def main():
    parser = argparse.ArgumentParser(description="Filter labeled panorama images by id and copy/symlink them.")
    parser.add_argument("--config", default="../configs/default.yaml")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--ext", default=".jpg", help="Image extension, default .jpg")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_xlsx = Path(cfg["paths"]["dataset_xlsx"])
    svi_all_dir = Path(cfg["paths"]["svi_all_dir"])
    labeled_pano_dir = ensure_dir(cfg["paths"]["labeled_pano_dir"])

    df = read_dataset(dataset_xlsx)
    if "id" not in df.columns:
        raise SystemExit("Column 'id' not found in dataset.")

    # Files are 5-digit ids like 00257.jpg
    width = 5
    all_files = {p.name for p in svi_all_dir.iterdir() if p.is_file()}

    records = []
    for img_id in df["id"].dropna().astype(int).tolist():
        name = id_to_filename(img_id, width, args.ext)
        src = svi_all_dir / name
        status = "ok" if src.name in all_files else "missing"
        dst = labeled_pano_dir / name
        if status == "ok":
            if dst.exists():
                status = "exists"
            else:
                if args.mode == "copy":
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src)
        records.append({"id": img_id, "filename": name, "status": status})

    out_csv = labeled_pano_dir / "labeled_manifest.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved manifest: {out_csv}")


if __name__ == "__main__":
    main()
