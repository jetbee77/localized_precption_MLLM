from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from common import ensure_dir, load_config, read_dataset


def main():
    parser = argparse.ArgumentParser(description="Plot distribution comparison between human labels and MLLM predictions.")
    parser.add_argument("--config", default="../configs/default.yaml")
    parser.add_argument("--mllm-group", default="baseline", help="Group key in mllm_groups")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_xlsx = Path(cfg["paths"]["dataset_xlsx"])
    stitched_dir = Path(cfg["paths"]["stitched_views_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["supervised_output_dir"]) / "plots" / "diagnostics")

    groups = cfg.get("mllm_groups", {})
    if args.mllm_group not in groups:
        raise SystemExit(f"Unknown mllm group: {args.mllm_group}")

    input_csv = Path(groups[args.mllm_group]["output_csv"])
    use_csv = input_csv

    df_true = read_dataset(dataset_xlsx)
    df_pred = pd.read_csv(use_csv)

    targets = cfg["supervised"]["targets"]
    merged = pd.merge(df_true, df_pred, on="id", how="inner", suffixes=("_true", "_mllm"))

    sns.set_theme(style="whitegrid")
    sns.set_palette(sns.color_palette("Blues", 4))

    for t in targets:
        true_col = f"{t}_true"
        pred_col = f"{t}_mllm"
        if true_col not in merged.columns or pred_col not in merged.columns:
            print(f"Skip {t}: missing columns")
            continue

        plt.figure(figsize=(7, 4))
        sns.histplot(merged[true_col], kde=True, stat="density", color="#9ecae1", label="Human", alpha=0.6)
        sns.histplot(merged[pred_col], kde=True, stat="density", color="#3182bd", label="MLLM", alpha=0.5)
        plt.title(f"Distribution Comparison - {t} ({args.mllm_group})")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / f"dist_compare_{args.mllm_group}_{t}.png", dpi=200)
        plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
