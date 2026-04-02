from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

from common import ensure_dir, load_config, read_dataset


def main():
    parser = argparse.ArgumentParser(description="Plot error vs feature scatter with trend lines.")
    parser.add_argument("--config", default="../configs/default.yaml")
    parser.add_argument("--mllm-group", default="baseline")
    parser.add_argument("--features", default=None, help="Comma-separated feature list")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_xlsx = Path(cfg["paths"]["dataset_xlsx"])
    out_dir = ensure_dir(Path(cfg["paths"]["supervised_output_dir"]) / "plots" / "diagnostics")

    groups = cfg.get("mllm_groups", {})
    if args.mllm_group not in groups:
        raise SystemExit(f"Unknown mllm group: {args.mllm_group}")

    input_csv = Path(groups[args.mllm_group]["output_csv"])
    calibrated_csv = input_csv.with_name(input_csv.stem + "_calibrated.csv")
    use_csv = calibrated_csv if calibrated_csv.exists() else input_csv

    df_true = read_dataset(dataset_xlsx)
    df_pred = pd.read_csv(use_csv)

    targets = cfg["supervised"]["targets"]
    features = cfg["supervised"]["features"]
    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]

    merged = pd.merge(df_true, df_pred, on="id", how="inner", suffixes=("_true", "_mllm"))

    sns.set_theme(style="whitegrid")
    sns.set_palette(sns.color_palette("Blues", 5))

    # Precompute error columns
    for t in targets:
        merged[f"{t}_gap"] = merged[f"{t}_mllm"] - merged[f"{t}_true"]

    # Load RF importance if available (expects diagnostics CSV next to other supervised outputs)
    importance_map = {}
    decomp_path = (
        Path(cfg["paths"]["supervised_output_dir"]) / "diagnostics" / f"error_decomposition_{args.mllm_group}.csv"
    )
    if decomp_path.exists():
        decomp_df = pd.read_csv(decomp_path)
        decomp_df = decomp_df[decomp_df["method"] == "rf_importance"]
        for _, row in decomp_df.iterrows():
            importance_map[(str(row["target"]), str(row["feature"]))] = float(row["importance"])

    def get_importance(target: str, feature: str):
        return importance_map.get((target, feature))

    def format_importance(val):
        if val is None or val != val:
            return "RF importance: NA"
        return f"RF importance: {val:.3f}"

    def add_scatter(ax, x, y, feat, target, importance_val, fontsize=10):
        # simple linear stats
        n = len(x)
        x_mean = x.mean()
        y_mean = y.mean()
        sxx = ((x - x_mean) ** 2).sum()
        sxy = ((x - x_mean) * (y - y_mean)).sum()
        slope = sxy / sxx if sxx != 0 else 0.0
        intercept = y_mean - slope * x_mean
        y_hat = intercept + slope * x
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

        # p-value for slope
        try:
            from scipy import stats
            if n > 2 and sxx != 0:
                s_err = (ss_res / (n - 2)) ** 0.5
                t_stat = slope / (s_err / (sxx ** 0.5))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
            else:
                p_val = float("nan")
        except Exception:
            p_val = float("nan")

        sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.4, "s": 18}, line_kws={"color": "#3182bd"}, ax=ax)
        ax.axhline(0, color="#9ecae1", linestyle="--", linewidth=1)
        ax.set_title(f"Error vs {feat} ({target}, {args.mllm_group})")
        ax.set_xlabel(feat)
        ax.set_ylabel("MLLM - Human")
        direction = "MLLM ↑" if slope > 0 else ("MLLM ↓" if slope < 0 else "MLLM =")

        ax.text(
            0.02,
            0.98,
            f"Slope: {slope:.3f} ({direction})\n{format_importance(importance_val)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bdbdbd", alpha=0.8),
        )

    features_in_data = [f for f in features if f in merged.columns]

    for t in targets:
        gap_col = f"{t}_gap"
        # top-3 importance features for this target
        feat_imps = [
            (feat, get_importance(t, feat)) for feat in features_in_data if get_importance(t, feat) is not None
        ]
        feat_imps = sorted(feat_imps, key=lambda x: x[1], reverse=True)
        top3_feats = {f for f, _ in feat_imps[:3]}

        # Individual scatter plots
        for feat in features_in_data:
            x = merged[feat].astype(float).values
            y = merged[gap_col].astype(float).values

            plt.figure(figsize=(6, 4))
            add_scatter(plt.gca(), x, y, feat, t, get_importance(t, feat), fontsize=10)
            plt.tight_layout()
            plt.savefig(out_dir / f"err_scatter_{args.mllm_group}_{t}_{feat}.png", dpi=200)
            plt.close()

        # Combined figure per target
        if features_in_data:
            ncols = 3
            nrows = int(math.ceil(len(features_in_data) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.4))
            axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

            for i, feat in enumerate(features_in_data):
                ax = axes[i]
                x = merged[feat].astype(float).values
                y = merged[gap_col].astype(float).values
                add_scatter(ax, x, y, feat, t, get_importance(t, feat), fontsize=8)

                if feat in top3_feats:
                    rect = patches.Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=ax.transAxes,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                        linestyle="--",
                        clip_on=False,
                    )
                    ax.add_patch(rect)

            # remove unused axes
            for j in range(len(features_in_data), len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle(f"Error vs Features ({t}, {args.mllm_group})", y=1.02)
            fig.tight_layout()
            fig.savefig(out_dir / f"err_scatter_grid_{args.mllm_group}_{t}.png", dpi=200)
            plt.close(fig)

    print(f"Saved scatter plots to {out_dir}")


if __name__ == "__main__":
    main()
