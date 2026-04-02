from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import ensure_dir, load_config, read_dataset

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


def spearman(a, b):
    if spearmanr is None:
        return np.nan
    return spearmanr(a, b).correlation


def main():
    parser = argparse.ArgumentParser(description="Compare MLLM vs supervised models and plot results.")
    parser.add_argument("--config", default="../configs/default.yaml")
    parser.add_argument("--only-mllm", action="store_true", help="Only plot MLLM groups")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include-expa", action="store_true", help="Include explanatory_promoptA group")
    group.add_argument("--exclude-expa", action="store_true", help="Exclude explanatory_promoptA group")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent.parent
    dataset_xlsx = Path(cfg["paths"]["dataset_xlsx"])
    if not dataset_xlsx.is_absolute():
        dataset_xlsx = (base_dir / dataset_xlsx).resolve()
    supervised_dir = Path(cfg["paths"]["supervised_output_dir"])
    if not supervised_dir.is_absolute():
        supervised_dir = (base_dir / supervised_dir).resolve()

    out_dir = ensure_dir(supervised_dir / "plots")

    # light blue palette
    sns.set_theme(style="whitegrid")

    df = read_dataset(dataset_xlsx)
    targets = cfg["supervised"]["targets"]

    # Use the supervised test IDs for fair comparison
    rf_pred_path = supervised_dir / "predictions_random_forest.csv"
    if not rf_pred_path.exists():
        raise SystemExit(f"Missing {rf_pred_path} for fair evaluation.")
    rf_pred = pd.read_csv(rf_pred_path)
    if "id" not in rf_pred.columns:
        raise SystemExit("predictions_random_forest.csv missing id column.")
    test_ids = set(rf_pred["id"].dropna().astype(int).tolist())

    # Determine MLLM groups to include
    groups = dict(cfg.get("mllm_groups", {}))
    if args.exclude_expa:
        for k in ["explanatory_promoptA", "explanatory_promoptB", "explanatory_promoptC"]:
            if k in groups:
                groups.pop(k)

    # Compute intersection ids across supervised test set and all available MLLM groups
    common_ids = set(test_ids)
    available_groups = []
    for name, info in groups.items():
        input_csv = Path(info.get("output_csv", ""))
        if not input_csv.is_absolute():
            input_csv = (base_dir / input_csv).resolve()
        if not input_csv.exists():
            print(f"Skip MLLM group {name}: missing {input_csv}")
            continue
        mllm_ids = set(pd.read_csv(input_csv)["id"].dropna().astype(int).tolist())
        common_ids &= mllm_ids
        available_groups.append(name)

    if not common_ids:
        raise SystemExit("No common ids across supervised test set and selected MLLM groups.")
    print(f"Common evaluation ids: {len(common_ids)} (groups: {', '.join(available_groups)})")

    metrics_rows = []

    # Supervised metrics on the same common IDs (includes RF/MLP/XGBoost/LightGBM if available)
    sup_files = sorted(supervised_dir.glob("predictions_*.csv"))
    for p in sup_files:
        model = p.stem.replace("predictions_", "")
        df_pred = pd.read_csv(p)
        if "id" not in df_pred.columns:
            continue
        df_pred = df_pred[df_pred["id"].isin(common_ids)]
        for t in targets:
            true_col = f"{t}_true"
            pred_col = f"{t}_pred"
            if true_col not in df_pred.columns or pred_col not in df_pred.columns:
                continue
            y_true = df_pred[true_col].values
            y_pred = df_pred[pred_col].values
            mse = mean_squared_error(y_true, y_pred)
            metrics_rows.append(
                {
                    "model": model,
                    "target": t,
                    "mae": mean_absolute_error(y_true, y_pred),
                    "mse": float(mse),
                    "rmse": float(np.sqrt(mse)),
                    "r2": r2_score(y_true, y_pred),
                    "spearman": spearman(y_true, y_pred),
                }
            )

    # MLLM metrics on the same common IDs
    for name, info in groups.items():
        input_csv = Path(info.get("output_csv", ""))
        if not input_csv.is_absolute():
            input_csv = (base_dir / input_csv).resolve()
        if not input_csv.exists():
            continue
        mllm = pd.read_csv(input_csv)
        merged = pd.merge(df, mllm, on="id", how="inner", suffixes=("_true", "_pred"))
        merged = merged[merged["id"].isin(common_ids)]
        if merged.empty:
            print(f"Skip MLLM group {name}: no overlap")
            continue
        for t in targets:
            true_col = f"{t}_true"
            if t in merged.columns:
                y_true = merged[true_col].values
                y_pred = merged[t].values
            elif f"{t}_pred" in merged.columns:
                y_true = merged[true_col].values
                y_pred = merged[f"{t}_pred"].values
            else:
                continue
            mse = mean_squared_error(y_true, y_pred)
            metrics_rows.append(
                {
                    "model": f"mllm_{name}",
                    "target": t,
                    "mae": mean_absolute_error(y_true, y_pred),
                    "mse": float(mse),
                    "rmse": float(np.sqrt(mse)),
                    "r2": r2_score(y_true, y_pred),
                    "spearman": spearman(y_true, y_pred),
                }
            )

    metrics_all = pd.DataFrame(metrics_rows)
    if args.only_mllm:
        metrics_all = metrics_all[metrics_all["model"].str.startswith("mllm_")]
    if "mse" not in metrics_all.columns:
        metrics_all["mse"] = metrics_all["rmse"] ** 2
    metrics_all.to_csv(supervised_dir / "metrics_all.csv", index=False)

    # build blue palette for model hues (default)
    model_order = metrics_all["model"].dropna().unique().tolist()
    palette = sns.color_palette("Blues", n_colors=max(3, len(model_order)))

    # Bar charts: if explanatory_promoptA is included, highlight it in blue and gray out others.
    shap_model_name = "mllm_explanatory_promoptA"
    if (not args.exclude_expa) and (shap_model_name in model_order):
        palette_bar = {m: "#B0B0B0" for m in model_order}
        palette_bar[shap_model_name] = "#4C72B0"
    else:
        palette_bar = palette

    # Plot MAE
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=metrics_all, x="target", y="mae", hue="model", palette=palette_bar)
    plt.title("MAE by Target")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_dir / "mae_by_target.png", dpi=200)
    plt.close()

    # Plot RMSE
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=metrics_all, x="target", y="rmse", hue="model", palette=palette_bar)
    plt.title("RMSE by Target")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_by_target.png", dpi=200)
    plt.close()

    # Plot R2
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=metrics_all, x="target", y="r2", hue="model", palette=palette_bar)
    plt.title("R2 by Target")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_dir / "r2_by_target.png", dpi=200)
    plt.close()

    # Plot Spearman
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=metrics_all, x="target", y="spearman", hue="model", palette=palette_bar)
    plt.title("Spearman by Target")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_dir / "spearman_by_target.png", dpi=200)
    plt.close()

    # If train-size metrics exist, plot learning curves (avg across targets)
    size_csv = supervised_dir / "metrics_by_train_size.csv"
    if size_csv.exists():
        df_size = pd.read_csv(size_csv)
        avg_size = df_size.groupby(["model", "train_size"])[["mae", "rmse", "r2", "spearman"]].mean().reset_index()
        if "mse" not in avg_size.columns:
            avg_size["mse"] = avg_size["rmse"] ** 2

        # MLLM baseline line (use calibrated baseline if available)
        mllm_baseline = metrics_all[metrics_all["model"] == "mllm_baseline"]
        if not mllm_baseline.empty:
            mllm_rmse = mllm_baseline["rmse"].mean()
            mllm_mae = mllm_baseline["mae"].mean()
            mllm_r2 = mllm_baseline["r2"].mean()
            mllm_spr = mllm_baseline["spearman"].mean()
        else:
            mllm_rmse = mllm_mae = mllm_r2 = mllm_spr = None

        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(data=avg_size, x="train_size", y="rmse", hue="model", marker="o", palette=palette)
        if mllm_rmse is not None:
            ax.axhline(mllm_rmse, linestyle="--", color="#4C72B0", label="mllm_baseline")
        plt.title("RMSE vs Train Size (Avg Across Targets)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(out_dir / "rmse_vs_train_size.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(data=avg_size, x="train_size", y="mae", hue="model", marker="o", palette=palette)
        if mllm_mae is not None:
            ax.axhline(mllm_mae, linestyle="--", color="#4C72B0", label="mllm_baseline")
        plt.title("MAE vs Train Size (Avg Across Targets)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(out_dir / "mae_vs_train_size.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(data=avg_size, x="train_size", y="r2", hue="model", marker="o", palette=palette)
        if mllm_r2 is not None:
            ax.axhline(mllm_r2, linestyle="--", color="#4C72B0", label="mllm_baseline")
        plt.title("R2 vs Train Size (Avg Across Targets)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(out_dir / "r2_vs_train_size.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(data=avg_size, x="train_size", y="spearman", hue="model", marker="o", palette=palette)
        if mllm_spr is not None:
            ax.axhline(mllm_spr, linestyle="--", color="#4C72B0", label="mllm_baseline")
        plt.title("Spearman vs Train Size (Avg Across Targets)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(out_dir / "spearman_vs_train_size.png", dpi=200)
        plt.close()

    # Heatmap: model x target for R2
    heat_r2 = metrics_all.pivot_table(index="model", columns="target", values="r2", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(heat_r2, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title("R2 Heatmap (Model x Target)")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_r2.png", dpi=200)
    plt.close()


    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
