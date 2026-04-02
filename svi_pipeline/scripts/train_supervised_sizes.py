from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import ensure_dir, load_config, read_dataset

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


def spearman(a, b):
    if spearmanr is None:
        return np.nan
    return spearmanr(a, b).correlation


def build_model(name: str):
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    if name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800, random_state=42)),
            ]
        )
    if name == "xgboost" and XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
    if name == "lightgbm" and LGBMRegressor is not None:
        return LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42)
    return None


def available_model_names():
    names = ["random_forest", "mlp"]
    if XGBRegressor is not None:
        names.append("xgboost")
    if LGBMRegressor is not None:
        names.append("lightgbm")
    return names


def main():
    parser = argparse.ArgumentParser(description="Train supervised models under multiple train sizes.")
    parser.add_argument("--config", default="../configs/default.yaml")
    parser.add_argument("--sizes", default="10,30,50,70,100,150,200,250,300")
    parser.add_argument("--plot-metric", default="both", choices=["r2", "rmse", "both"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent.parent
    dataset_xlsx = Path(cfg["paths"]["dataset_xlsx"])
    if not dataset_xlsx.is_absolute():
        dataset_xlsx = (base_dir / dataset_xlsx).resolve()
    out_dir = Path(cfg["paths"]["supervised_output_dir"])
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
    out_dir = ensure_dir(out_dir)

    df = read_dataset(dataset_xlsx)
    features = cfg["supervised"]["features"]
    # Prefer CQ/AQ/HQ/VQ for this step; fall back to config targets if missing.
    preferred_targets = ["CQ", "AQ", "HQ", "VQ"]
    cfg_targets = cfg["supervised"]["targets"]
    if all(t in df.columns for t in preferred_targets):
        targets = preferred_targets
    else:
        targets = cfg_targets
        print(
            "Warning: CQ/AQ/HQ/VQ not all found in dataset. "
            "Falling back to targets from config."
        )

    df = df.dropna(subset=features + targets)
    # Keep feature names for models (e.g., LightGBM) to avoid warnings.
    X = df[features]
    y_all = {t: df[t].values for t in targets}

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    sizes = sorted(list(set(sizes)))

    # Fixed test set for fair comparison
    X_train_full, X_test, idx_train_full, idx_test = train_test_split(
        X,
        np.arange(len(X)),
        test_size=cfg["supervised"]["test_size"],
        random_state=cfg["supervised"]["random_state"],
    )

    test_ids = None
    if "id" in df.columns:
        test_ids = set(df.iloc[idx_test]["id"].dropna().astype(int).tolist())

    rng = np.random.default_rng(cfg["supervised"]["random_state"])

    rows = []
    for size in sizes:
        if size > len(X_train_full):
            print(f"Skip size {size}: larger than available train set {len(X_train_full)}")
            continue
        subset_idx = rng.choice(len(X_train_full), size=size, replace=False)
        X_train = X_train_full.iloc[subset_idx]

        for model_name in available_model_names():
            for target in targets:
                model = build_model(model_name)
                if model is None:
                    continue
                y = y_all[target]
                y_train = y[idx_train_full][subset_idx]
                y_test = y[idx_test]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rows.append(
                    {
                        "model": model_name,
                        "target": target,
                        "train_size": size,
                        "mae": mean_absolute_error(y_test, y_pred),
                        "rmse": float(np.sqrt(mse)),
                        "r2": r2_score(y_test, y_pred),
                        "spearman": spearman(y_test, y_pred),
                    }
                )

    out_csv = out_dir / "metrics_by_train_size.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # MLLM dashed lines for each target
    mllm_groups = dict(cfg.get("mllm_groups", {}))
    mllm_order = ["baseline", "anchor_prompt", "rubric_prompt", "local_prompt"]
    # Use higher-contrast blue shades for MLLM dashed references
    mllm_colors = ["#0b2e8a", "#1f5fbf", "#3c8dd9", "#8bbcf0"]
    mllm_alpha = 0.8
    mllm_metric_by_target: dict[str, dict[str, dict[str, float]]] = {g: {} for g in mllm_order}
    if test_ids is None:
        print("Warning: dataset missing id column; skip MLLM R2 dashed lines.")
    else:
        for name, color in zip(mllm_order, mllm_colors):
            info = mllm_groups.get(name)
            if not info:
                continue
            input_csv = Path(info.get("output_csv", ""))
            if not input_csv.is_absolute():
                input_csv = (base_dir / input_csv).resolve()
            if not input_csv.exists():
                print(f"Skip MLLM group {name}: missing {input_csv}")
                continue
            mllm = pd.read_csv(input_csv)
            if "id" not in mllm.columns:
                print(f"Skip MLLM group {name}: missing id column in {input_csv}")
                continue
            merged = pd.merge(
                df[["id"] + targets],
                mllm,
                on="id",
                how="inner",
                suffixes=("_true", "_pred"),
            )
            merged = merged[merged["id"].isin(test_ids)]
            if merged.empty:
                print(f"Skip MLLM group {name}: no overlap with test ids")
                continue
            for t in targets:
                true_col = f"{t}_true" if f"{t}_true" in merged.columns else t
                if t in merged.columns:
                    y_pred = merged[t].values
                elif f"{t}_pred" in merged.columns:
                    y_pred = merged[f"{t}_pred"].values
                else:
                    continue
                y_true = merged[true_col].values
                mllm_metric_by_target[name].setdefault(t, {})
                mllm_metric_by_target[name][t]["r2"] = r2_score(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                mllm_metric_by_target[name][t]["rmse"] = float(np.sqrt(mse))

    # Plot metric vs train size for each target (separate figures)
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.lines import Line2D

        df_plot = pd.DataFrame(rows)
        plot_dir = ensure_dir(out_dir / "plots")
        sns.set_theme(style="whitegrid")
        base_font = float(mpl.rcParams.get("font.size", 10))
        title_fs = base_font + 3
        label_fs = base_font + 3
        legend_fs = base_font + 3

        plot_metrics = ["r2", "rmse"] if args.plot_metric == "both" else [args.plot_metric]

        for metric_tag in plot_metrics:
            for target in targets:
                df_t = df_plot[df_plot["target"] == target]
                if df_t.empty:
                    continue
                plt.figure(figsize=(6, 6))
                ax = sns.lineplot(
                    data=df_t,
                    x="train_size",
                    y=metric_tag,
                    hue="model",
                    marker="o",
                    zorder=3,
                )
                ax.set_xticks(sizes)
                ax.set_xticklabels([str(s) for s in sizes])
                mllm_legend = []
                for name, color in zip(mllm_order, mllm_colors):
                    metric_val = mllm_metric_by_target.get(name, {}).get(target, {}).get(metric_tag)
                    if metric_val is None:
                        continue
                    ax.axhline(
                        metric_val,
                        linestyle="--",
                        color=color,
                        linewidth=1.6,
                        alpha=mllm_alpha,
                        zorder=1,
                    )
                    mllm_legend.append(
                        Line2D([0], [0], color=color, linestyle="--", label=f"mllm_{name}", alpha=mllm_alpha)
                    )
                metric_title = "R2" if metric_tag == "r2" else "RMSE"
                plt.title(f"{metric_title} vs Train Size ({target})", fontsize=title_fs)
                ax.set_xlabel("train_size", fontsize=label_fs)
                ax.set_ylabel(metric_title, fontsize=label_fs)
                ax.tick_params(axis="both", labelsize=label_fs)
                handles, labels = ax.get_legend_handles_labels()
                # Append MLLM dashed-line legend entries
                handles.extend(mllm_legend)
                labels.extend([h.get_label() for h in mllm_legend])
                # De-duplicate while preserving order
                seen = set()
                dedup_handles = []
                dedup_labels = []
                for h, l in zip(handles, labels):
                    if l in seen:
                        continue
                    seen.add(l)
                    dedup_handles.append(h)
                    dedup_labels.append(l)
                # Remove legend from main figure
                if ax.legend_ is not None:
                    ax.legend_.remove()
                plt.tight_layout()
                plt.savefig(plot_dir / f"{metric_tag}_vs_train_size_{target}.png", dpi=200)
                plt.close()

                # Save legend as separate single-row figure
                if dedup_handles:
                    legend_fig = plt.figure(figsize=(max(6, 0.8 * len(dedup_handles)), 0.8))
                    legend_fig.legend(
                        dedup_handles,
                        dedup_labels,
                        loc="center",
                        ncol=len(dedup_handles),
                        frameon=False,
                        fontsize=legend_fs,
                    )
                    legend_fig.tight_layout(pad=0.2)
                    legend_fig.savefig(
                        plot_dir / f"{metric_tag}_vs_train_size_{target}_legend.png",
                        dpi=200,
                        bbox_inches="tight",
                        pad_inches=0.05,
                    )
                    plt.close(legend_fig)
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()
