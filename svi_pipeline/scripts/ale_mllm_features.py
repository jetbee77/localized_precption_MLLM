#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    MLPRegressor = None
    Pipeline = None
    StandardScaler = None

try:
    from common import load_config, read_dataset
except Exception:  # pragma: no cover
    load_config = None
    read_dataset = None


def build_feature_table(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    # Drop obvious non-feature columns
    drop_cols = {"OBJECTID", "id", "x", "y", "CQ", "AQ", "HQ", "VQ"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    feat_df = df[["id"] + feat_cols].copy()
    return feat_df, feat_cols


def find_mllm_files(root: Path, pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(str(root / pattern), recursive=True)]
    paths = [p for p in paths if p.name.startswith("mllm_") and not p.name.startswith("._")]
    return sorted(paths)


def ale_1d(model, X: pd.DataFrame, feature: str, bins: int = 20, edges: np.ndarray | None = None):
    x = X[feature].values.astype(float)
    if edges is None:
        qs = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(x, qs))
    if len(edges) < 3:
        return None
    k = len(edges) - 1
    effects = np.zeros(k)
    for j in range(k):
        if j == k - 1:
            mask = (x >= edges[j]) & (x <= edges[j + 1])
        else:
            mask = (x >= edges[j]) & (x < edges[j + 1])
        if not np.any(mask):
            effects[j] = 0.0
            continue
        X_low = X.loc[mask].copy()
        X_high = X.loc[mask].copy()
        X_low[feature] = edges[j]
        X_high[feature] = edges[j + 1]
        pred_high = model.predict(X_high)
        pred_low = model.predict(X_low)
        effects[j] = float(np.mean(pred_high - pred_low))
    ale = np.cumsum(effects)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.digitize(x, edges[1:-1], right=False)
    ale_centered = ale - np.mean(ale[bin_idx])
    return centers, ale_centered


def plot_ale_grid(out_path: Path, feature_names: list[str], ale_dict: dict, title: str):
    n = len(feature_names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for i, feat in enumerate(feature_names):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        data = ale_dict.get(feat)
        if data is None:
            ax.text(0.5, 0.5, "NA", ha="center", va="center", fontsize=10)
            ax.set_axis_off()
            continue
        x, y = data
        ax.plot(x, y, color="#2C7FB8", linewidth=2)
        ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
    # hide empty subplots
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_axis_off()
    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ale_overlay_grid(
    out_path: Path,
    feature_names: list[str],
    ale_by_group: dict,
    title: str,
    group_colors: dict | None = None,
    legend_ncol: int | None = None,
):
    groups = list(ale_by_group.keys())
    n = len(feature_names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    if group_colors is None:
        cmap = plt.get_cmap("tab10")
        colors = {g: cmap(i % 10) for i, g in enumerate(groups)}
    else:
        colors = group_colors
    for i, feat in enumerate(feature_names):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        has_any = False
        for g in groups:
            data = ale_by_group[g].get(feat)
            if data is None:
                continue
            x, y = data
            ax.plot(x, y, color=colors[g], linewidth=2, label=g)
            has_any = True
        if not has_any:
            ax.text(0.5, 0.5, "NA", ha="center", va="center", fontsize=10)
            ax.set_axis_off()
            continue
        ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_axis_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        # Save legend as a separate figure
        legend_fig = plt.figure(figsize=(6, 0.8))
        legend_fig.legend(
            handles,
            labels,
            loc="center",
            ncol=legend_ncol if legend_ncol is not None else min(4, len(groups)),
            fontsize=9,
            frameon=False,
        )
        legend_path = out_path.with_name(out_path.stem + "_legend.png")
        legend_path.parent.mkdir(parents=True, exist_ok=True)
        legend_fig.savefig(legend_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(legend_fig)
    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_supervised_model(name: str):
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    if name == "mlp" and MLPRegressor is not None:
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


def main():
    ap = argparse.ArgumentParser(description="Compute ALE curves for MLLM groups using existing tabular features.")
    ap.add_argument("--features-xlsx", default="/Volumes/t7/python_file/LLM_SVIs_exp/data/数据集.xlsx")
    ap.add_argument("--mllm-glob", default="outputs/**/mllm_*.csv")
    ap.add_argument("--outdir", default="/Volumes/t7/python_file/LLM_SVIs_exp/results/ale_curves")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--groups",
        nargs="+",
        default=["mllm_baseline", "mllm_anchor_prompt", "mllm_rubric_prompt", "mllm_local_prompt"],
        help="Which MLLM csv basenames to include (without .csv).",
    )
    ap.add_argument(
        "--per-group",
        action="store_true",
        help="Also output per-group ALE grids (one group per figure).",
    )
    ap.add_argument(
        "--targets",
        nargs="+",
        default=["CQ", "AQ", "HQ", "VQ"],
        help="Targets to run (e.g., CQ AQ HQ VQ).",
    )
    ap.add_argument(
        "--supervised",
        action="store_true",
        help="Run supervised model ALE overlay (mlp/rf/xgboost/lightgbm) instead of MLLM groups.",
    )
    ap.add_argument(
        "--overlay-both",
        action="store_true",
        help="Overlay MLLM groups and supervised models together (blue for MLLM, red for supervised).",
    )
    ap.add_argument(
        "--supervised-models",
        nargs="+",
        default=["mlp", "random_forest", "xgboost", "lightgbm"],
        help="Supervised models to include (mlp, random_forest, xgboost, lightgbm).",
    )
    ap.add_argument(
        "--config",
        default="/Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml",
        help="Config path (used for supervised features list).",
    )
    args = ap.parse_args()

    features_xlsx = Path(args.features_xlsx)
    outdir = Path(args.outdir)
    root = Path("/Volumes/t7/python_file/LLM_SVIs_exp")

    if args.supervised:
        if load_config is None or read_dataset is None:
            print("Supervised mode requires common.py (load_config/read_dataset).")
            return
        cfg = load_config(args.config)
        cfg_path = Path(args.config).resolve()
        # Paths in config are authored relative to the svi_pipeline directory
        base_dir = cfg_path.parent.parent
        dataset_xlsx = Path(cfg["paths"]["dataset_xlsx"])
        if not dataset_xlsx.is_absolute():
            dataset_xlsx = (base_dir / dataset_xlsx).resolve()
        df = read_dataset(dataset_xlsx)
        feat_cols = cfg["supervised"]["features"]
        targets = args.targets

        df = df.dropna(subset=feat_cols + targets)
        X = df[feat_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
        df = df.loc[X.index]

        edges_by_feat = {}
        for feat in feat_cols:
            x = X[feat].values.astype(float)
            qs = np.linspace(0, 1, args.bins + 1)
            edges = np.unique(np.quantile(x, qs))
            edges_by_feat[feat] = edges

        model_names = ["mlp", "random_forest", "xgboost", "lightgbm"]
        for target in targets:
            ale_by_group = {}
            y = pd.to_numeric(df[target], errors="coerce")
            keep = y.notna()
            X_use = X.loc[keep].copy()
            y_use = y.loc[keep].values
            if len(y_use) < 50:
                print("Skip (too few rows):", target, len(y_use))
                continue

            for name in model_names:
                model = build_supervised_model(name)
                if model is None:
                    print("Skip (model unavailable):", name)
                    continue
                model.fit(X_use, y_use)
                ale_dict = {}
                for feat in feat_cols:
                    edges = edges_by_feat.get(feat)
                    ale = ale_1d(model, X_use, feat, bins=args.bins, edges=edges)
                    ale_dict[feat] = ale
                ale_by_group[name] = ale_dict

            if not ale_by_group:
                continue
            title = f"ALE Overlay - Supervised - {target}"
            out_path = outdir / "overlay_supervised" / f"ale_overlay_supervised_{target}.png"
            plot_ale_overlay_grid(out_path, feat_cols, ale_by_group, title)
            print("Saved:", out_path)
        return

    if args.overlay_both:
        if load_config is None or read_dataset is None:
            print("Overlay-both mode requires common.py (load_config/read_dataset).")
            return
        # load MLLM groups
        feat_df, feat_cols = build_feature_table(features_xlsx)
        mllm_files = find_mllm_files(root, args.mllm_glob)
        mllm_files = [p for p in mllm_files if p.stem in set(args.groups)]
        if not mllm_files:
            print("No MLLM files found with pattern:", args.mllm_glob)
            return

        mllm_groups = {}
        for csv_path in mllm_files:
            df = pd.read_csv(csv_path)
            if "id" not in df.columns:
                print("Skip (missing id):", csv_path)
                continue
            group_name = csv_path.stem
            merged = df.merge(feat_df, on="id", how="inner")
            if merged.empty:
                print("Skip (no merged rows):", csv_path)
                continue
            X = merged[feat_cols].copy()
            X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
            mllm_groups[group_name] = {"merged": merged, "X": X}

        # load supervised dataset
        cfg = load_config(args.config)
        cfg_path = Path(args.config).resolve()
        base_dir = cfg_path.parent.parent
        dataset_xlsx = Path(cfg["paths"]["dataset_xlsx"])
        if not dataset_xlsx.is_absolute():
            dataset_xlsx = (base_dir / dataset_xlsx).resolve()
        sup_df = read_dataset(dataset_xlsx)
        sup_feat_cols = cfg["supervised"]["features"]
        targets = args.targets
        sup_df = sup_df.dropna(subset=sup_feat_cols + targets)
        sup_X = sup_df[sup_feat_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
        sup_df = sup_df.loc[sup_X.index]

        # use MLLM feature set for overlay
        all_X = pd.concat([g["X"] for g in mllm_groups.values()], axis=0)
        edges_by_feat = {}
        for feat in feat_cols:
            x = all_X[feat].values.astype(float)
            qs = np.linspace(0, 1, args.bins + 1)
            edges = np.unique(np.quantile(x, qs))
            edges_by_feat[feat] = edges

        blue_cmap = plt.get_cmap("Blues")
        gray_levels = [0.25, 0.45, 0.65, 0.8]
        red_color = (0.839, 0.152, 0.156, 1.0)

        for target in targets:
            ale_by_group = {}
            group_colors = {}

            # MLLM groups (blue)
            mllm_names = list(mllm_groups.keys())
            for i, group_name in enumerate(mllm_names):
                g = mllm_groups[group_name]
                merged = g["merged"]
                X = g["X"]
                if target not in merged.columns:
                    continue
                y = pd.to_numeric(merged.loc[X.index, target], errors="coerce")
                keep = y.notna()
                X_use = X.loc[keep].copy()
                y_use = y.loc[keep].values
                if len(y_use) < 50:
                    print("Skip (too few rows):", group_name, target, len(y_use))
                    continue
                model = RandomForestRegressor(
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    n_jobs=-1,
                )
                model.fit(X_use, y_use)
                ale_dict = {}
                for feat in feat_cols:
                    edges = edges_by_feat.get(feat)
                    ale = ale_1d(model, X_use, feat, bins=args.bins, edges=edges)
                    ale_dict[feat] = ale
                ale_by_group[group_name] = ale_dict
                group_colors[group_name] = blue_cmap(0.45 + 0.5 * (i / max(1, len(mllm_names) - 1)))

            # supervised models (red)
            sup_names = args.supervised_models
            y_sup = pd.to_numeric(sup_df[target], errors="coerce")
            keep_sup = y_sup.notna()
            X_use_sup = sup_X.loc[keep_sup].copy()
            y_use_sup = y_sup.loc[keep_sup].values
            if len(y_use_sup) >= 50:
                for j, name in enumerate(sup_names):
                    model = build_supervised_model(name)
                    if model is None:
                        print("Skip (model unavailable):", name)
                        continue
                    model.fit(X_use_sup, y_use_sup)
                    ale_dict = {}
                    for feat in feat_cols:
                        edges = edges_by_feat.get(feat)
                        ale = ale_1d(model, X_use_sup, feat, bins=args.bins, edges=edges)
                        ale_dict[feat] = ale
                    label = name if name == "xgboost" else f"sup_{name}"
                    ale_by_group[label] = ale_dict
                    if name == "xgboost":
                        group_colors[label] = red_color
                    else:
                        gray = gray_levels[j % len(gray_levels)]
                        group_colors[label] = (gray, gray, gray, 1.0)

            if not ale_by_group:
                continue
            title = f"ALE Overlay - MLLM vs Supervised - {target}"
            out_path = outdir / "overlay_both" / f"ale_overlay_both_{target}.png"
            plot_ale_overlay_grid(
                out_path,
                feat_cols,
                ale_by_group,
                title,
                group_colors=group_colors,
                legend_ncol=len(ale_by_group),
            )
            print("Saved:", out_path)
        return

    feat_df, feat_cols = build_feature_table(features_xlsx)
    mllm_files = find_mllm_files(root, args.mllm_glob)
    mllm_files = [p for p in mllm_files if p.stem in set(args.groups)]
    if not mllm_files:
        print("No MLLM files found with pattern:", args.mllm_glob)
        return

    # load all groups
    groups = {}
    for csv_path in mllm_files:
        df = pd.read_csv(csv_path)
        if "id" not in df.columns:
            print("Skip (missing id):", csv_path)
            continue
        group_name = csv_path.stem
        merged = df.merge(feat_df, on="id", how="inner")
        if merged.empty:
            print("Skip (no merged rows):", csv_path)
            continue
        X = merged[feat_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=0, how="any")
        groups[group_name] = {
            "csv": csv_path,
            "merged": merged,
            "X": X,
        }

    if not groups:
        print("No usable MLLM groups after merge.")
        return

    # global edges per feature for overlay (computed across all groups)
    all_X = pd.concat([g["X"] for g in groups.values()], axis=0)
    edges_by_feat = {}
    for feat in feat_cols:
        x = all_X[feat].values.astype(float)
        qs = np.linspace(0, 1, args.bins + 1)
        edges = np.unique(np.quantile(x, qs))
        edges_by_feat[feat] = edges

    for target in args.targets:
        ale_by_group = {}
        for group_name, g in groups.items():
            merged = g["merged"]
            X = g["X"]
            if target not in merged.columns:
                continue
            y = pd.to_numeric(merged.loc[X.index, target], errors="coerce")
            keep = y.notna()
            X_use = X.loc[keep].copy()
            y_use = y.loc[keep].values
            if len(y_use) < 50:
                print("Skip (too few rows):", group_name, target, len(y_use))
                continue

            model = RandomForestRegressor(
                n_estimators=args.n_estimators,
                random_state=args.random_state,
                n_jobs=-1,
            )
            model.fit(X_use, y_use)

            ale_dict = {}
            for feat in feat_cols:
                edges = edges_by_feat.get(feat)
                ale = ale_1d(model, X_use, feat, bins=args.bins, edges=edges)
                ale_dict[feat] = ale
            ale_by_group[group_name] = ale_dict

            if args.per_group:
                title = f"ALE - {group_name} - {target}"
                out_path = outdir / "per_group" / group_name / f"ale_{target}.png"
                plot_ale_grid(out_path, feat_cols, ale_dict, title)
                print("Saved:", out_path)

        if not ale_by_group:
            continue
        title = f"ALE Overlay - {target}"
        out_path = outdir / "overlay" / f"ale_overlay_{target}.png"
        plot_ale_overlay_grid(out_path, feat_cols, ale_by_group, title)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
