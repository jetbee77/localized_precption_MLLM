from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_dir, load_config, read_dataset


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for XGBoost supervised model.")
    parser.add_argument("--config", default="../configs/default.yaml")
    parser.add_argument("--sample", type=int, default=300, help="Sample size for SHAP computation")
    args = parser.parse_args()

    try:
        import xgboost as xgb
        from xgboost import XGBRegressor
    except Exception as e:
        raise SystemExit("XGBoost is not installed. Please run: pip install xgboost") from e

    cfg = load_config(args.config)
    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent.parent
    dataset_xlsx = Path(cfg["paths"]["dataset_xlsx"])
    if not dataset_xlsx.is_absolute():
        dataset_xlsx = (base_dir / dataset_xlsx).resolve()
    sup_out_dir = Path(cfg["paths"]["supervised_output_dir"])
    if not sup_out_dir.is_absolute():
        sup_out_dir = (base_dir / sup_out_dir).resolve()
    out_dir = ensure_dir(sup_out_dir / "shap" / "xgb")

    df = read_dataset(dataset_xlsx)
    features = cfg["supervised"]["features"]
    targets = cfg["supervised"]["targets"]

    df = df.dropna(subset=features + targets)
    X = df[features].values

    # downsample for SHAP speed
    if len(X) > args.sample:
        df = df.sample(n=args.sample, random_state=cfg["supervised"]["random_state"])
        X = df[features].values

    try:
        import shap
    except Exception as e:
        raise SystemExit("SHAP is not installed. Please run: pip install shap") from e

    for target in targets:
        y = df[target].values
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X, y)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            # Fallback for SHAP<->XGBoost compatibility issues (base_score parsing, etc.)
            print(f"TreeExplainer failed ({e}); falling back to pred_contribs.")
            dmat = xgb.DMatrix(X, feature_names=features)
            contribs = model.get_booster().predict(dmat, pred_contribs=True)
            # last column is bias term
            shap_values = contribs[:, :-1]

        # Save SHAP values as CSV
        shap_df = pd.DataFrame(shap_values, columns=features)
        shap_df.insert(0, "id", df["id"].values if "id" in df.columns else np.arange(len(df)))
        shap_path = out_dir / f"shap_values_xgb_{target}.csv"
        shap_df.to_csv(shap_path, index=False)

        # Summary plot
        plot_path = out_dir / f"shap_summary_xgb_{target}.png"
        shap.summary_plot(shap_values, X, feature_names=features, show=False)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

        # Bar plot of mean |SHAP|
        bar_path = out_dir / f"shap_bar_xgb_{target}.png"
        shap.summary_plot(shap_values, X, feature_names=features, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=200)
        plt.close()

        print(f"Saved: {shap_path}")
        print(f"Saved: {plot_path}")
        print(f"Saved: {bar_path}")


if __name__ == "__main__":
    main()
