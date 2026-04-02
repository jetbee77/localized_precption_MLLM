from __future__ import annotations

import argparse
from pathlib import Path

import joblib
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
    parser = argparse.ArgumentParser(description="Train supervised models on labeled dataset.")
    parser.add_argument("--config", default="../configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_xlsx = Path(cfg["paths"]["dataset_xlsx"])
    out_dir = ensure_dir(cfg["paths"]["supervised_output_dir"])

    df = read_dataset(dataset_xlsx)
    features = cfg["supervised"]["features"]
    targets = cfg["supervised"]["targets"]

    df = df.dropna(subset=features + targets)
    # Keep feature names for models (e.g., LightGBM) to avoid warnings.
    X = df[features]
    y_all = {t: df[t].values for t in targets}
    ids = df["id"].astype(int).values if "id" in df.columns else np.arange(len(df))

    X_train, X_test, idx_train, idx_test = train_test_split(
        X,
        np.arange(len(X)),
        test_size=cfg["supervised"]["test_size"],
        random_state=cfg["supervised"]["random_state"],
    )

    metrics_rows = []

    for model_name in available_model_names():
        preds = {"id": ids[idx_test]}
        for target in targets:
            model = build_model(model_name)
            if model is None:
                continue

            y = y_all[target]
            y_train = y[idx_train]
            y_test = y[idx_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_test, y_pred)
            sp = spearman(y_test, y_pred)

            metrics_rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "spearman": sp,
                }
            )

            preds[target + "_true"] = y_test
            preds[target + "_pred"] = y_pred

            joblib.dump(model, out_dir / f"model_{model_name}_{target}.joblib")

        pred_df = pd.DataFrame(preds)
        pred_df.to_csv(out_dir / f"predictions_{model_name}.csv", index=False)

    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_supervised.csv", index=False)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
