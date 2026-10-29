"""
Train, compare and save the price model.

    python -m src.models.train_models            # 10-fold CV of every candidate, save the best
    python -m src.models.train_models --quick    # 3 folds, 2 small models (CI smoke test)

Every candidate is scored with the same 10-fold cross-validation on the
de-duplicated data. Models learn log1p(price); errors are reported on the
original ₹ crore scale. The best model by MAE is refit on all the data and
saved with its metrics, its calibrated price-range widths and the library
versions it was trained with (models/metrics.json).

Price ranges are split-conformal intervals built from the out-of-fold
residuals, calibrated separately for flats and houses (Mondrian conformal),
because house prices are much harder to pin down. Their coverage is checked
honestly: each fold's range width is calibrated on the other folds only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.make_dataset import CATEGORICAL, FEATURES, NUMERIC, TARGET, load_dataset  # noqa: E402

MODELS_DIR = ROOT / "models"
MODEL_FILE = MODELS_DIR / "price_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
SEED = 42
INTERVAL_LEVELS = (0.8, 0.9)
LOW_CARDINALITY = [c for c in CATEGORICAL if c != "sector"]
# Cheapest to ship first (file size, load time, prediction speed). The chosen
# model is the first of these whose CV MAE is within one standard error of
# the best model's — differences smaller than that are fold-to-fold noise.
DEPLOYMENT_PREFERENCE = ["ridge", "xgboost", "lightgbm", "extra_trees", "random_forest", "shipped_random_forest"]


def preprocessor(kind: str) -> ColumnTransformer:
    """linear: one-hot everything + scaled numbers. tree: target-encode the 104
    sectors (cross-fitted inside TargetEncoder), one-hot the rest."""
    if kind == "linear":
        return ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])
    if kind == "shipped":  # the original app's encoding, kept as a reference point
        return ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ])
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("sector", TargetEncoder(target_type="continuous", random_state=SEED), ["sector"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW_CARDINALITY),
    ])


def candidates(quick: bool = False) -> dict[str, tuple[str, object]]:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    if quick:
        return {
            "ridge": ("linear", Ridge(alpha=1.0)),
            "lightgbm": ("tree", LGBMRegressor(n_estimators=150, learning_rate=0.1, random_state=SEED, verbose=-1)),
        }
    return {
        "ridge": ("linear", Ridge(alpha=1.0)),
        "shipped_random_forest": ("shipped", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)),
        "random_forest": ("tree", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
        "extra_trees": ("tree", ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
        "xgboost": ("tree", XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                                         colsample_bytree=0.8, random_state=SEED, n_jobs=-1)),
        "lightgbm": ("tree", LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                           subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)),
    }


def build_pipeline(kind: str, estimator) -> Pipeline:
    return Pipeline([("preprocess", preprocessor(kind)), ("model", estimator)])


def regression_metrics(price: np.ndarray, pred_price: np.ndarray) -> dict[str, float]:
    ape = np.abs(pred_price - price) / price
    return {
        "mae_cr": round(float(mean_absolute_error(price, pred_price)), 4),
        "rmse_cr": round(float(np.sqrt(np.mean((pred_price - price) ** 2))), 4),
        "median_ape": round(float(np.median(ape)), 4),
        "mape": round(float(np.mean(ape)), 4),
        "r2_log": round(float(r2_score(np.log1p(price), np.log1p(pred_price))), 4),
        "r2": round(float(r2_score(price, pred_price)), 4),
    }


def conformal_quantile(abs_residuals: np.ndarray, level: float) -> float:
    """Finite-sample split-conformal quantile of |log residuals|."""
    n = len(abs_residuals)
    rank = min(int(np.ceil((n + 1) * level)), n)
    return float(np.sort(abs_residuals)[rank - 1])


def interval_widths(residuals: np.ndarray, groups: np.ndarray, level: float) -> dict[str, float]:
    """Half-width (in log space) of the price range, per property type."""
    return {str(g): conformal_quantile(np.abs(residuals[groups == g]), level) for g in np.unique(groups)}


def interval_coverage(residuals: np.ndarray, groups: np.ndarray, folds: np.ndarray, level: float) -> dict:
    """Coverage when each fold's width is calibrated on the other folds only."""
    covered = np.zeros(len(residuals), dtype=bool)
    for fold in np.unique(folds):
        test, calib = folds == fold, folds != fold
        widths = interval_widths(residuals[calib], groups[calib], level)
        q = np.array([widths.get(str(g), np.inf) for g in groups[test]])
        covered[test] = np.abs(residuals[test]) <= q
    return {
        "overall": round(float(covered.mean()), 3),
        **{str(g): round(float(covered[groups == g].mean()), 3) for g in np.unique(groups)},
    }


def evaluate(df: pd.DataFrame, models: dict, n_splits: int) -> tuple[dict, dict, np.ndarray]:
    """Out-of-fold predictions and metrics for every candidate."""
    X, y_log = df[FEATURES], np.log1p(df[TARGET].to_numpy())
    cv = KFold(n_splits, shuffle=True, random_state=SEED)
    folds = np.empty(len(df), dtype=int)
    for i, (_, test_idx) in enumerate(cv.split(X)):
        folds[test_idx] = i

    results, oof = {}, {}
    for name, (kind, estimator) in models.items():
        start = time.perf_counter()
        pred_log = cross_val_predict(build_pipeline(kind, estimator), X, y_log, cv=cv)
        oof[name] = pred_log
        price, pred_price = df[TARGET].to_numpy(), np.expm1(pred_log)
        fold_mae = np.array([np.mean(np.abs(pred_price[folds == f] - price[folds == f])) for f in range(n_splits)])
        results[name] = {**regression_metrics(price, pred_price),
                         "mae_cr_fold_std": round(float(fold_mae.std(ddof=1)), 4),
                         "mae_cr_se": round(float(fold_mae.std(ddof=1) / np.sqrt(n_splits)), 4),
                         "cv_seconds": round(time.perf_counter() - start, 1)}
        print(f"  {name:22s} MAE {results[name]['mae_cr']:.3f} Cr   median error {results[name]['median_ape']:.1%}   "
              f"R² (log) {results[name]['r2_log']:.3f}   ({results[name]['cv_seconds']}s)")
    return results, oof, folds


def select_model(results: dict) -> tuple[str, dict]:
    """One-standard-error rule: the cheapest model within one SE of the best MAE."""
    best = min(results, key=lambda name: results[name]["mae_cr"])
    threshold = results[best]["mae_cr"] + results[best]["mae_cr_se"]
    eligible = [n for n in results if results[n]["mae_cr"] <= threshold]
    ranked = sorted(eligible, key=lambda n: DEPLOYMENT_PREFERENCE.index(n) if n in DEPLOYMENT_PREFERENCE else 99)
    return ranked[0], {
        "rule": "cheapest model whose CV MAE is within one standard error of the best",
        "lowest_mae_model": best,
        "threshold_mae_cr": round(threshold, 4),
        "eligible": ranked,
    }


def train(quick: bool = False, out_dir: Path = MODELS_DIR) -> dict:
    df = load_dataset()
    raw_rows = len(pd.read_csv(ROOT / "data" / "processed" / "gurgaon_properties_post_feature_selection_v2.csv"))
    models = candidates(quick)
    n_splits = 3 if quick else 10
    print(f"Cross-validating {len(models)} models ({n_splits} folds, {len(df)} rows)")
    results, oof, folds = evaluate(df, models, n_splits)

    best, selection = select_model(results)
    residuals = np.log1p(df[TARGET].to_numpy()) - oof[best]
    groups = df["property_type"].to_numpy()
    intervals = {
        f"{int(level * 100)}": {
            "log_half_width": interval_widths(residuals, groups, level),
            "coverage": interval_coverage(residuals, groups, folds, level),
        }
        for level in INTERVAL_LEVELS
    }

    kind, estimator = models[best]
    final = build_pipeline(kind, estimator).fit(df[FEATURES], np.log1p(df[TARGET]))
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out_dir / MODEL_FILE.name, compress=3)

    import lightgbm
    import xgboost

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model": best,
        "selection": selection,
        "cv_folds": n_splits,
        "data": {"rows_raw": raw_rows, "rows_after_dedup": len(df), "duplicates_dropped": raw_rows - len(df),
                 "flats": int((groups == "flat").sum()), "houses": int((groups == "house").sum())},
        "target": "log1p(price in ₹ crore); metrics on the crore scale",
        "models": results,
        "intervals": intervals,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "quick": quick,
    }
    (out_dir / METRICS_FILE.name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    c80 = intervals["80"]["coverage"]
    print(f"\nChosen: {best} (MAE {results[best]['mae_cr']:.3f} Cr; lowest MAE {selection['lowest_mae_model']}, "
          f"within-1-SE: {', '.join(selection['eligible'])}). 80% ranges cover {c80['overall']:.1%} of prices "
          f"(flats {c80.get('flat', 0):.1%}, houses {c80.get('house', 0):.1%}). Saved to {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="3 folds and 2 small models (CI smoke test)")
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR, help="where to save the model and metrics")
    args = parser.parse_args()
    train(quick=args.quick, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
