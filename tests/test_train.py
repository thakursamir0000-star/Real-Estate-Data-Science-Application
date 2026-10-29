import json

import numpy as np
import sklearn

from src.models.train_models import (
    METRICS_FILE,
    conformal_quantile,
    interval_coverage,
    select_model,
    train,
)


def test_conformal_quantile_is_the_finite_sample_rank():
    residuals = np.arange(1, 10, dtype=float)  # 9 values
    assert conformal_quantile(residuals, 0.8) == 8.0  # rank ceil(10 * 0.8) = 8
    assert conformal_quantile(residuals, 0.99) == 9.0  # capped at the largest


def test_interval_coverage_matches_the_level_on_exchangeable_residuals():
    rng = np.random.default_rng(0)
    residuals = rng.normal(size=4000)
    groups = np.where(rng.random(4000) < 0.3, "house", "flat")
    folds = rng.integers(0, 10, 4000)
    coverage = interval_coverage(residuals, groups, folds, 0.8)
    assert abs(coverage["overall"] - 0.8) < 0.02
    assert abs(coverage["house"] - 0.8) < 0.03


def test_one_standard_error_rule_prefers_the_cheaper_model():
    results = {
        "random_forest": {"mae_cr": 0.522, "mae_cr_se": 0.016},
        "xgboost": {"mae_cr": 0.529, "mae_cr_se": 0.017},
        "lightgbm": {"mae_cr": 0.547, "mae_cr_se": 0.018},
    }
    chosen, selection = select_model(results)
    assert chosen == "xgboost" and selection["lowest_mae_model"] == "random_forest"
    assert "lightgbm" not in selection["eligible"]


def test_quick_training_produces_a_model_and_metrics(tmp_path):
    report = train(quick=True, out_dir=tmp_path)
    assert (tmp_path / "price_model.joblib").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["best_model"] == report["best_model"] and saved["quick"] is True
    assert {"80", "90"} <= set(saved["intervals"])
    assert 0.7 < saved["intervals"]["80"]["coverage"]["overall"] < 0.9


def test_committed_model_card_matches_the_pinned_environment():
    card = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    assert card["quick"] is False and card["cv_folds"] == 10
    assert card["versions"]["scikit-learn"] == sklearn.__version__  # pickled with the pinned version
    best = card["models"][card["best_model"]]
    assert best["mae_cr"] <= card["models"]["ridge"]["mae_cr"]
    assert abs(card["intervals"]["80"]["coverage"]["overall"] - 0.8) < 0.02
