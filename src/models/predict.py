"""
Price predictions with a calibrated range and per-prediction explanations.

    from src.models.predict import PricePredictor
    predictor = PricePredictor()
    estimate = predictor.predict({...12 features...})
    drivers = predictor.explain({...})
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.make_dataset import FEATURES, NUMERIC, load_dataset

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "models" / "price_model.joblib"
METRICS_FILE = ROOT / "models" / "metrics.json"


@dataclass(frozen=True)
class PriceEstimate:
    price_cr: float
    low_cr: float
    high_cr: float
    level: int  # e.g. 80 → the true price falls in [low, high] ~80% of the time


@dataclass(frozen=True)
class Driver:
    feature: str
    value: object
    typical: object
    effect_pct: float  # how much this value moves the estimate vs the typical value


class PricePredictor:
    def __init__(self, model_path: Path = MODEL_FILE, metrics_path: Path = METRICS_FILE,
                 reference: pd.DataFrame | None = None):
        self.model = joblib.load(model_path)
        self.metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        self._reference = reference

    @property
    def reference(self) -> pd.DataFrame:
        if self._reference is None:
            self._reference = load_dataset()
        return self._reference

    def _frame(self, inputs) -> pd.DataFrame:
        frame = pd.DataFrame([inputs]) if isinstance(inputs, dict) else pd.DataFrame(inputs)
        missing = [f for f in FEATURES if f not in frame.columns]
        if missing:
            raise ValueError(f"Missing inputs: {missing}")
        frame = frame[FEATURES].copy()
        for column in NUMERIC:
            frame[column] = frame[column].astype(float)
        frame["balcony"] = frame["balcony"].astype(str)
        return frame

    def predict_log(self, inputs) -> np.ndarray:
        return np.asarray(self.model.predict(self._frame(inputs)), dtype=float)

    def predict(self, inputs: dict, level: int = 80) -> PriceEstimate:
        """Point estimate plus the conformal price range for this property type."""
        pred_log = float(self.predict_log(inputs)[0])
        widths = self.metrics["intervals"][str(level)]["log_half_width"]
        half = widths.get(str(inputs["property_type"]), max(widths.values()))
        return PriceEstimate(
            price_cr=round(float(np.expm1(pred_log)), 3),
            low_cr=round(float(max(np.expm1(pred_log - half), 0.0)), 3),
            high_cr=round(float(np.expm1(pred_log + half)), 3),
            level=level,
        )

    def typical_values(self, property_type: str) -> dict:
        """Median / most common value of each feature among this property type."""
        same = self.reference[self.reference["property_type"] == property_type]
        same = same if len(same) else self.reference
        return {f: (float(same[f].median()) if f in NUMERIC else same[f].mode().iloc[0]) for f in FEATURES}

    def explain(self, inputs: dict, top: int = 5) -> list[Driver]:
        """What-if explanation: swap each feature for its typical value (among the
        same property type) and see how much the estimate changes.

        Approximate by design — features interact (area and bedrooms move
        together) — but it answers "why this price?" for any model.
        """
        base = self._frame(inputs)
        typical = self.typical_values(str(inputs["property_type"]))
        variants = []
        for feature in FEATURES:
            if feature == "property_type":
                continue
            row = base.copy()
            row[feature] = typical[feature]
            variants.append((feature, row))
        batch = pd.concat([base] + [row for _, row in variants], ignore_index=True)
        logs = self.predict_log(batch)
        drivers = [
            Driver(feature, base.iloc[0][feature], typical[feature],
                   round(float(np.expm1(logs[0] - logs[i + 1])) * 100, 1))
            for i, (feature, _) in enumerate(variants)
        ]
        drivers = [d for d in drivers if abs(d.effect_pct) >= 0.5]
        return sorted(drivers, key=lambda d: -abs(d.effect_pct))[:top]
