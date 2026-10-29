"""
Load and validate the modelling dataset.

The cleaning, feature engineering and feature selection steps live in
notebooks/ (see the README); their output is
data/processed/gurgaon_properties_post_feature_selection_v2.csv. This module
is the single entry point every script, test and app page uses to read it:
it checks the schema, drops exact duplicate listings and turns codes into
readable labels.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_FILE = ROOT / "data" / "processed" / "gurgaon_properties_post_feature_selection_v2.csv"
SOCIETIES_FILE = ROOT / "data" / "raw" / "appartments.csv"
COORDINATES_FILE = ROOT / "data" / "processed" / "latlong.csv"

TARGET = "price"  # in ₹ crore
NUMERIC = ["bedRoom", "bathroom", "built_up_area", "servant room", "store room"]
CATEGORICAL = ["property_type", "sector", "balcony", "agePossession",
               "furnishing_type", "luxury_category", "floor_category"]
# Column order the model and the app's input form use
FEATURES = ["property_type", "sector", "bedRoom", "bathroom", "balcony", "agePossession",
            "built_up_area", "servant room", "store room", "furnishing_type",
            "luxury_category", "floor_category"]

# furnishing_type came from k-means clusters of furnishing details (feature-engineering.ipynb)
FURNISHING_LABELS = {0: "Unfurnished", 1: "Semi-furnished", 2: "Furnished"}

ALLOWED_VALUES = {
    "property_type": {"flat", "house"},
    "balcony": {"0", "1", "2", "3", "3+"},
    "agePossession": {"New Property", "Relatively New", "Moderately Old", "Old Property", "Under Construction"},
    "furnishing_type": set(FURNISHING_LABELS.values()),
    "luxury_category": {"Low", "Medium", "High"},
    "floor_category": {"Low Floor", "Mid Floor", "High Floor"},
}


class DataValidationError(ValueError):
    """The dataset doesn't have the shape the model expects."""


def validate(df: pd.DataFrame) -> None:
    """Raise DataValidationError listing every problem found."""
    problems = []
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    nulls = df[[*FEATURES, TARGET]].isna().sum()
    if nulls.any():
        problems.append(f"null values: {nulls[nulls > 0].to_dict()}")
    if (df[TARGET] <= 0).any():
        problems.append(f"{int((df[TARGET] <= 0).sum())} non-positive prices")
    if (df["built_up_area"] <= 0).any():
        problems.append(f"{int((df['built_up_area'] <= 0).sum())} non-positive areas")
    for column, allowed in ALLOWED_VALUES.items():
        unexpected = set(df[column].astype(str)) - allowed
        if unexpected:
            problems.append(f"unexpected {column} values: {sorted(unexpected)[:5]}")
    if problems:
        raise DataValidationError("; ".join(problems))


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Readable labels and consistent types (no rows dropped)."""
    df = raw.copy()
    df["furnishing_type"] = df["furnishing_type"].map(lambda v: FURNISHING_LABELS.get(int(v), v))
    df["balcony"] = df["balcony"].astype(str)
    df["sector"] = df["sector"].astype(str).str.strip().str.lower()
    for column in NUMERIC:
        df[column] = df[column].astype(float)
    return df


def load_dataset(path: Path = PROCESSED_FILE, drop_duplicates: bool = True) -> pd.DataFrame:
    """The validated modelling table. Exact duplicate listings are dropped so a
    copy of a row can't sit on both sides of a train/test split."""
    df = prepare(pd.read_csv(path))
    validate(df)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df[[*FEATURES, TARGET]]


_COORDINATE = re.compile(r"([\d.]+)\D*?([NS])\W+([\d.]+)\D*?([EW])")


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """"28.3663° N, 76.9456° E" → (28.3663, 76.9456). The file's degree signs are
    mis-encoded, so only the numbers and N/S/E/W letters are relied on."""
    match = _COORDINATE.search(str(text))
    if not match:
        return None
    lat, ns, lon, ew = match.groups()
    return (float(lat) * (1 if ns == "N" else -1), float(lon) * (1 if ew == "E" else -1))


def load_sector_coordinates(path: Path = COORDINATES_FILE) -> pd.DataFrame:
    """sector, lat, lon for every sector whose coordinates parse."""
    raw = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    parsed = raw["coordinates"].map(parse_coordinates)
    out = raw.assign(lat=parsed.map(lambda p: p and p[0]), lon=parsed.map(lambda p: p and p[1]))
    out["sector"] = out["sector"].astype(str).str.strip().str.lower()
    return out.dropna(subset=["lat", "lon"])[["sector", "lat", "lon"]].reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(PROCESSED_FILE)
    data = load_dataset()
    print(f"{len(raw)} rows, {len(raw) - len(data)} exact duplicates dropped -> {len(data)} rows; schema OK")
