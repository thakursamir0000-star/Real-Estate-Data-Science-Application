"""
Two recommenders.

similar_listings — "show me listings like this one": among the same property
type, nearest neighbours on standardized log price, log area, bedrooms and
bathrooms (standardizing matters: raw square feet would otherwise swamp
everything else), optionally within the same sector.

SocietyRecommender — "societies like this one" (data/raw/appartments.csv):
the notebook approach (recommender-system.ipynb) of blending three
similarities — shared facilities, price level and nearby places — with the
reasons returned alongside each recommendation.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

from src.data.make_dataset import SOCIETIES_FILE

SIMILARITY_FEATURES = ["log_price", "log_area", "bedRoom", "bathroom"]
# Weights from recommender-system.ipynb (30 / 20 / 8), normalized
SOCIETY_WEIGHTS = {"facilities": 30 / 58, "price": 20 / 58, "location": 8 / 58}
SOCIETY_COLUMNS = ["name", "sector", "facilities", "places", "price_low", "price_high", "link"]


# ── listings ────────────────────────────────────────────────────────

def similar_listings(df: pd.DataFrame, index: int, n: int = 5, same_sector: bool = False) -> pd.DataFrame:
    """The n listings most similar to df.loc[index], with a 0-1 similarity score.

    With same_sector, only listings in the same sector are considered, so fewer
    than n (or none) can come back. It used to fall back to every sector when
    fewer than n matched, silently ignoring the filter.
    """
    query = df.loc[index]
    pool = df[(df["property_type"] == query["property_type"]) & (df.index != index)]
    if same_sector:
        pool = pool[pool["sector"] == query["sector"]]
    features = pd.DataFrame({
        "log_price": np.log(df["price"]), "log_area": np.log(df["built_up_area"]),
        "bedRoom": df["bedRoom"], "bathroom": df["bathroom"],
    })
    scale = features.loc[df["property_type"] == query["property_type"]].std().replace(0, 1)
    z = (features - features.loc[index]) / scale
    distance = np.sqrt((z.loc[pool.index] ** 2).sum(axis=1))
    out = pool.assign(similarity=(1 / (1 + distance)).round(3)).sort_values("similarity", ascending=False)
    return out.head(n)


# ── societies ───────────────────────────────────────────────────────

_PRICE = re.compile(r"([\d.]+)\s*(Cr|L(?:ac|akh)?)?", re.IGNORECASE)
_DISTANCE = re.compile(r"([\d.]+)\s*(km|m)\b", re.IGNORECASE)


def parse_price_range(text: str) -> list[float]:
    """"₹ 1.55 - 3.21 Cr" → [1.55, 3.21]; "₹ 85 L - 1.2 Cr" → [0.85, 1.2] (₹ crore)."""
    tokens = [(float(v), (u or "").lower()) for v, u in _PRICE.findall(str(text).replace(",", "")) if v != "."]
    values = []
    for i, (value, unit) in enumerate(tokens):
        unit = unit or next((u for _, u in tokens[i + 1:] if u), "cr")  # "1.55 - 3.21 Cr": unit given once
        values.append(value / 100 if unit.startswith("l") else value)
    return values


def parse_distance_km(text: str) -> float | None:
    match = _DISTANCE.search(str(text))
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000 if match.group(2).lower() == "m" else value


def _literal(text, default):
    try:
        return ast.literal_eval(str(text))
    except (ValueError, SyntaxError):
        return default


def _is_repeated_header(row: pd.Series) -> bool:
    """True for embedded CSV header rows, including partially populated copies."""
    matches = sum(
        pd.notna(value) and str(value).strip().casefold() == str(column).strip().casefold()
        for column, value in row.items()
    )
    property_name = row.get("PropertyName")
    return (pd.notna(property_name)
            and str(property_name).strip().casefold() == "propertyname"
            and matches >= 2)


def _empty_societies() -> pd.DataFrame:
    return pd.DataFrame(columns=SOCIETY_COLUMNS)


def load_societies(path: Path = SOCIETIES_FILE) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8-sig", errors="replace", newline="") as source:
        reader = csv.reader(source)
        try:
            columns = [str(column).strip() for column in next(reader)]
        except (StopIteration, csv.Error):
            return _empty_societies()
        rows = []
        try:
            for row in reader:
                if len(row) <= len(columns):
                    rows.append([*row, *([None] * (len(columns) - len(row)))])
        except csv.Error:
            pass
    raw = pd.DataFrame(rows, columns=columns)
    if "PropertyName" not in raw.columns:
        return _empty_societies()
    raw = raw.loc[~raw.apply(_is_repeated_header, axis=1)]

    rows = []
    for _, r in raw.iterrows():
        if pd.isna(r["PropertyName"]) or not (name := str(r["PropertyName"]).strip()):
            continue
        price_details = _literal(r.get("PriceDetails"), {})
        if not isinstance(price_details, dict):
            price_details = {}
        prices = [p for detail in price_details.values() if isinstance(detail, dict)
                  for p in parse_price_range(detail.get("price-range", ""))]
        location_advantages = _literal(r.get("LocationAdvantages"), {})
        if not isinstance(location_advantages, dict):
            location_advantages = {}
        places = {place: d for place, text in location_advantages.items()
                  if (d := parse_distance_km(text)) is not None}
        facilities = _literal(r.get("TopFacilities"), [])
        if not isinstance(facilities, (list, tuple, set)):
            facilities = []
        sector = re.search(r" in (.+?), Gurgaon", str(r.get("PropertySubName", "")))
        rows.append({
            "name": name,
            "sector": sector.group(1).lower() if sector else "",
            "facilities": list(facilities),
            "places": places,
            "price_low": min(prices) if prices else np.nan,
            "price_high": max(prices) if prices else np.nan,
            "link": "" if pd.isna(r.get("Link")) else str(r.get("Link", "")),
        })
    if not rows:
        return _empty_societies()
    return pd.DataFrame(rows, columns=SOCIETY_COLUMNS).drop_duplicates("name").reset_index(drop=True)


class SocietyRecommender:
    def __init__(self, societies: pd.DataFrame | None = None):
        self.societies = societies if societies is not None else load_societies()
        s = self.societies

        facilities = MultiLabelBinarizer().fit_transform(s["facilities"])
        self.facility_sim = cosine_similarity(facilities) if facilities.size else np.zeros((len(s), len(s)))

        places = sorted({p for d in s["places"] for p in d})
        proximity = np.array([[1 / (1 + d.get(p, np.inf)) for p in places] for d in s["places"]])
        self.location_sim = cosine_similarity(proximity) if places else np.zeros((len(s), len(s)))

        midpoint = (pd.to_numeric(s["price_low"], errors="coerce")
                    + pd.to_numeric(s["price_high"], errors="coerce")) / 2
        price_available = midpoint.notna() & midpoint.gt(0)
        level = np.full(len(s), np.nan)
        level[price_available] = np.log(midpoint[price_available])
        price_pairs = price_available.to_numpy()[:, None] & price_available.to_numpy()[None, :]
        self.price_sim = np.full((len(s), len(s)), np.nan)
        differences = np.abs(level[:, None] - level[None, :])
        self.price_sim[price_pairs] = np.exp(-differences[price_pairs])

        w = SOCIETY_WEIGHTS
        weighted = w["facilities"] * self.facility_sim + w["location"] * self.location_sim
        weighted += np.where(price_pairs, w["price"] * self.price_sim, 0)
        available_weight = w["facilities"] + w["location"] + np.where(price_pairs, w["price"], 0)
        self.similarity = weighted / available_weight

    def names(self) -> list[str]:
        return sorted(self.societies["name"])

    def recommend(self, name: str, n: int = 5) -> pd.DataFrame:
        """The n most similar societies with the reasons for each match."""
        s = self.societies
        matches = s.index[s["name"] == name]
        if not len(matches):
            raise KeyError(f"Unknown society: {name}")
        i = matches[0]
        order = [j for j in np.argsort(-self.similarity[i]) if j != i][:n]
        query = s.loc[i]
        rows = []
        for j in order:
            other = s.loc[j]
            shared_places = sorted(set(query["places"]) & set(other["places"]),
                                   key=lambda p: other["places"][p])
            rows.append({
                "name": other["name"],
                "sector": other["sector"],
                "similarity": round(float(self.similarity[i, j]), 3),
                "price_range_cr": (f"{other['price_low']:.2f}–{other['price_high']:.2f}"
                                   if pd.notna(other["price_low"]) else "on request"),
                "shared_facilities": ", ".join(sorted(set(query["facilities"]) & set(other["facilities"]))[:5]),
                "shared_nearby": ", ".join(shared_places[:3]),
                "facilities_match": round(float(self.facility_sim[i, j]), 2),
                "price_match": (round(float(self.price_sim[i, j]), 2)
                                if np.isfinite(self.price_sim[i, j]) else None),
                "location_match": round(float(self.location_sim[i, j]), 2),
                "link": other["link"],
            })
        return pd.DataFrame(rows)
