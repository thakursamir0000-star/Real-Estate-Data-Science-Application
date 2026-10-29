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


# ── listings ────────────────────────────────────────────────────────

def similar_listings(df: pd.DataFrame, index: int, n: int = 5, same_sector: bool = False) -> pd.DataFrame:
    """The n listings most similar to df.loc[index], with a 0-1 similarity score."""
    query = df.loc[index]
    pool = df[(df["property_type"] == query["property_type"]) & (df.index != index)]
    if same_sector:
        in_sector = pool[pool["sector"] == query["sector"]]
        pool = in_sector if len(in_sector) >= n else pool
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


def load_societies(path: Path = SOCIETIES_FILE) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows = []
    for r in raw.itertuples():
        prices = [p for detail in _literal(r.PriceDetails, {}).values()
                  for p in parse_price_range(detail.get("price-range", ""))]
        places = {place: d for place, text in _literal(r.LocationAdvantages, {}).items()
                  if (d := parse_distance_km(text)) is not None}
        sector = re.search(r" in (.+?), Gurgaon", str(r.PropertySubName))
        rows.append({
            "name": r.PropertyName,
            "sector": sector.group(1).lower() if sector else "",
            "facilities": list(_literal(r.TopFacilities, [])),
            "places": places,
            "price_low": min(prices) if prices else np.nan,
            "price_high": max(prices) if prices else np.nan,
            "link": r.Link,
        })
    return pd.DataFrame(rows).drop_duplicates("name").reset_index(drop=True)


class SocietyRecommender:
    def __init__(self, societies: pd.DataFrame | None = None):
        self.societies = societies if societies is not None else load_societies()
        s = self.societies

        facilities = MultiLabelBinarizer().fit_transform(s["facilities"])
        self.facility_sim = cosine_similarity(facilities) if facilities.size else np.zeros((len(s), len(s)))

        places = sorted({p for d in s["places"] for p in d})
        proximity = np.array([[1 / (1 + d.get(p, np.inf)) for p in places] for d in s["places"]])
        self.location_sim = cosine_similarity(proximity) if places else np.zeros((len(s), len(s)))

        level = np.log(((s["price_low"] + s["price_high"]) / 2).fillna(s["price_low"].median()))
        self.price_sim = np.exp(-np.abs(level.to_numpy()[:, None] - level.to_numpy()[None, :]))

        w = SOCIETY_WEIGHTS
        self.similarity = w["facilities"] * self.facility_sim + w["price"] * self.price_sim + w["location"] * self.location_sim

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
                "price_match": round(float(self.price_sim[i, j]), 2),
                "location_match": round(float(self.location_sim[i, j]), 2),
                "link": other["link"],
            })
        return pd.DataFrame(rows)
