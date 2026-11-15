import numpy as np
import pandas as pd
import pytest

from src.recommend import (
    SOCIETY_WEIGHTS,
    SocietyRecommender,
    load_societies,
    parse_distance_km,
    parse_price_range,
    similar_listings,
)


@pytest.fixture(scope="module")
def societies():
    return SocietyRecommender()


def test_similar_listings_are_same_type_and_ranked(dataset):
    index = dataset[(dataset["sector"] == "sector 54") & (dataset["property_type"] == "flat")].index[0]
    matches = similar_listings(dataset, index, n=5)
    assert len(matches) == 5 and index not in matches.index
    assert (matches["property_type"] == "flat").all()
    assert list(matches["similarity"]) == sorted(matches["similarity"], reverse=True)


def test_area_does_not_swamp_the_other_features(dataset):
    # With standardized features, a match should share the bedroom count,
    # not just have a similar number of square feet.
    index = dataset[(dataset["property_type"] == "flat") & (dataset["bedRoom"] == 3)].index[0]
    matches = similar_listings(dataset, index, n=5)
    assert (matches["bedRoom"] == 3).mean() >= 0.6


def test_same_sector_filter(dataset):
    index = dataset[(dataset["sector"] == "sector 54") & (dataset["property_type"] == "flat")].index[0]
    matches = similar_listings(dataset, index, n=3, same_sector=True)
    assert (matches["sector"] == "sector 54").all()


@pytest.mark.parametrize("text, expected", [
    ("₹ 1.55 - 3.21 Cr", [1.55, 3.21]),
    ("₹ 85 L - 1.2 Cr", [0.85, 1.2]),
    ("₹ 95 Lac", [0.95]),
    ("Price on Request", []),
])
def test_parse_price_range(text, expected):
    assert parse_price_range(text) == pytest.approx(expected)


@pytest.mark.parametrize("text, km", [("2.9 KM", 2.9), ("10.7  KM", 10.7), ("500 m", 0.5), ("nearby", None)])
def test_parse_distance(text, km):
    assert parse_distance_km(text) == km


def test_society_recommendations(societies):
    result = societies.recommend("Sobha City", n=5)
    assert len(result) == 5 and "Sobha City" not in set(result["name"])
    assert result["similarity"].between(0, 1).all()
    assert list(result["similarity"]) == sorted(result["similarity"], reverse=True)
    assert result["shared_facilities"].str.len().gt(0).any()


def test_unknown_society(societies):
    with pytest.raises(KeyError):
        societies.recommend("Not A Real Society")


def test_society_loader_removes_repeated_headers_and_preserves_names(tmp_path):
    columns = ["PropertyName", "PropertySubName", "NearbyLocations", "LocationAdvantages",
               "Link", "PriceDetails", "TopFacilities"]
    legitimate = {
        "PropertyName": "PropertyName Heights",
        "PropertySubName": "3 BHK Apartment in Sector 42, Gurgaon",
        "LocationAdvantages": "{'Metro': '500 m'}",
        "Link": "https://example.test/property-name-heights",
        "PriceDetails": "{'3 BHK': {'price-range': '₹ 2 - 3 Cr'}}",
        "TopFacilities": "['Pool', 'Gym']",
    }
    missing_price = {
        "PropertyName": "BPTP Pedestal",
        "PropertySubName": "Apartment in Sector 70A, Gurgaon",
        "LocationAdvantages": "not a dictionary",
        "PriceDetails": "[]",
        "TopFacilities": "malformed",
    }
    repeated_header = dict(zip(columns, columns))
    path = tmp_path / "societies.csv"
    pd.DataFrame([legitimate, repeated_header, missing_price], columns=columns).to_csv(path, index=False)

    loaded = load_societies(path)

    assert list(loaded["name"]) == ["PropertyName Heights", "BPTP Pedestal"]
    assert "PropertyName" not in set(loaded["name"])
    assert loaded.loc[0, "sector"] == "sector 42"
    assert loaded.loc[0, "facilities"] == ["Pool", "Gym"]
    assert pd.isna(loaded.loc[1, "price_low"])
    assert loaded.loc[1, "places"] == {} and loaded.loc[1, "facilities"] == []


def test_society_loader_handles_empty_wrong_schema_and_bad_csv_rows(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert load_societies(empty).empty

    wrong_schema = tmp_path / "wrong.csv"
    wrong_schema.write_text("wrong,columns\n1,2\n", encoding="utf-8")
    assert load_societies(wrong_schema).empty

    malformed = tmp_path / "malformed.csv"
    malformed.write_text(
        "PropertyName,PropertySubName,NearbyLocations,LocationAdvantages,Link,PriceDetails,TopFacilities\n"
        "bad,row,with,too,many,columns,to,load,safely\n"
        "Legitimate Society,Apartment,,,,,\n",
        encoding="utf-8",
    )
    assert list(load_societies(malformed)["name"]) == ["Legitimate Society"]


def _scoring_societies():
    return pd.DataFrame([
        {"name": "Valid Query", "sector": "sector 1", "facilities": ["Pool", "Gym"],
         "places": {"Metro": 1.0}, "price_low": 2.0, "price_high": 3.0, "link": ""},
        {"name": "Valid Match", "sector": "sector 2", "facilities": ["Pool"],
         "places": {"Metro": 2.0}, "price_low": 2.5, "price_high": 3.5, "link": ""},
        {"name": "Anant Raj Estates", "sector": "sector 3", "facilities": ["Pool", "Gym"],
         "places": {"Metro": 1.5}, "price_low": np.nan, "price_high": np.nan, "link": ""},
        {"name": "BPTP Pedestal", "sector": "sector 4", "facilities": ["Pool"],
         "places": {"Metro": 2.5}, "price_low": np.nan, "price_high": np.nan, "link": ""},
    ])


def test_missing_prices_are_unavailable_but_other_features_still_rank():
    recommender = SocietyRecommender(_scoring_societies())
    result = recommender.recommend("Anant Raj Estates", n=3)
    pedestal = result.loc[result["name"] == "BPTP Pedestal"].iloc[0]

    assert pd.isna(pedestal["price_match"])
    assert pedestal["similarity"] > 0
    assert pedestal["facilities_match"] > 0 or pedestal["location_match"] > 0


def test_valid_price_similarity_formula_is_unchanged():
    valid = _scoring_societies().dropna(subset=["price_low", "price_high"]).reset_index(drop=True)
    recommender = SocietyRecommender(valid)
    midpoint = (valid["price_low"] + valid["price_high"]) / 2
    level = np.log(midpoint.to_numpy())
    expected_price = np.exp(-np.abs(level[:, None] - level[None, :]))
    w = SOCIETY_WEIGHTS
    expected_total = (w["facilities"] * recommender.facility_sim
                      + w["price"] * expected_price
                      + w["location"] * recommender.location_sim)

    np.testing.assert_allclose(recommender.price_sim, expected_price)
    np.testing.assert_allclose(recommender.similarity, expected_total)


def test_same_sector_filter_is_strict_even_when_the_sector_is_small(dataset):
    # A review found "only this sector" silently returning other sectors when
    # fewer than n listings matched (about a third of type/sector groups).
    counts = dataset.groupby(["property_type", "sector"]).size()
    small_type, small_sector = counts[counts.between(2, 4)].index[0]
    index = dataset[(dataset["property_type"] == small_type) & (dataset["sector"] == small_sector)].index[0]
    matches = similar_listings(dataset, index, n=5, same_sector=True)
    assert 1 <= len(matches) < 5
    assert (matches["sector"] == small_sector).all()
