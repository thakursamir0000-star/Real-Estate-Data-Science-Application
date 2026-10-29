import pytest

from src.recommend import SocietyRecommender, parse_distance_km, parse_price_range, similar_listings


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
