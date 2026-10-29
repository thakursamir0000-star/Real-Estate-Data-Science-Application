import pandas as pd
import pytest

from src.data.make_dataset import (
    FEATURES,
    FURNISHING_LABELS,
    PROCESSED_FILE,
    TARGET,
    DataValidationError,
    load_sector_coordinates,
    parse_coordinates,
    prepare,
    validate,
)


def test_dataset_is_clean_and_deduplicated(dataset):
    raw = pd.read_csv(PROCESSED_FILE)
    assert list(dataset.columns) == [*FEATURES, TARGET]
    assert len(dataset) == len(raw.drop_duplicates()) and not dataset.duplicated().any()
    assert not dataset.isna().any().any()
    assert set(dataset["furnishing_type"]) <= set(FURNISHING_LABELS.values())
    assert (dataset[TARGET] > 0).all()


def test_furnishing_codes_become_labels():
    raw = pd.read_csv(PROCESSED_FILE).head(3).assign(furnishing_type=[0.0, 1.0, 2.0])
    assert list(prepare(raw)["furnishing_type"]) == ["Unfurnished", "Semi-furnished", "Furnished"]


@pytest.mark.parametrize("break_it, message", [
    (lambda d: d.drop(columns=["sector"]), "Missing columns"),
    (lambda d: d.assign(price=-1.0), "non-positive prices"),
    (lambda d: d.assign(property_type="castle"), "unexpected property_type"),
    (lambda d: d.assign(built_up_area=None), "null values"),
])
def test_validation_catches_bad_data(dataset, break_it, message):
    with pytest.raises(DataValidationError, match=message):
        validate(break_it(dataset.head(20).copy()))


@pytest.mark.parametrize("text, expected", [
    ("28.3663° N, 76.9456° E", (28.3663, 76.9456)),
    ("28.3663� N, 76.9456� E", (28.3663, 76.9456)),  # the file's broken degree signs
    ("12.5 S, 45.0 W", (-12.5, -45.0)),
    ("not a coordinate", None),
])
def test_parse_coordinates(text, expected):
    assert parse_coordinates(text) == expected


def test_sector_coordinates_load_and_stay_in_gurgaon(dataset):
    coords = load_sector_coordinates()
    assert len(coords) >= 120
    assert coords["lat"].between(28.2, 28.7).all() and coords["lon"].between(76.7, 77.3).all()
    assert dataset["sector"].isin(coords["sector"]).mean() > 0.9
