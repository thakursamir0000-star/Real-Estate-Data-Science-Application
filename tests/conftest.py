import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def dataset():
    from src.data.make_dataset import load_dataset

    return load_dataset()


@pytest.fixture(scope="session")
def predictor(dataset):
    from src.models.predict import PricePredictor

    return PricePredictor(reference=dataset)


@pytest.fixture
def flat():
    return {
        "property_type": "flat", "sector": "sector 54", "bedRoom": 3, "bathroom": 3, "balcony": "3+",
        "agePossession": "Relatively New", "built_up_area": 2000, "servant room": 1.0, "store room": 0.0,
        "furnishing_type": "Semi-furnished", "luxury_category": "High", "floor_category": "High Floor",
    }
