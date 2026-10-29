import pytest

from src.data.make_dataset import FEATURES


def test_estimate_sits_inside_its_range(predictor, flat):
    estimate = predictor.predict(flat)
    assert 0 < estimate.low_cr < estimate.price_cr < estimate.high_cr
    wider = predictor.predict(flat, level=90)
    assert wider.low_cr <= estimate.low_cr and wider.high_cr >= estimate.high_cr


def test_house_ranges_are_relatively_wider(predictor, flat):
    house = {**flat, "property_type": "house"}
    f, h = predictor.predict(flat), predictor.predict(house)
    assert h.high_cr / h.low_cr > f.high_cr / f.low_cr


def test_prime_sector_costs_more(predictor, flat):
    prime = predictor.predict(flat).price_cr
    outskirts = predictor.predict({**flat, "sector": "sohna road"}).price_cr
    assert prime > outskirts * 1.3


def test_explanations_are_ranked_and_named(predictor, flat):
    drivers = predictor.explain(flat)
    assert drivers and all(d.feature in FEATURES for d in drivers)
    effects = [abs(d.effect_pct) for d in drivers]
    assert effects == sorted(effects, reverse=True)
    assert drivers[0].feature == "sector"  # sector 54 is far from the typical sector


def test_effects_are_on_the_price_itself(predictor, flat):
    # The sector effect must equal the ratio of the two predicted prices.
    sector = next(d for d in predictor.explain(flat) if d.feature == "sector")
    typical_sector = predictor.predict({**flat, "sector": sector.typical}).price_cr
    expected = (predictor.predict(flat).price_cr / typical_sector - 1) * 100
    assert sector.effect_pct == pytest.approx(expected, abs=0.5)


def test_a_typical_property_has_no_big_drivers(predictor):
    typical = predictor.typical_values("flat")
    assert all(abs(d.effect_pct) < 0.5 for d in predictor.explain({**typical, "property_type": "flat"}))


def test_missing_inputs_are_rejected(predictor, flat):
    with pytest.raises(ValueError, match="Missing inputs"):
        predictor.predict({k: v for k, v in flat.items() if k != "sector"})
