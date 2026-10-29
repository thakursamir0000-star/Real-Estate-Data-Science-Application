import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.make_dataset import ALLOWED_VALUES, load_dataset  # noqa: E402
from src.models.predict import PricePredictor  # noqa: E402

st.set_page_config(page_title="Price Predictor", page_icon="💰")

NOISE_PCT = 5
LABELS = {
    "property_type": "Property type", "sector": "Sector", "bedRoom": "Bedrooms", "bathroom": "Bathrooms",
    "balcony": "Balconies", "agePossession": "Age / possession", "built_up_area": "Built-up area",
    "servant room": "Servant room", "store room": "Store room", "furnishing_type": "Furnishing",
    "luxury_category": "Luxury level", "floor_category": "Floor",
}


def readable(feature: str, value) -> str:
    if feature in ("servant room", "store room"):
        return "Yes" if float(value) else "No"
    if isinstance(value, (int, float)):
        return f"{value:,.0f}" + (" sq ft" if feature == "built_up_area" else "")
    return str(value)


@st.cache_resource
def get_predictor() -> PricePredictor:
    return PricePredictor(reference=load_dataset())


try:
    predictor = get_predictor()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.models.train_models` first.")
    st.stop()

data = predictor.reference
st.title("💰 Property Price Predictor")
st.write("Describe the property to get an estimate in ₹ crore, a likely price range, and what drives the price.")

col1, col2 = st.columns(2)
with col1:
    property_type = st.selectbox("Property type", ["flat", "house"])
    sector = st.selectbox("Sector", sorted(data["sector"].unique()))
    bedroom = st.number_input("Bedrooms", min_value=1, max_value=12, value=3)
    bathroom = st.number_input("Bathrooms", min_value=1, max_value=12, value=3)
    balcony = st.selectbox("Balconies", ["0", "1", "2", "3", "3+"], index=3)
    age = st.selectbox("Age / possession", ["New Property", "Relatively New", "Moderately Old", "Old Property",
                                            "Under Construction"], index=1)
with col2:
    area = st.number_input("Built-up area (sq ft)", min_value=200, max_value=20000, value=1650, step=50)
    servant = st.checkbox("Servant room")
    store = st.checkbox("Store room")
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])
    luxury = st.selectbox("Luxury level", sorted(ALLOWED_VALUES["luxury_category"], key=["Low", "Medium", "High"].index))
    floor = st.selectbox("Floor", ["Low Floor", "Mid Floor", "High Floor"], index=1)

inputs = {
    "property_type": property_type, "sector": sector, "bedRoom": bedroom, "bathroom": bathroom,
    "balcony": balcony, "agePossession": age, "built_up_area": area, "servant room": float(servant),
    "store room": float(store), "furnishing_type": furnishing, "luxury_category": luxury, "floor_category": floor,
}

if st.button("Predict price", width="stretch", type="primary"):
    estimate = predictor.predict(inputs)
    st.success(f"### Estimated price: ₹ {estimate.price_cr:.2f} Cr")
    st.write(f"**Likely range ({estimate.level}%):** ₹ {estimate.low_cr:.2f} – {estimate.high_cr:.2f} Cr  ·  "
             f"₹ {estimate.price_cr * 1e7 / area:,.0f} per sq ft")
    st.caption(f"In cross-validation, {estimate.level}% ranges like this contained the real listed price "
               f"{predictor.metrics['intervals'][str(estimate.level)]['coverage'][property_type]:.0%} of the time "
               f"for {property_type}s.")

    explained = predictor.explain(inputs, top=8)
    # Effects this small are within the model's noise (e.g. a few % either way for ±70 sq ft),
    # so they aren't presented as drivers.
    drivers = [d for d in explained if abs(d.effect_pct) >= NOISE_PCT]
    minor = [LABELS.get(d.feature, d.feature) for d in explained if abs(d.effect_pct) < NOISE_PCT]
    if drivers:
        st.subheader("What drives this price")
        table = pd.DataFrame([{
            "Feature": LABELS.get(d.feature, d.feature),
            "This property": readable(d.feature, d.value),
            "Typical": readable(d.feature, d.typical),
            "Effect on price (%)": d.effect_pct,
        } for d in drivers])
        fig = px.bar(table, x="Effect on price (%)", y="Feature", orientation="h", text="Effect on price (%)",
                     color=table["Effect on price (%)"] > 0, color_discrete_map={True: "#2e7d32", False: "#c62828"})
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"}, height=320)
        st.plotly_chart(fig, width="stretch")
        st.caption("Each bar: how much higher (or lower) the estimate is than if only that feature had the most "
                   f"common value for a {property_type} (the Typical column). Approximate, because features interact.")
        st.dataframe(table, hide_index=True, width="stretch")
    if minor:
        st.caption(f"Little clear effect (under ±{NOISE_PCT}%): {', '.join(minor)}.")
    if not drivers:
        st.info("This property is close to a typical one: no feature moves the estimate by more than "
                f"±{NOISE_PCT}%.")

with st.expander("How accurate is the model?"):
    m = predictor.metrics
    best = m["models"][m["best_model"]]
    st.write(
        f"**{m['best_model']}**, chosen by {m['cv_folds']}-fold cross-validation on "
        f"{m['data']['rows_after_dedup']:,} listings: mean absolute error ₹{best['mae_cr']:.2f} Cr, "
        f"median error {best['median_ape']:.0%}, R² {best['r2_log']:.3f} (log price)."
    )
    st.write("Prices are listing prices from 99acres, not transaction prices.")
