import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.make_dataset import load_dataset  # noqa: E402
from src.recommend import SocietyRecommender, similar_listings  # noqa: E402

st.set_page_config(page_title="Recommend Properties", page_icon="🏡", layout="wide")
st.title("🏡 Property Recommendations")


@st.cache_data
def listings():
    return load_dataset()


@st.cache_resource
def societies() -> SocietyRecommender:
    return SocietyRecommender()


tab_listings, tab_societies = st.tabs(["Similar listings", "Similar housing societies"])

with tab_listings:
    df = listings()
    st.write("Pick a listing you like; we find the closest matches of the same type by price, size, "
             "bedrooms and bathrooms.")
    c1, c2, c3 = st.columns(3)
    property_type = c1.selectbox("Property type", ["flat", "house"])
    sectors = sorted(df.loc[df["property_type"] == property_type, "sector"].unique())
    sector = c2.selectbox("Sector", sectors, index=sectors.index("sector 54") if "sector 54" in sectors else 0)
    same_sector = c3.toggle("Only this sector", value=False)

    options = df[(df["property_type"] == property_type) & (df["sector"] == sector)].sort_values("price")
    labels = {i: f"{r.bedRoom:.0f} BHK · {r.built_up_area:,.0f} sq ft · ₹{r.price:.2f} Cr · {r.furnishing_type}"
              for i, r in options.iterrows()}
    choice = st.selectbox("Listing", list(labels), format_func=labels.get)
    if choice is not None:
        matches = similar_listings(df, choice, n=8, same_sector=same_sector)
        st.dataframe(
            matches[["similarity", "sector", "bedRoom", "bathroom", "built_up_area", "price",
                     "furnishing_type", "agePossession"]].rename(columns={
                "bedRoom": "Bedrooms", "bathroom": "Bathrooms", "built_up_area": "Area (sq ft)",
                "price": "Price (₹ Cr)", "furnishing_type": "Furnishing", "agePossession": "Age"}),
            hide_index=True, width="stretch")

with tab_societies:
    rec = societies()
    st.write("Societies similar to one you like, by shared facilities, price level and nearby places "
             f"({len(rec.societies)} Gurgaon societies).")
    names = rec.names()
    name = st.selectbox("Society", names, index=names.index("Sobha City") if "Sobha City" in names else 0)
    result = rec.recommend(name, n=6)
    st.dataframe(
        result[["name", "sector", "similarity", "price_range_cr", "shared_facilities", "shared_nearby"]].rename(
            columns={"name": "Society", "sector": "Sector", "similarity": "Similarity",
                     "price_range_cr": "Price range (₹ Cr)", "shared_facilities": "Shared facilities",
                     "shared_nearby": "Shared nearby places"}),
        hide_index=True, width="stretch")
    with st.expander("Why these? (match per signal)"):
        st.dataframe(result[["name", "facilities_match", "price_match", "location_match"]],
                     hide_index=True, width="stretch")
        st.caption("Similarity = 52% facilities + 34% price + 14% location, the weights from "
                   "notebooks/recommender-system.ipynb.")
