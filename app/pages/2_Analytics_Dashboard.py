import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.make_dataset import load_dataset, load_sector_coordinates  # noqa: E402

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Gurgaon Real Estate Analytics")


@st.cache_data
def load():
    df = load_dataset()
    df["price_per_sqft"] = df["price"] * 1e7 / df["built_up_area"]
    return df, load_sector_coordinates()


df, coords = load()
property_types = st.multiselect("Property types", ["flat", "house"], default=["flat", "house"])
view = df[df["property_type"].isin(property_types)] if property_types else df

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(view):,}")
c2.metric("Median price", f"₹{view['price'].median():.2f} Cr")
c3.metric("Median price / sq ft", f"₹{view['price_per_sqft'].median():,.0f}")
c4.metric("Sectors", view["sector"].nunique())

st.subheader("Where prices are high: median price per sq ft by sector")
by_sector = (view.groupby("sector")
             .agg(median_price_per_sqft=("price_per_sqft", "median"), listings=("price", "size"),
                  median_price_cr=("price", "median"))
             .reset_index().merge(coords, on="sector", how="inner"))
by_sector = by_sector[by_sector["listings"] >= 3]  # too few listings make a sector's median unreliable
fig_map = px.scatter_map(
    by_sector, lat="lat", lon="lon", size="listings", color="median_price_per_sqft",
    color_continuous_scale="Viridis", size_max=28, zoom=10.3, height=520, hover_name="sector",
    hover_data={"median_price_per_sqft": ":,.0f", "median_price_cr": ":.2f", "listings": True,
                "lat": False, "lon": False},
    labels={"median_price_per_sqft": "₹ / sq ft", "median_price_cr": "Median price (Cr)"},
)
fig_map.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
st.plotly_chart(fig_map, width="stretch")
st.caption(f"{len(by_sector)} sectors with at least 3 listings and known coordinates; bubble size = number of listings.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Most expensive sectors (₹ / sq ft)")
    top = by_sector.nlargest(10, "median_price_per_sqft")
    fig = px.bar(top, x="median_price_per_sqft", y="sector", orientation="h",
                 labels={"median_price_per_sqft": "Median ₹ / sq ft", "sector": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")
with col2:
    st.subheader("Price vs built-up area")
    fig = px.scatter(view, x="built_up_area", y="price", color="property_type", log_x=True, log_y=True,
                     opacity=0.5, labels={"built_up_area": "Built-up area (sq ft)", "price": "Price (₹ Cr)"})
    st.plotly_chart(fig, width="stretch")

col3, col4 = st.columns(2)
with col3:
    st.subheader("Price by number of bedrooms")
    st.plotly_chart(px.box(view[view["bedRoom"] <= 6], x="bedRoom", y="price", log_y=True,
                           labels={"bedRoom": "Bedrooms", "price": "Price (₹ Cr)"}), width="stretch")
with col4:
    metrics_file = ROOT / "models" / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
        st.subheader(f"Model comparison ({m['cv_folds']}-fold cross-validation)")
        models = pd.DataFrame(m["models"]).T.reset_index(names="model")
        models["chosen"] = models["model"].eq(m["best_model"]).map({True: "✅", False: ""})
        st.dataframe(models[["chosen", "model", "mae_cr", "median_ape", "r2_log"]].rename(columns={
            "mae_cr": "MAE (₹ Cr)", "median_ape": "Median error", "r2_log": "R² (log)"}),
            hide_index=True, width="stretch")
        st.caption(f"Chosen by the one-standard-error rule: the cheapest model within one standard error of the "
                   f"lowest MAE ({m['selection']['lowest_mae_model']}).")
