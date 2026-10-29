import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    page_icon="🏘️",
    layout="wide"
)

st.title("🏘️ Gurgaon Real Estate: Price Prediction & Recommendations")
st.markdown("""
An end-to-end data science project on ~3,500 Gurgaon property listings (scraped from 99acres):
cleaning and feature engineering in notebooks, a cross-validated price model, and this app.

👈 **Pick a module from the sidebar:**

* **💰 Price Predictor** — an estimate with an honest price range and the features driving it.
* **📊 Analytics Dashboard** — a sector map of price per sq ft, distributions and price drivers.
* **🏡 Recommend Properties** — listings similar to one you like, and housing societies with similar facilities, prices and surroundings.
""")

metrics_file = ROOT / "models" / "metrics.json"
if metrics_file.exists():
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    best = metrics["models"][metrics["best_model"]]
    coverage = metrics["intervals"]["80"]["coverage"]["overall"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Typical error (median)", f"{best['median_ape']:.0%}")
    c2.metric("Mean absolute error", f"₹{best['mae_cr']:.2f} Cr")
    c3.metric("80% price ranges that contain the true price", f"{coverage:.0%}")
    st.caption(f"{metrics['cv_folds']}-fold cross-validation on {metrics['data']['rows_after_dedup']:,} de-duplicated "
               f"listings · model: {metrics['best_model']}")

st.markdown("---\n**Author:** Samir Thakur")
