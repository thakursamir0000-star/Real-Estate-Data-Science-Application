import streamlit as st

st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    page_icon="🏘️",
    layout="wide"
)

st.title("🏘️ Real Estate Data Science Application")
st.markdown("""
### Welcome to the Real Estate Analytics & Recommendation System!
This is an end-to-end Data Science project that analyzes real estate data, predicts property prices, and recommends properties based on user preferences.

👈 **Please select a module from the sidebar to get started:**

* **💰 Price Predictor:** Estimate property prices based on various features such as location, area, bedrooms, and more.
* **📊 Analytics Dashboard:** Gain insights into the real estate market through geographic distributions, trend visualizations, and scatter plots.
* **🏡 Recommend Properties:** Get personalized property suggestions based on top facilities, price preferences, and location advantages.

---
**Author:** Samir Thakur
""")
