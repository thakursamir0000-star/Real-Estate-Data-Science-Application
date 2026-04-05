import streamlit as st
import pickle
import pandas as pd
import numpy as np

import os

st.set_page_config(page_title="Price Predictor", page_icon="💰")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
models_dir = os.path.join(project_root, 'models')

try:
    with open(os.path.join(models_dir, 'pipeline.pkl'), 'rb') as file:
        pipeline = pickle.load(file)
    with open(os.path.join(models_dir, 'df.pkl'), 'rb') as file:
        df = pickle.load(file)
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure train_models.py has been run.")
    st.stop()

st.title("💰 Property Price Predictor")
st.write("Enter the details of the property to estimate its price in Crores.")

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Property Type", sorted(df['property_type'].unique().tolist()))
    sector = st.selectbox("Sector", sorted(df['sector'].unique().tolist()))
    bedroom = st.number_input("Number of Bedrooms", min_value=1.0, max_value=20.0, value=2.0, step=1.0)
    bathroom = st.number_input("Number of Bathrooms", min_value=1.0, max_value=20.0, value=2.0, step=1.0)
    balcony = st.selectbox("Number of Balconies", sorted(df['balcony'].unique().tolist()))
    agePossession = st.selectbox("Age of Property / Possession", sorted(df['agePossession'].unique().tolist()))

with col2:
    built_up_area = st.number_input("Built Up Area (sq ft)", min_value=100.0, max_value=50000.0, value=1000.0, step=10.0)
    servant_room = st.selectbox("Servant Room", [0.0, 1.0])
    store_room = st.selectbox("Store Room", [0.0, 1.0])
    furnishing_type = st.selectbox("Furnishing Type", sorted(df['furnishing_type'].unique().tolist()))
    luxury_category = st.selectbox("Luxury Category", sorted(df['luxury_category'].unique().tolist()))
    floor_category = st.selectbox("Floor Category", sorted(df['floor_category'].unique().tolist()))

if st.button("Predict Price", use_container_width=True):
    input_df = pd.DataFrame(
        [[property_type, sector, bedroom, bathroom, balcony, agePossession, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]],
        columns=['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 'agePossession', 'built_up_area', 'servant room', 'store room', 'furnishing_type', 'luxury_category', 'floor_category']
    )
    
    try:
        # Predict uses expm1 because the model was trained on log1p
        predicted_price = np.expm1(pipeline.predict(input_df))[0]
        st.success(f"Estimated Property Price: **₹ {predicted_price:.2f} Crores**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
