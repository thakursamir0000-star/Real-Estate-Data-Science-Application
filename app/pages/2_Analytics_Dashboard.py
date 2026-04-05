import streamlit as st
import pandas as pd
import plotly.express as px
import ast

import os

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

st.title("📊 Real Estate Analytics Dashboard")

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_processed = os.path.join(project_root, 'data', 'processed', 'gurgaon_properties_post_feature_selection_v2.csv')
    data_raw = os.path.join(project_root, 'data', 'raw', 'gurgaon_properties.csv')

    try:
        # Load the cleaned dataset
        df = pd.read_csv(data_processed)
    except Exception:
        df = pd.read_csv(data_raw)
    return df

df = load_data()

st.write("### Property Distribution")

col1, col2 = st.columns(2)

with col1:
    st.write("#### Distribution of Property Types")
    if 'property_type' in df.columns:
        fig1 = px.pie(df, names='property_type', title='Property Types', hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.write("#### Sectors with Highest Average Prices")
    if 'sector' in df.columns and 'price' in df.columns:
        avg_price = df.groupby('sector')['price'].mean().sort_values(ascending=False).head(10).reset_index()
        fig2 = px.bar(avg_price, x='sector', y='price', title='Top 10 Sectors by Average Price')
        st.plotly_chart(fig2, use_container_width=True)

st.write("### Price vs Attributes")
cols = st.columns(2)

with cols[0]:
    st.write("#### Price vs Built-up Area")
    if 'built_up_area' in df.columns and 'price' in df.columns:
        fig3 = px.scatter(df, x='built_up_area', y='price', color='property_type', 
                          title="Price vs Built-up Area")
        st.plotly_chart(fig3, use_container_width=True)

with cols[1]:
    st.write("#### Price vs Number of Bedrooms")
    if 'bedRoom' in df.columns and 'price' in df.columns:
        fig4 = px.box(df, x='bedRoom', y='price', title="Price vs Number of Bedrooms")
        st.plotly_chart(fig4, use_container_width=True)

st.write("### Additional Attributes")
if 'luxury_category' in df.columns:
    fig5 = px.histogram(df, x='luxury_category', color='property_type', barmode='group', title="Property Count by Luxury Category")
    st.plotly_chart(fig5, use_container_width=True)

