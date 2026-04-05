import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

import os

st.set_page_config(page_title="Recommend Properties", page_icon="🏡", layout="wide")

st.title("🏡 Property Recommendation System")
st.write("Get personalized property suggestions based on your preferences.")

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

st.sidebar.header("Your Preferences")

# Preferences
if 'sector' in df.columns:
    sectors = st.sidebar.multiselect("Preferred Sectors", df['sector'].unique())
else:
    sectors = []

if 'price' in df.columns:
    min_price, max_price = st.sidebar.slider("Price Range (Crores)", float(df['price'].min()), float(df['price'].max()), (float(df['price'].min()), float(df['price'].max())))
else:
    min_price, max_price = 0, 100

if 'bedRoom' in df.columns:
    bedrooms = st.sidebar.slider("Minimum Bedrooms", int(df['bedRoom'].min()), int(df['bedRoom'].max()), int(df['bedRoom'].min()))
else:
    bedrooms = 1

st.write("### Recommended Properties")

# Filtering logic
filtered_df = df.copy()

if sectors:
    filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]

if 'price' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['price'] >= min_price) & (filtered_df['price'] <= max_price)]

if 'bedRoom' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['bedRoom'] >= bedrooms]

if filtered_df.empty:
    st.warning("No properties match your exact preferences. Here are some close matches.")
    # Show generic top 10
    if 'price' in df.columns:
        st.dataframe(df.head(10))
else:
    st.success(f"Found {len(filtered_df)} properties matching your criteria!")
    st.dataframe(filtered_df.reset_index(drop=True))

st.markdown("---")
st.write("##### Similar Properties (Content-Based)")
st.write("Select a property you like to see 5 similar ones.")

if 'sector' in df.columns and 'property_type' in df.columns:
    # Creating a unique identifier for selection
    df['prop_name'] = df['property_type'].astype(str) + " in " + df['sector'].astype(str) + " (₹ " + df['price'].astype(str) + " Cr)"
    selected_prop = st.selectbox("Select a Property", df['prop_name'].tolist())

    if st.button("Find Similar Properties"):
        if 'built_up_area' in df.columns and 'price' in df.columns and 'bedRoom' in df.columns:
            features = ['price', 'bedRoom', 'bathroom', 'built_up_area']
            
            # Handling NaNs
            df_features = df[features].fillna(0)
            
            prop_idx = df[df['prop_name'] == selected_prop].index[0]
            selected_features = df_features.iloc[prop_idx].values.reshape(1, -1)
            
            # Compute Euclidean distances
            distances = euclidean_distances(df_features, selected_features)
            df['distance'] = distances
            
            # Get top 5 excluding the identical property
            similar_props = df[df.index != prop_idx].sort_values('distance').head(5)
            
            st.write(f"Properties similar to **{selected_prop}**:")
            st.dataframe(similar_props.drop(columns=['distance', 'prop_name']).reset_index(drop=True))
        else:
            st.info("Additional features required for similarity matching.")
