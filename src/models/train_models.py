import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import sys

# Ensure the root directory is accessible for src package absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.build_features import get_preprocessor

def train_and_save():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    models_dir = os.path.join(project_root, 'models')
    data_processed = os.path.join(project_root, 'data', 'processed', 'gurgaon_properties_post_feature_selection_v2.csv')
    data_raw = os.path.join(project_root, 'data', 'raw', 'gurgaon_properties.csv')
    
    os.makedirs(models_dir, exist_ok=True)
    print("Loading data...")
    try:
        df = pd.read_csv(data_processed)
    except Exception:
        df = pd.read_csv(data_raw)
        
    if 'price' not in df.columns:
        print("Price column not found! Ensure the correct CSV is used.")
        return

    # Assuming these are the columns used for modeling based on notebooks
    columns_to_encode = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']
    
    if set(columns_to_encode).issubset(df.columns):
        X = df.drop(columns=['price'])
        y = df['price']
        y_log = np.log1p(y)

        print("Building pipeline...")
        preprocessor = get_preprocessor()

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        print("Training model...")
        pipeline.fit(X, y_log)

        print("Saving pipeline and dataframe...")
        with open(os.path.join(models_dir, 'pipeline.pkl'), 'wb') as file:
            pickle.dump(pipeline, file)

        with open(os.path.join(models_dir, 'df.pkl'), 'wb') as file:
            pickle.dump(X, file)
            
        print("Models saved successfully in 'models/' directory.")
    else:
        print("Columns missing from dataset for training model.")

if __name__ == '__main__':
    train_and_save()
