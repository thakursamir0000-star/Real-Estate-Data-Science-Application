import os
import pandas as pd

def load_and_clean_data(input_path: str, output_path: str):
    """
    Standard script to load raw data, perform initial cleaning, 
    and save exactly what's needed for modeling to the processed directory.
    """
    print(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Normally, your data cleaning logic from notebooks goes here
    # Example: df.dropna(inplace=True)
    # Example: df = remove_outliers(df)
    
    print(f"Saving processed data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    raw_path = os.path.join(project_root, 'data', 'raw', 'gurgaon_properties.csv')
    processed_path = os.path.join(project_root, 'data', 'processed', 'gurgaon_properties_cleaned.csv')
    
    load_and_clean_data(raw_path, processed_path)
