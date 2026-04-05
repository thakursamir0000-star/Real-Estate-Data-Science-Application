from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def get_preprocessor():
    """
    Returns the configured ColumnTransformer for the Real Estate Price Prediction pipeline.
    """
    columns_to_encode = [
        'property_type', 'sector', 'balcony', 'agePossession', 
        'furnishing_type', 'luxury_category', 'floor_category'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
            ('cat', OrdinalEncoder(), columns_to_encode)
        ], 
        remainder='passthrough'
    )
    return preprocessor
