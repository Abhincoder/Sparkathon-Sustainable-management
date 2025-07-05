import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import os

def create_features(df_input):
    """
    Applies all feature engineering steps to the DataFrame.
    Assumes 'Date' column is already datetime.
    """
    df = df_input.copy()

    # Date Features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Sort for lag/rolling features
    df = df.sort_values(by=['Store ID', 'Product ID', 'Date'])

    # Lagged Features
    for col in ['Units Sold', 'Inventory Level', 'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing']:
        for lag in [1, 7, 14]: # Lag by 1 day, 1 week, 2 weeks
            df[f'{col}_lag_{lag}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag)

    # Rolling Window Features
    for col in ['Units Sold', 'Inventory Level', 'Demand Forecast']:
        for window in [7, 30]: # 7-day and 30-day rolling averages
            df[f'{col}_rolling_mean_{window}'] = df.groupby(['Store ID', 'Product ID'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'{col}_rolling_std_{window}'] = df.groupby(['Store ID', 'Product ID'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).std())
            # Fill NaN for std where window is 1 (std is 0)
            df[f'{col}_rolling_std_{window}'] = df[f'{col}_rolling_std_{window}'].fillna(0)
    
    # Handle NaNs created by lags/rolling for the start of series (fill with 0 for simplicity)
    # In a real system, you'd ensure proper historical data exists for lag calculations.
    df.fillna(0, inplace=True) 

    return df

def train_models(file_path='retail_store_inventory.csv'):
    """
    Loads data, preprocesses, trains regression and classification models,
    and saves models and encoders.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Create features
    df_processed = create_features(df.copy())

    # Define Target Variable: Potential Waste (Regression)
    # This is an inference: how many units might be left over if Demand Forecast holds true
    df_processed['Waste_Potential'] = np.maximum(0, df_processed['Inventory Level'] - df_processed['Demand Forecast'])

    # Define Waste Risk (Classification)
    # Set a threshold for what constitutes 'High Waste Risk'
    WASTE_THRESHOLD = 10 # Units of potential waste
    df_processed['High_Waste_Risk'] = (df_processed['Waste_Potential'] > WASTE_THRESHOLD).astype(int)

    # Convert Holiday/Promotion to int if not already
    df_processed['Holiday/Promotion'] = df_processed['Holiday/Promotion'].replace({'No':0, 'Yes':1}).astype(int)

    # Categorical Encoding
    categorical_cols_to_encode = ['Category', 'Region', 'Weather Condition', 'Seasonality']
    
    # Use OneHotEncoder to ensure consistent columns during prediction
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df_processed[categorical_cols_to_encode])
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols_to_encode)
    
    df_encoded_part = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_processed.index)
    
    # Drop original categorical columns and concatenate encoded ones
    df_final = pd.concat([df_processed.drop(columns=categorical_cols_to_encode), df_encoded_part], axis=1)

    # Define features and targets
    exclude_cols = ['Date', 'Store ID', 'Product ID', 'Units Sold', 'Units Ordered', 
                    'Inventory Level', 'Demand Forecast', 'Waste_Potential', 'High_Waste_Risk']
                                                              
    final_features = [col for col in df_final.columns if col not in exclude_cols]
    
    X = df_final[final_features]
    y_reg = df_final['Waste_Potential']
    y_clf = df_final['High_Waste_Risk']

    # Time-Series Split
    # Using the last few months for testing, the rest for training
    train_cutoff = df_final['Date'].max() - pd.DateOffset(months=3) # Use last 3 months for test
    
    X_train = X[df_final['Date'] <= train_cutoff]
    y_train_reg = y_reg[df_final['Date'] <= train_cutoff]
    y_train_clf = y_clf[df_final['Date'] <= train_cutoff]

    # Train Regression Model
    lgbm_reg = LGBMRegressor(random_state=42)
    lgbm_reg.fit(X_train, y_train_reg)

    # Train Classification Model
    # Use class_weight for imbalanced datasets if 'High_Waste_Risk' is rare
    lgbm_clf = LGBMClassifier(random_state=42, class_weight='balanced') 
    lgbm_clf.fit(X_train, y_train_clf)

    # Save models and encoder
    joblib.dump(lgbm_reg, 'lgbm_regressor_model.pkl')
    joblib.dump(lgbm_clf, 'lgbm_classifier_model.pkl')
    joblib.dump(encoder, 'onehot_encoder.pkl')
    joblib.dump(final_features, 'model_features.pkl') # Save feature names for consistent prediction input

    print("Models and encoder trained and saved successfully.")
    
    # Return df_processed for RAG system to create historical context
    return df_processed # This df still has original categorical columns, suitable for text description


if __name__ == '__main__':
    # When run directly, train the models
    # Make sure 'retail_inventory_forecasting.csv' is in the same directory
    # or provide the correct path.
    if not os.path.exists('retail_store_inventory.csv'):
        print("Error: 'retail_store_inventory.csv' not found.")
        print("Please ensure the dataset file is in the same directory as model_pipeline.py")
    else:
        trained_df_for_rag = train_models()
        # Optionally save this df to be loaded by rag_system, or pass directly if in same script
        trained_df_for_rag.to_csv('processed_data_for_rag.csv', index=False)
        print("Training complete. Processed data for RAG saved to 'processed_data_for_rag.csv'.")
