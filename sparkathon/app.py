import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Add current directory to path to import local modules
# This is crucial for Streamlit to find your model_pipeline.py and rag_system.py
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import functions from your custom modules
from model_pipeline import create_features # We need this function to re-create features for new data
from rag_system import build_rag_context_store, get_gemini_recommendations, make_single_prediction, mock_vector_store

# --- Load Models and Encoder ---
@st.cache_resource
def load_resources():
    try:
        reg_model = joblib.load(os.path.join(script_dir, 'lgbm_regressor_model.pkl'))
        clf_model = joblib.load(os.path.join(script_dir, 'lgbm_classifier_model.pkl'))
        encoder = joblib.load(os.path.join(script_dir, 'onehot_encoder.pkl'))
        model_features = joblib.load(os.path.join(script_dir, 'model_features.pkl'))
        
        # Load the original processed data for RAG context building and lag calculation
        processed_data_path = os.path.join(script_dir, 'processed_data_for_rag.csv')
        if os.path.exists(processed_data_path):
            historical_df_for_lags = pd.read_csv(processed_data_path)
            historical_df_for_lags['Date'] = pd.to_datetime(historical_df_for_lags['Date'])
            
            # Save the create_features function itself for make_single_prediction to load
            joblib.dump(create_features, os.path.join(script_dir, 'create_features_function.pkl'))
            
            # Build RAG context (conceptual for mock vector store)
            build_rag_context_store(historical_df_for_lags)
        else:
            st.error("Error: 'processed_data_for_rag.csv' not found. Please run model_pipeline.py first to generate it.")
            historical_df_for_lags = pd.DataFrame() # Empty df to prevent errors
            
        return reg_model, clf_model, encoder, model_features, historical_df_for_lags
    except FileNotFoundError as e:
        st.error(f"Required model or data file not found: {e}. Please ensure 'model_pipeline.py' was run successfully and all necessary files are in the correct directory.")
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

reg_model, clf_model, encoder, model_features, historical_df_for_lags = load_resources()


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI-Powered Waste Management for Retail")

st.title("♻️ AI-Powered Waste Management for Retail")
st.markdown("Optimize inventory, reduce waste, and enhance sustainability using predictive analytics and Generative AI.")

st.sidebar.header("Prediction Input")

# Input fields for a new prediction
current_date = st.sidebar.date_input("Date for Prediction", pd.to_datetime('2023-01-01')) # Example date

# Get unique values for dropdowns, handling empty dataframe if load_resources failed
unique_store_ids = historical_df_for_lags['Store ID'].unique() if not historical_df_for_lags.empty else ['S001']
unique_product_ids = historical_df_for_lags['Product ID'].unique() if not historical_df_for_lags.empty else ['P0001']
unique_categories = historical_df_for_lags['Category'].unique() if not historical_df_for_lags.empty else ['Groceries']
unique_regions = historical_df_for_lags['Region'].unique() if not historical_df_for_lags.empty else ['North']
unique_weather_conditions = historical_df_for_lags['Weather Condition'].unique() if not historical_df_for_lags.empty else ['Sunny']
unique_seasonalities = historical_df_for_lags['Seasonality'].unique() if not historical_df_for_lags.empty else ['Autumn']


store_id = st.sidebar.selectbox("Store ID", unique_store_ids)
product_id = st.sidebar.selectbox("Product ID", unique_product_ids)

# Get default values from historical_df_for_lags for selected store/product
filtered_product_data = historical_df_for_lags[
    (historical_df_for_lags['Store ID'] == store_id) & 
    (historical_df_for_lags['Product ID'] == product_id)
].iloc[-1] if not historical_df_for_lags.empty and not historical_df_for_lags[(historical_df_for_lags['Store ID'] == store_id) & (historical_df_for_lags['Product ID'] == product_id)].empty else {}

# Helper to safely get index for selectbox default value
def get_selectbox_index(options, current_value):
    try:
        return list(options).index(current_value)
    except ValueError:
        return 0 # Default to first option if current_value not found

category = st.sidebar.selectbox("Category", unique_categories, 
                               index=get_selectbox_index(unique_categories, filtered_product_data.get('Category', 'Groceries')))
region = st.sidebar.selectbox("Region", unique_regions,
                             index=get_selectbox_index(unique_regions, filtered_product_data.get('Region', 'North')))

inventory_level = st.sidebar.number_input("Current Inventory Level", min_value=0, value=int(filtered_product_data.get('Inventory Level', 100)))
units_sold_prev = st.sidebar.number_input("Previous Day Units Sold", min_value=0, value=int(filtered_product_data.get('Units Sold_lag_1', 50)))
units_ordered = st.sidebar.number_input("Units Ordered for this period", min_value=0, value=int(filtered_product_data.get('Units Ordered', 20)))
demand_forecast = st.sidebar.number_input("Manual Demand Forecast (if available)", min_value=0, value=int(filtered_product_data.get('Demand Forecast', 60)))
price = st.sidebar.number_input("Current Price ($)", min_value=0.01, value=float(filtered_product_data.get('Price', 10.0)))
discount = st.sidebar.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=float(filtered_product_data.get('Discount', 0.0)))
competitor_pricing = st.sidebar.number_input("Competitor Pricing ($)", min_value=0.01, value=float(filtered_product_data.get('Competitor Pricing', 9.5)))

weather_condition = st.sidebar.selectbox("Weather Condition", unique_weather_conditions,
                                        index=get_selectbox_index(unique_weather_conditions, filtered_product_data.get('Weather Condition', 'Sunny')))
holiday_promotion = st.sidebar.radio("Holiday/Promotion", ['No', 'Yes'], 
                                    index=int(filtered_product_data.get('Holiday/Promotion', 0)) if not historical_df_for_lags.empty else 0)
seasonality = st.sidebar.selectbox("Seasonality", unique_seasonalities,
                                  index=get_selectbox_index(unique_seasonalities, filtered_product_data.get('Seasonality', 'Autumn')))

if st.sidebar.button("Predict Waste & Get Recommendations"):
    # Prepare input for prediction
    input_data = {
        'Date': current_date,
        'Store ID': store_id,
        'Product ID': product_id,
        'Category': category,
        'Region': region,
        'Inventory Level': inventory_level,
        'Units Sold': units_sold_prev, # Using this as previous day's sales for lag1 calculation conceptually
        'Units Ordered': units_ordered,
        'Demand Forecast': demand_forecast,
        'Price': price,
        'Discount': discount,
        'Competitor Pricing': competitor_pricing,
        'Weather Condition': weather_condition,
        'Holiday/Promotion': holiday_promotion,
        'Seasonality': seasonality
        # Lagged and rolling features will be calculated internally by make_single_prediction
    }

    st.subheader("Waste Prediction Results")
    with st.spinner("Predicting potential waste..."):
        predicted_waste, waste_risk_proba = make_single_prediction(
            input_data, reg_model, clf_model, encoder, model_features, historical_df_for_lags
        )

    st.success(f"**Predicted Potential Waste:** {predicted_waste:.1f} units")
    st.info(f"**Waste Risk Probability:** {waste_risk_proba:.1%} (Higher means more risk)")

    if predicted_waste > 5 or waste_risk_proba > 0.5: # Threshold for showing recommendations
        st.subheader("AI-Powered Recommendations to Minimize Waste")
        with st.spinner("Generating recommendations with Generative AI..."):
            # Retrieve historical context for RAG
            # For demonstration, mock_vector_store will provide general examples
            # In a real system, you'd use a more sophisticated query based on current_data
            historical_context = mock_vector_store.retrieve_similar(
                {'Category': category, 'Weather Condition': weather_condition, 
                 'Promotion Status': holiday_promotion}, top_k=3
            )

            recommendations = get_gemini_recommendations(
                product_id=product_id,
                store_id=store_id,
                category=category,
                current_inventory=inventory_level,
                predicted_waste=predicted_waste,
                waste_risk_proba=waste_risk_proba,
                weather=weather_condition,
                promotion_status=holiday_promotion,
                historical_context_list=historical_context
            )
        st.markdown(recommendations)
    else:
        st.success("Waste risk is currently low. Continue to monitor inventory and demand.")

st.markdown("---")
st.markdown("### How it Works:")
st.markdown("""
1.  **Data Ingestion & Feature Engineering:** The system processes historical sales, inventory, weather, and promotion data. It creates advanced features like lagged sales and rolling averages.
2.  **Predictive Models (LightGBM):** Two machine learning models are trained:
    * A **Regression Model** predicts the *quantity* of potential unsold items (waste potential).
    * A **Classification Model** predicts the *probability* of an item being at high risk of waste.
3.  **Generative AI (Gemini) with RAG:**
    * When a potential waste risk is identified, the system retrieves similar historical scenarios (e.g., past overstock situations during similar weather or promotions).
    * This historical context is provided to a Generative AI model (Google Gemini).
    * Gemini then generates actionable, context-aware recommendations (e.g., dynamic pricing, smart stocking, inter-store transfers) to proactively prevent waste.
4.  **Actionable Dashboard:** The predictions and recommendations are presented in an easy-to-understand format for retail associates.
""")
st.caption("Developed for a smarter, greener, and more responsible future for retail.")
