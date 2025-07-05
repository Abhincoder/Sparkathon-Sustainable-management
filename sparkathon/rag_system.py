import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import joblib
import sys
from dotenv import load_dotenv
load_dotenv()
# Mock for vector store / embedding (replace with actual implementation)
# In a real scenario, you'd use sentence-transformers and a vector database (e.g., FAISS, ChromaDB)
class MockVectorStore:
    def __init__(self):
        self.documents = [] # Stores textual descriptions
        self.embeddings = [] # Stores mock embeddings (for conceptual use)
        # In a real system, you'd load/build your actual vector index here

    def add_document(self, text_description, data_row):
        # In real RAG, you'd compute a real embedding for text_description here
        mock_embedding = np.random.rand(128) # Simulate an embedding vector
        self.documents.append({"text": text_description, "data": data_row.to_dict()})
        self.embeddings.append(mock_embedding)

    def retrieve_similar(self, query_text_dict, top_k=3):
        # In real RAG, you'd embed query_text and perform similarity search
        # For mock, we'll just return some relevant documents
        # This function would query your actual vector DB
        
        retrieved_context = []
        
        # Simple heuristic for mock: try to find documents that conceptually match
        # category, weather, or promotion status.
        query_category = query_text_dict.get('Category', '').lower()
        query_weather = query_text_dict.get('Weather Condition', '').lower()
        query_promotion = query_text_dict.get('Promotion Status', 'No').lower() # Ensure it's 'no' or 'yes'

        # Filter documents based on conceptual match for demonstration
        potential_matches = []
        for doc_info in self.documents:
            doc_data = doc_info['data']
            doc_category = doc_data.get('Category', '').lower()
            doc_weather = doc_data.get('Weather Condition', '').lower()
            doc_promotion = ('yes' if doc_data.get('Holiday/Promotion', 0) == 1 else 'no').lower()

            match_score = 0
            if query_category and query_category == doc_category:
                match_score += 1
            if query_weather and query_weather == doc_weather:
                match_score += 1
            if query_promotion and query_promotion == doc_promotion:
                match_score += 1
            
            if match_score > 0: # At least one conceptual match
                potential_matches.append((match_score, doc_info['text']))
        
        # Sort by match score (descending) and take top_k
        potential_matches.sort(key=lambda x: x[0], reverse=True)
        retrieved_context = [text for score, text in potential_matches[:top_k]]

        # If no specific matches, just return some general examples or the first few
        if not retrieved_context and self.documents:
            retrieved_context = [doc['text'] for doc in self.documents[:top_k]]

        return retrieved_context


# Global mock vector store
mock_vector_store = MockVectorStore()

def build_rag_context_store(processed_df):
    """
    Builds the conceptual RAG context store from the processed DataFrame.
    This would ideally happen after initial model training.
    """
    print("Building RAG conceptual context store...")
    # Limit for demonstration to avoid huge mock data
    sample_df = processed_df.sample(min(len(processed_df), 1000), random_state=42) 

    for index, row in sample_df.iterrows():
        text_description = (
            f"On {row['Date'].strftime('%Y-%m-%d')}, for Product {row['Product ID']} ({row['Category']}) "
            f"at Store {row['Store ID']} ({row['Region']}), Inventory Level was {row['Inventory Level']:.0f} units, "
            f"Units Sold were {row['Units Sold']:.0f}. Demand Forecast was {row['Demand Forecast']:.0f}. "
            f"Price was ${row['Price']:.2f}, Discount was {row['Discount']:.0%}. Weather was {row['Weather Condition']}, "
            f"Holiday/Promotion was {'Yes' if row['Holiday/Promotion'] == 1 else 'No'}. Competitor Pricing was ${row['Competitor Pricing']:.2f}. "
            f"Seasonality was {row['Seasonality']}. This resulted in a Waste Potential of {row['Waste_Potential']:.0f} units "
            f"and was {'HIGH RISK' if row['High_Waste_Risk'] == 1 else 'low risk'} for waste."
        )
        mock_vector_store.add_document(text_description, row)
    print(f"RAG context store built with {len(mock_vector_store.documents)} historical examples.")


def get_gemini_recommendations(product_id, store_id, category, current_inventory, predicted_waste, 
                               waste_risk_proba, weather, promotion_status, historical_context_list):
    """
    Generates recommendations using Google Gemini, augmented by retrieved historical context.
    """
    try:
        # # Check for API key
        # api_key = os.getenv("GOOGLE_API_KEY")
        api_key=os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "ERROR: GOOGLE_API_KEY not found in environment variables. Please set it to use GenAI features."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        context_str = "\n".join([f"- {h}" for h in historical_context_list])
        if not context_str:
            context_str = "No directly similar historical scenarios were found."

        prompt = f"""
        You are an AI assistant for retail waste management. Your goal is to provide proactive, actionable, and data-driven recommendations to minimize waste.

        Here's the current situation for Product {product_id} ({category}) at Store {store_id}:
        - Current Inventory: {current_inventory} units
        - Predicted Potential Waste: {predicted_waste:.1f} units
        - Waste Risk Probability: {waste_risk_proba:.1%} (Consider high risk if > 50%)
        - Current Weather Condition: {weather}
        - Promotion Status: {promotion_status}

        Here are some historical scenarios and their outcomes/actions that are similar to the current situation:
        {context_str}

        Based on the current situation and the historical context, provide 2-3 specific, actionable recommendations to prevent or reduce waste for Product {product_id}. Think about pricing adjustments, stocking levels, marketing, or inter-store transfers.
        Each recommendation should have a brief justification, referencing the data or insights provided.
        """
        
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"An error occurred while generating recommendations: {e}. Please check your API key and network connection."

# This is a simplified prediction function for the app.py to call.
# It assumes you have the necessary historical data to calculate lags/rolling features
# for the "new" single data point. In a full system, this would be more complex.
def make_single_prediction(current_data, reg_model, clf_model, encoder, model_features, historical_df_for_lags):
    """
    Preprocesses a single new data point and makes a prediction.
    `current_data` is a dictionary or Series with the raw input features.
    `historical_df_for_lags` is needed to compute accurate lags/rolling features for the *new* data point.
    """
    
    # Create a DataFrame for processing, ensure it has all necessary columns
    # Create a DataFrame from the current data point
    input_df = pd.DataFrame([current_data])
    input_df['Date'] = pd.to_datetime(input_df['Date'])
    
    # --- STEP 1: Preprocessing for the new data point ---
    # To correctly calculate lags and rolling means for the *current* row,
    # you need the previous historical data for that specific product and store.
    # We'll append the current data to the historical data, then re-calculate features.
    
    # Filter historical data for this specific product and store
    specific_history = historical_df_for_lags[
        (historical_df_for_lags['Store ID'] == current_data['Store ID']) &
        (historical_df_for_lags['Product ID'] == current_data['Product ID'])
    ].copy()
    
    # Ensure current_data is within a similar date range or sequence for lags
    combined_df = pd.concat([specific_history, input_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=['Date'])

    # Load the create_features function saved during model training
    if not os.path.exists('create_features_function.pkl'):
        # Fallback if not found, though model_pipeline should save it
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model_pipeline import create_features as loaded_create_features
        joblib.dump(loaded_create_features, 'create_features_function.pkl')
    else:
        loaded_create_features = joblib.load('create_features_function.pkl')

    processed_combined_df = loaded_create_features(combined_df.copy())
    
    # The last row of processed_combined_df is our target for prediction
    processed_new_data = processed_combined_df.tail(1)
    
    # Apply OneHotEncoder trained on training data
    categorical_cols_to_encode = ['Category', 'Region', 'Weather Condition', 'Seasonality']
    encoded_features = encoder.transform(processed_new_data[categorical_cols_to_encode])
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols_to_encode)
    
    encoded_df_part = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=processed_new_data.index)
    
    # Drop original categorical columns and concatenate encoded ones
    processed_new_data_final = pd.concat([processed_new_data.drop(columns=categorical_cols_to_encode), encoded_df_part], axis=1)

    # Ensure 'Holiday/Promotion' is integer
    processed_new_data_final['Holiday/Promotion'] = processed_new_data_final['Holiday/Promotion'].replace({'No':0, 'Yes':1}).astype(int)

    # Align columns for prediction: important!
    # Ensure all features expected by the model are present, fill with 0 if missing
    # And ensure the order is correct
    final_input_for_model = pd.DataFrame(columns=model_features)
    for col in model_features:
        if col in processed_new_data_final.columns:
            final_input_for_model[col] = processed_new_data_final[col]
        else:
            final_input_for_model[col] = 0 # Feature not present in new data, assume 0
            
    final_input_for_model = final_input_for_model.fillna(0) # Final check for any NaNs

    # --- STEP 2: Make Predictions ---
    predicted_waste_potential = reg_model.predict(final_input_for_model)[0]
    predicted_waste_risk_proba = clf_model.predict_proba(final_input_for_model)[0, 1]

    return max(0, predicted_waste_potential), predicted_waste_risk_proba

if __name__ == '__main__':
    # This block is for testing rag_system functionality standalone
    # In the real app, these are loaded by app.py
    print("Running rag_system.py standalone test (conceptual)...")
    if os.path.exists('processed_data_for_rag.csv'):
        processed_df = pd.read_csv('processed_data_for_rag.csv')
        processed_df['Date'] = pd.to_datetime(processed_df['Date'])
        
        # Load models and encoder for standalone test
        try:
            reg_model_test = joblib.load('lgbm_regressor_model.pkl')
            clf_model_test = joblib.load('lgbm_classifier_model.pkl')
            encoder_test = joblib.load('onehot_encoder.pkl')
            model_features_test = joblib.load('model_features.pkl')
        except FileNotFoundError:
            print("Models or encoder not found. Please run model_pipeline.py first.")
            sys.exit(1)

        build_rag_context_store(processed_df)

        # Example usage of recommendation function (mock data)
        historical_mock_context = mock_vector_store.retrieve_similar(
            {'Category': 'Groceries', 'Weather Condition': 'Rainy', 'Promotion Status': 'No'}, top_k=2
        )
        
        print("\n--- Example Recommendation (requires GOOGLE_API_KEY) ---")
        print(get_gemini_recommendations(
            product_id='P0001',
            store_id='S001',
            category='Groceries',
            current_inventory=150,
            predicted_waste=75,
            waste_risk_proba=0.8,
            weather='Rainy',
            promotion_status='No',
            historical_context_list=historical_mock_context
        ))

        # Example for make_single_prediction
        print("\n--- Example Single Prediction ---")
        sample_data_point = {
            'Date': pd.to_datetime('2023-01-01'),
            'Store ID': 'S001',
            'Product ID': 'P0001',
            'Category': 'Groceries',
            'Region': 'North',
            'Inventory Level': 250,
            'Units Sold': 100, # Previous day's sales
            'Units Ordered': 50,
            'Demand Forecast': 120,
            'Price': 35.0,
            'Discount': 10.0,
            'Competitor Pricing': 30.0,
            'Weather Condition': 'Sunny',
            'Holiday/Promotion': 'No',
            'Seasonality': 'Winter'
        }
        
        predicted_waste_test, waste_risk_proba_test = make_single_prediction(
            sample_data_point, reg_model_test, clf_model_test, encoder_test, model_features_test, processed_df
        )
        print(f"Predicted Waste Potential: {predicted_waste_test:.1f}")
        print(f"Predicted Waste Risk Probability: {waste_risk_proba_test:.1%}")

    else:
        print("processed_data_for_rag.csv not found. Please run model_pipeline.py first.")
