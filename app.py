import streamlit as st
import pandas as pd
import pickle

# Load model and expected columns
model = pickle.load(open("lead_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# Streamlit Page Config
st.set_page_config(page_title="🔮 Lead Conversion Predictor", layout="centered")
st.title("🔮 Predictive Lead Conversion App")
st.markdown("Enter lead details to predict conversion probability.")

# --- UI FORM ---
with st.form("lead_form"):
    col1, col2 = st.columns(2)

    with col1:
        total_visits = st.number_input("Total Visits", min_value=0, max_value=100, value=5)
        lead_origin = st.selectbox("Lead Origin", [
            "Landing Page Submission", "API", "Lead Add Form", "Lead Import"
        ])
        do_not_email = st.selectbox("Do Not Email", ["Yes", "No"])

    with col2:
        page_views = st.number_input("Page Views Per Visit", min_value=0, max_value=20, value=2)
        lead_source = st.selectbox("Lead Source", [
            "Google", "Direct Traffic", "Olark Chat", "Reference", "Welingak Website", "Facebook", "Others"
        ])
        do_not_call = st.selectbox("Do Not Call", ["Yes", "No"])

    submitted = st.form_submit_button("🚀 Predict Conversion")

# --- Prediction Logic ---
if submitted:
    # Create input dictionary
    input_dict = {
        "TotalVisits": total_visits,
        "PageViewsPerVisit": page_views,
        "Lead Origin": lead_origin,
        "Lead Source": lead_source,
        "Do Not Email": do_not_email,
        "Do Not Call": do_not_call
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encoding to match model training
    input_df_encoded = pd.get_dummies(input_df)

    # Align with model structure
    input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    input_df_encoded = input_df_encoded.fillna(0)
    input_df_encoded = input_df_encoded.astype(float)

    try:
        prediction = model.predict(input_df_encoded)[0]
        result = "✅ Likely to Convert" if prediction == 1 else "❌ Not Likely to Convert"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error("Prediction failed due to input mismatch or model issue.")
        st.text(f"Details: {e}")
