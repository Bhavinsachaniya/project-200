import streamlit as st
import pandas as pd
import pickle

# Load model and required columns
model = pickle.load(open("lead_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# Page setup
st.set_page_config(page_title="🔮 Lead Conversion Predictor", layout="centered")
st.title("🔮 Predictive Lead Conversion App")

# -------- Load CSV --------
st.markdown("Loaded from `Leads.csv` — Edit the values or predict all as-is.")

try:
    # Load and clean data
    df = pd.read_csv("Leads.csv")

    # Show original data (top 100 rows for performance)
    st.dataframe(df.head(100))

    # Encode
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    df_encoded = df_encoded.fillna(0)
    df_encoded = df_encoded.astype(float)

    # Predict
    predictions = model.predict(df_encoded)
    df["Prediction"] = ["✅ Likely to Convert" if p == 1 else "❌ Not Likely to Convert" for p in predictions]

    st.subheader("📊 Predictions on CSV Leads")
    st.dataframe(df)

except Exception as e:
    st.error("Error while predicting from Leads.csv")
    st.code(str(e))

# -------- Manual Prediction --------
st.subheader("🎯 Predict Manually")

with st.form("manual_predict_form"):
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

if submitted:
    try:
        # Input dict and DataFrame
        input_dict = {
            "TotalVisits": total_visits,
            "PageViewsPerVisit": page_views,
            "Lead Origin": lead_origin,
            "Lead Source": lead_source,
            "Do Not Email": do_not_email,
            "Do Not Call": do_not_call
        }

        input_df = pd.DataFrame([input_dict])

        # Encode
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        input_encoded = input_encoded.fillna(0).astype(float)

        # Predict
        prediction = model.predict(input_encoded)[0]
        result = "✅ Likely to Convert" if prediction == 1 else "❌ Not Likely to Convert"
        st.success(f"Prediction: {result}")

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
