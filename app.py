import streamlit as st
import pandas as pd
import pickle

# Load model and columns
model = pickle.load(open("lead_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# Load leads from CSV
leads_df = pd.read_csv("Leads.csv")

# Streamlit Page Setup
st.set_page_config(page_title="🔮 Lead Conversion Predictor", layout="centered")
st.title("🔮 Predictive Lead Conversion App")
st.markdown("Loaded from `Leads.csv` — Edit the values or predict all below.")

# --------- Prediction on CSV ---------
st.subheader("📊 Predictions from Leads.csv")

# Encode and predict
leads_encoded = pd.get_dummies(leads_df)
leads_encoded = leads_encoded.reindex(columns=model_columns, fill_value=0)
leads_encoded = leads_encoded.fillna(0).astype(float)

# Predict
try:
    predictions = model.predict(leads_encoded)
    leads_df["Prediction"] = ["✅ Likely to Convert" if p == 1 else "❌ Not Likely to Convert" for p in predictions]
    st.dataframe(leads_df)
except Exception as e:
    st.error("Prediction on CSV failed.")
    st.text(f"Details: {e}")

# --------- Manual Form ---------
st.subheader("🧮 Predict Manually")

with st.form("lead_form"):
    col1, col2 = st.columns(2)

    with col1:
        total_visits = st.number_input("Total Visits", min_value=0, max_value=100, value=5)
        lead_origin = st.selectbox("Lead Origin", [
            "Landing Page Submission", "API", "Lead Add Form", "Lead Import"
        ])
        do_not_email = st.selectbox("Do Not Email", ["Yes", "No"])

    with col2:
        page_views = st.number_input("Page Views Per Visit", min_value=0, max_value=20, value=3)
        lead_source = st.selectbox("Lead Source", [
            "Google", "Direct Traffic", "Olark Chat", "Reference", "Welingak Website", "Facebook", "Others"
        ])
        do_not_call = st.selectbox("Do Not Call", ["Yes", "No"])

    submitted = st.form_submit_button("🚀 Predict Conversion")

# Manual Prediction
if submitted:
    input_dict = {
        "TotalVisits": total_visits,
        "PageViewsPerVisit": page_views,
        "Lead Origin": lead_origin,
        "Lead Source": lead_source,
        "Do Not Email": do_not_email,
        "Do Not Call": do_not_call
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    input_encoded = input_encoded.fillna(0).astype(float)

    try:
        prediction = model.predict(input_encoded)[0]
        result = "✅ Likely to Convert" if prediction == 1 else "❌ Not Likely to Convert"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error("Manual prediction failed.")
        st.text(f"Details: {e}")
