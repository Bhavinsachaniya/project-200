import streamlit as st
import pandas as pd
import pickle

# --- Load model and required columns ---
model = pickle.load(open("lead_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# --- Page Configuration ---
st.set_page_config(page_title="🔍 Predictive Lead Conversion AI", layout="centered")
st.title("🔍 Predictive Lead Conversion AI")
st.markdown("Enter lead details below to check if the user is likely to convert.")

# --- Upload CSV Section ---
uploaded_file = st.file_uploader("📂 Upload Leads CSV File (Optional)", type=["csv"])

if uploaded_file:
    st.subheader("📄 Uploaded Data Preview")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    try:
        df.fillna(0, inplace=True)

        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
        df_encoded = df_encoded.astype(float)

        predictions = model.predict(df_encoded)
        df["Prediction"] = predictions
        df["Result"] = ["✅ Likely to Convert" if p == 1 else "❌ Not Likely to Convert" for p in predictions]

        # Show only "Yes" predictions
        st.subheader("✅ Leads Likely to Convert")
        yes_leads = df[df["Prediction"] == 1]

        if not yes_leads.empty:
            st.dataframe(yes_leads)
            csv = yes_leads.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Likely Leads as CSV", csv, "likely_leads.csv", "text/csv")
        else:
            st.info("No leads were predicted as likely to convert.")

    except Exception as e:
        st.error("Prediction failed on uploaded data.")
        st.text(f"Details: {e}")

st.markdown("---")

# --- Manual Entry Form ---
st.subheader("🧮 Predict Manually")

with st.form("lead_form"):
    total_visits = st.number_input("Total Visits", min_value=0, max_value=100, value=5)
    total_time_spent = st.number_input("Total Time Spent on Website", min_value=0, max_value=200, value=10)
    page_views = st.number_input("Page Views Per Visit", min_value=0, max_value=20, value=2)

    lead_origin = st.selectbox("Lead Origin", [
        "Landing Page Submission", "API", "Lead Add Form", "Lead Import"
    ], index=0)

    lead_source = st.selectbox("Lead Source", [
        "Google", "Direct Traffic", "Olark Chat", "Reference", "Welingak Website", "Facebook", "Others"
    ], index=0)

    do_not_email = st.selectbox("Do Not Email", ["Yes", "No"], index=1)
    do_not_call = st.selectbox("Do Not Call", ["Yes", "No"], index=1)

    submitted = st.form_submit_button("🚀 Predict Conversion")

# --- Manual Prediction Logic ---
if submitted:
    input_dict = {
        "TotalVisits": total_visits,
        "Total Time Spent on Website": total_time_spent,
        "PageViewsPerVisit": page_views,
        "Lead Origin": lead_origin,
        "Lead Source": lead_source,
        "Do Not Email": do_not_email,
        "Do Not Call": do_not_call
    }

    input_df = pd.DataFrame([input_dict])
    input_df_encoded = pd.get_dummies(input_df)
    input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    input_df_encoded = input_df_encoded.astype(float)

    try:
        prediction = model.predict(input_df_encoded)[0]
        result = "✅ Likely to Convert" if prediction == 1 else "❌ Not Likely to Convert"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error("Prediction failed.")
        st.text(f"Details: {e}")
