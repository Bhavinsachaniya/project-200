import streamlit as st
import pandas as pd
import pickle

# Load the model and expected input columns
try:
    model = pickle.load(open("lead_model.pkl", "rb"))
    model_columns = pickle.load(open("model_columns.pkl", "rb"))
except Exception as e:
    st.error("❌ Failed to load model or column structure.")
    st.stop()

# App title and config
st.set_page_config(page_title="🔮 Predictive Lead Conversion App", layout="centered")
st.title("🔮 Predictive Lead Conversion App")
st.write("Upload your lead data CSV to predict which leads are likely to convert.")

# CSV upload
uploaded_file = st.file_uploader("📁 Upload Lead CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        # Validate structure
        if df.empty:
            st.error("⚠️ Uploaded file is empty.")
            st.stop()

        # Preprocess: one-hot encode, align columns
        df_encoded = pd.get_dummies(df)

        # Align with model structure
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        # Clean: fill missing, ensure numeric
        df_encoded = df_encoded.fillna(0)
        df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Extra check for infinite or invalid values
        if not df_encoded.replace([float("inf"), float("-inf")], 0).applymap(pd.api.types.is_number).all().all():
            st.error("⚠️ Invalid or non-numeric values found in input.")
            st.stop()

        # Predict
        predictions = model.predict(df_encoded)

        # Append predictions to original data
        df["Prediction"] = ["✅ Likely to Convert" if p == 1 else "❌ Not Likely" for p in predictions]

        # Show predictions
        st.subheader("📈 Prediction Results")
        st.dataframe(df)

        # Download button
        result_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Prediction Results", data=result_csv, file_name="lead_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error("⚠️ Error processing file or predicting.")
        st.text(f"Details: {e}")

else:
    st.info("Please upload a CSV file to begin.")
