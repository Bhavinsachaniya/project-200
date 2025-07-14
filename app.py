import streamlit as st
import pandas as pd
import pickle

# Load the model and the expected column structure
model = pickle.load(open("lead_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# Streamlit App Setup
st.set_page_config(page_title="🔮 Predictive Lead Conversion App", layout="centered")
st.title("🔮 Predictive Lead Conversion App")
st.write("Upload your lead data CSV to predict which leads are likely to convert.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload Lead CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df)

        # Align with model training columns
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        # Clean data
        df_encoded = df_encoded.fillna(0)
        df_encoded = df_encoded.astype(float)  # ensure no text/nan/infinite values

        # Optional Debug Info
        # st.write("Model Input Preview:")
        # st.dataframe(df_encoded.head())
        # st.write("Any NaNs?", df_encoded.isna().sum().sum())

        # Predict
        predictions = model.predict(df_encoded)

        # Attach predictions to original data
        df["Prediction"] = ["✅ Likely to Convert" if p == 1 else "❌ Not Likely" for p in predictions]

        # Show results
        st.subheader("📈 Prediction Results")
        st.dataframe(df)

        # Download button
        csv_output = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv_output, file_name="lead_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error("⚠️ An error occurred while processing the file.")
        st.text(f"Error details: {e}")

else:
    st.info("Please upload a CSV file to begin.")
