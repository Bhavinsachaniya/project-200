import streamlit as st
import pandas as pd
import pickle

# Load your trained model and expected input columns
model = pickle.load(open("lead_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.set_page_config(page_title="Lead Conversion Predictor", layout="centered")
st.title("📊 Lead Conversion Predictor")
st.write("Upload your lead data in CSV format to predict whether leads will convert.")

# Upload section
uploaded_file = st.file_uploader("📁 Upload Lead CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and display uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Lead Data")
        st.dataframe(df.head())

        # One-hot encoding (matching model training)
        df_encoded = pd.get_dummies(df)

        # Align with model's expected input structure
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        # Clean any missing values
        df_encoded = df_encoded.fillna(0)

        # Run predictions
        predictions = model.predict(df_encoded)

        # Add prediction results to the original DataFrame
        df["Prediction"] = ["✅ Likely to Convert" if p == 1 else "❌ Not Likely" for p in predictions]

        # Show prediction results
        st.subheader("✅ Prediction Results")
        st.dataframe(df)

        # Download button for results
        csv_output = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv_output, file_name="lead_predictions.csv", mime='text/csv')

    except Exception as e:
        st.error("⚠️ Error processing the file. Please ensure the CSV has correct structure.")
        st.text(f"Error: {e}")
else:
    st.info("Please upload your lead CSV file to get started.")
