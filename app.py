
import streamlit as st
import pickle
import pandas as pd

# Load model and expected columns
model = pickle.load(open("lead_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("🔮 Predictive Lead Conversion App")
st.write("This app uses built-in lead data to predict conversion outcomes.")

# Load the provided CSV internally
input_df = pd.read_csv("Leads.csv")

# Drop identifier and target columns
drop_columns = ['Prospect ID', 'Lead Number', 'Converted']
input_df = input_df.drop(columns=[col for col in drop_columns if col in input_df.columns])

# Handle missing columns
missing_cols = [col for col in model_columns if col not in input_df.columns]
for col in missing_cols:
    input_df[col] = 0  # Default value

input_df = input_df[model_columns]  # Reorder and trim
input_df = pd.get_dummies(input_df)  # Encode categoricals

# Align with model columns
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

# Make predictions
predictions = model.predict(input_df)
input_df['Prediction'] = ['Converted ✅' if pred == 1 else 'Not Converted ❌' for pred in predictions]

st.success("Predictions on built-in CSV complete!")
st.dataframe(input_df[['Prediction']])
st.download_button("Download Results as CSV", input_df.to_csv(index=False), file_name="predictions.csv")
