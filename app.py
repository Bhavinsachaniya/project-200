
import streamlit as st
import pickle
import pandas as pd

# Load the model and column structure
model = pickle.load(open("lead_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("🔮 Predictive Lead Conversion App")
st.write("Enter lead information to predict if they will convert.")

# Create input fields for each column
input_data = {}
for col in model_columns:
    input_data[col] = st.text_input(f"{col}", "")

if st.button("Predict Conversion"):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df.fillna("0").astype(float)
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {'Converted ✅' if prediction == 1 else 'Not Converted ❌'}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
