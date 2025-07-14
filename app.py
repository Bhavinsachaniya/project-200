import streamlit as st
import pickle
import pandas as pd

# Load model and column structure
model = pickle.load(open("lead_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Leads.csv")

df = load_data()

st.title("🔮 Predictive Lead Conversion App")
st.write("Select a lead from the dataset to check if it will convert.")

# Show dropdown to pick a lead row
lead_index = st.selectbox("Select Lead Row Index", df.index)

# Get the selected lead data
selected_lead = df.loc[lead_index]

# Display the selected lead
st.subheader("Selected Lead Details")
st.dataframe(selected_lead.to_frame().T)

# Prepare lead for prediction
if st.button("Predict Conversion"):
    try:
        input_features = selected_lead.drop(labels=["Prospect ID", "Lead Number", "Converted"], errors="ignore")
        input_df = pd.DataFrame([input_features])
        input_df = input_df.fillna("0").astype(str)
        input_df = input_df[model_columns].astype(float)
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {'Converted ✅' if prediction == 1 else 'Not Converted ❌'}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
