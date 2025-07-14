import streamlit as st
import pickle
import pandas as pd

# Load everything
model = pickle.load(open("lead_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("Leads.csv")

df = load_data()

st.title("🔮 Predictive Lead Conversion App")
st.write("Select a lead from the dataset to check if it will convert.")

lead_index = st.selectbox("Select Lead Row Index", df.index)
selected_lead = df.loc[lead_index]
st.subheader("Selected Lead Details")
st.dataframe(selected_lead.to_frame().T)

if st.button("Predict Conversion"):
    try:
        input_features = selected_lead.drop(labels=["Prospect ID", "Lead Number", "Converted"], errors="ignore")
        input_df = pd.DataFrame([input_features])

        # Fill NA and apply label encoding
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            else:
                input_df[col] = input_df[col].fillna(0).astype(float)

        input_df = input_df[model_columns]
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {'Converted ✅' if prediction == 1 else 'Not Converted ❌'}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
