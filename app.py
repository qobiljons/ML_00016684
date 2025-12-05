import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/best_model.pkl")
features = joblib.load("models/model_features.pkl")

num_cols = ["Sample_Size","Low_Confidence_Limit","High_Confidence_Limit","Data_Value"]

scaler = joblib.load("models/scaler.pkl")

st.title("Obesity Prevalence Prediction App")

year = st.selectbox("YearStart", [2011, 2012, 2013, 2014, 2015])
location = st.text_input("Location (State Name or Abbreviation)", "Alabama")
education = st.selectbox("Education", ["Less than high school","High school graduate","Some college","College graduate"])
sex = st.selectbox("Sex", ["Male","Female"])
income = st.selectbox("Income", [
    "Less than $15,000",
    "$15,000 - $24,999",
    "$25,000 - $34,999",
    "$35,000 - $49,999",
    "$50,000 - $74,999",
    "$75,000 or greater"
])
race = st.selectbox("Race/Ethnicity", [
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Hispanic",
    "Asian",
    "American Indian/Alaska Native",
    "Other"
])

sample_size = st.number_input("Sample Size", min_value=1, value=500)
low_conf = st.number_input("Low Confidence Limit", value=20.0)
high_conf = st.number_input("High Confidence Limit", value=30.0)

df_input = pd.DataFrame({
    "YearStart": [year],
    "LocationDesc": [location],
    "Education": [education],
    "Sex": [sex],
    "Income": [income],
    "Race/Ethnicity": [race],
    "Sample_Size": [sample_size],
    "Low_Confidence_Limit": [low_conf],
    "High_Confidence_Limit": [high_conf],
    "Data_Value": [0]
})

df_input[num_cols] = scaler.transform(df_input[num_cols])

df_input["Confidence_Range"] = df_input["High_Confidence_Limit"] - df_input["Low_Confidence_Limit"]
df_input["Value_Per_Sample"] = df_input["Data_Value"] / (df_input["Sample_Size"] + 1e-6)

df_input = pd.get_dummies(df_input, drop_first=True)
df_input = df_input.reindex(columns=features, fill_value=0)

if st.button("Predict Obesity %"):
    pred_scaled = model.predict(df_input)[0]

    mean_val = scaler.mean_[num_cols.index("Data_Value")]
    scale_val = scaler.scale_[num_cols.index("Data_Value")]
    pred_real = pred_scaled * scale_val + mean_val

    st.subheader(f"Predicted Obesity Prevalence: {pred_real:.2f}%")
