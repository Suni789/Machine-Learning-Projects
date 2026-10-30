import streamlit as st
import pandas as pd
import joblib

model = joblib.load("SVM_Heart_Disease.pkl")
scaler = joblib.load("scaler.pkl")
Expected_columns = joblib.load("columns.pkl")

st.title("Heart Stroke Prediction by Sara")
st.markdown("provide the following details")
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex",['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ["typical angina", "asymptomatic", "non-anginal", "atypical angina"])
trestbps = st.number_input("Trest Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.selectbox("Resting ECG", ["lv hypertrophy", "normal", "st-t abnormality"])
thalch = st.slider("Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'TrestBP': trestbps,
        'Cholesterol': chol,
        'FastingBS': fbs,
        'Heart Rate': thalch,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'Chest Pain Type_'+ chest_pain: 1,
        'RestingECG_'+ restecg: 1,
        'Exercise Angina_'+ exang: 1,
        'ST_Slope_' + slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in Expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[Expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]        
    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Rist of Heart Disease")


