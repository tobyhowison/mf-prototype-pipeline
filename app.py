import streamlit as st
import pickle
import pandas as pd
from config_params import MODEL_SAVE_PATH

# Load pretrained model
model = pickle.load(open(MODEL_SAVE_PATH , 'rb'))

# Title
st.title('Heart Disease Prediction App')

# Input fields
age = st.number_input('age', min_value=0, max_value=100, value=50)
sex = st.selectbox('sex', [0, 1])
chest_pain_type = st.selectbox('chest pain type', [0, 1, 2, 3])
resting_bp = st.number_input('resting blood pressure', min_value=0, max_value=250, value=125)
chol = st.number_input('chol', min_value=0, max_value=600, value=300)
fasting_blood_sugar = st.selectbox('fasting Blood Sugar', [0, 1])
resting_ecg = st.selectbox('resting ecg', [0, 1])
max_heart_rate = st.number_input('max heart rate', min_value=0, max_value=250, value=125)
exang = st.selectbox('exang', [0, 1])
oldpeak = st.number_input('oldpeak', min_value=0.0, max_value=10.0, value=5.0)
slope = st.selectbox('slope', [0, 1, 2])
number_vessels_fluorosopy = st.number_input('number vessels flourosopy', min_value=0, max_value=3, value=0)
thal = st.selectbox('thal', [1, 2, 3])

# Convert input values into a DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'chest pain type': [chest_pain_type],
    'resting blood pressure': [resting_bp],
    'chol': [chol],
    'fasting blood sugar': [fasting_blood_sugar],
    'resting ECG': [resting_ecg],
    'max heart rate': [max_heart_rate],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'number vessels flourosopy': [number_vessels_fluorosopy],
    'thal': [thal]
})

if st.button('Evaluate'):

    # Make prediction
    prediction = model.predict(input_data)

    # Display
    if prediction == 1:
        st.write(f'Prediction {prediction} (patient is likely to have heart disease.)')
    else:
        st.write(f'Prediction {prediction} (patient is unlikely to have heart disease.)')