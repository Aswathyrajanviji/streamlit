import pickle
import pandas as pd
import numpy as np
import streamlit as st
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except:
    scaler = None
def preprocess_and_predict(features):
    # convert to dataframe
    input_data=pd.DataFrame([features])
    # get required columns
    required_columns = model.feature_names_in_
    # add missing columns
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0  
    input_data = input_data[required_columns]
    if scaler is not None:
        input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:,1]
    return prediction[0], probability[0]





st.title("Heart Disease Prediction App")
age=st.number_input("Age",1,120,50)
sex=st.selectbox("Sex",["Male","Female"])
sex=1 if sex=="Male" else 0
cp=st.selectbox("chest pain type",[0,1,2,3])
tres=st.number_input("resting blood pressure",80,200,120)
chol=st.number_input("cholestoral",100,600,200)
fbs=st.selectbox('fasing blood sugar',[0,1])
restecg=st.selectbox('resting  ECG',[0,1,2])
thalach=st.number_input("maximum heart rate ",60,220,150)
exang=st.selectbox('exercise induced angina',[0,1])
oldpeak=st.number_input("st depression",0.0,6.0,1.0, step=0.1)
slope=st.selectbox('slope',[0,1,2])
ca=st.selectbox('number of major vessels',[0,1,2,3,4])
thal=st.selectbox('thalassemia',[0,1,2,3])

features={
    'age': age,
    'sex':sex,
    'cp':cp,
    'trestbps':tres,
    'chol':chol,
    'fbs':fbs,
    'restecg':restecg,
    'thalach':thalach,
    'exang':exang,
    'oldpeak':oldpeak,
    'slope':slope,
    'ca':ca,
    'thal':thal
}

if st.button("Predict"):
    prediction, probability = preprocess_and_predict(features)
    if prediction == 1:
        st.error(f"high chnace of heart disease(probalility:{probability:.2f})")
    else:
        st.success(f"low chnace of heart disease(probalility:{probability:.2f})")