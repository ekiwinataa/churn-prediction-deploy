import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon=":iphone:"
)

st.title('Telco Customer Churn Prediction Model Deployment')

if 'model' not in st.session_state:
    model = pickle.load(open('model/model.sav', 'rb'))
    st.session_state['model'] = model

Dependents = st.selectbox('Dependents',
                      ('Yes', 'No'))
tenure = st.number_input('tenure')
OnlineSecurity = st.selectbox('OnlineSecurity',
                      ('Yes', 'No'))
OnlineBackup = st.selectbox('OnlineBackup',
                      ('Yes', 'No'))
InternetService = st.selectbox('InternetService',
                      ('DSL', 'Fiber optic', 'No'))
DeviceProtection = st.selectbox('DeviceProtection',
                      ('Yes', 'No'))
TechSupport = st.selectbox('TechSupport',
                      ('Yes', 'No'))
Contract = st.selectbox('Contract',
                      ('Month-to-month', 'Two year', 'One year'))
PaperlessBilling = st.selectbox('PaperlessBilling',
                      ('Yes', 'No'))
MonthlyCharges = st.number_input('MonthlyCharges')

if st.button('Model Predict'):
    data = pd.DataFrame({
        'Dependants': [Dependents],
        'tenure': [tenure],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'InternetService': [InternetService],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'MonthlyCharges': [MonthlyCharges]
    })
    result = st.session_state['model'].predict(data)
    if result[0] == 0:
        st.write(f'Not Churn')
    else:
        st.write(f'Churn')

else:
    st.write('Please input the feature above to start the prediction') 
