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

dependant = st.selectbox('Dependants',
                      ('Yes', 'No'))
Tenure = st.number_input('Tenure')
online_security = st.selectbox('Online Security',
                      ('Yes', 'No'))
online_backup = st.selectbox('Online Backup',
                      ('Yes', 'No'))
internet_service = st.selectbox('Internet Service',
                      ('DSL', 'Fiber optic', 'No'))
device_protection = st.selectbox('Device Protection',
                      ('Yes', 'No'))
tech_support = st.selectbox('Tech Support',
                      ('Yes', 'No'))
contract = st.selectbox('Contract',
                      ('Month-to-month', 'Two year', 'One year'))
paperless_billing = st.selectbox('Paperless Billing',
                      ('Yes', 'No'))
monthly_charges = st.number_input('Monthly Charges')

if st.button('Model Predict'):
    data = pd.DataFrame({
        'Dependants': [dependant],
        'Tenure': [Tenure],
        'Online Security': [online_security],
        'Online Backup': [online_backup],
        'Internet Service': [internet_service],
        'Device Protection': [device_protection],
        'Tech Support': [tech_support],
        'Contract': [contract],
        'Paperless Billing': [paperless_billing],
        'Monthly Charges': [monthly_charges]
    })
    result = st.session_state['model'].predict(data)
    if result[0] == 0:
        st.write(f'Not Churn')
    else:
        st.write(f'Churn')

else:
    st.write('Please input the feature above to start the prediction') 