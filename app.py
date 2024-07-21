import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load models
with open('regression_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)
with open('classification_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Streamlit app
st.title('Industrial Copper Modeling')

task = st.selectbox('Select Task', ['Regression', 'Classification'])

if task == 'Regression':
    st.header('Predict Selling Price')
    input_data = {
        'quantity_tons': st.number_input('Quantity (tons)', value=0.0),
        'customer': st.text_input('Customer'),
        'country': st.text_input('Country'),
        'item_type': st.text_input('Item Type'),
        'application': st.text_input('Application'),
        'thickness': st.number_input('Thickness', value=0.0),
        'width': st.number_input('Width', value=0.0),
        'material_ref': st.text_input('Material Reference'),
        'product_ref': st.text_input('Product Reference'),
        'delivery_date': st.date_input('Delivery Date')
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input
    input_df['customer'] = label_encoder.transform([input_df['customer'][0]])
    input_df['country'] = label_encoder.transform([input_df['country'][0]])
    input_df['material_ref'] = input_df['material_ref'].replace('Unknown', np.nan)
    input_df = input_df.fillna('Unknown')
    
    input_df_scaled = scaler.transform(input_df)

    if st.button('Predict Selling Price'):
        prediction = reg_model.predict(input_df_scaled)
        st.write(f'Predicted Selling Price: ${prediction[0]:.2f}')

elif task == 'Classification':
    st.header('Classify Status')
    input_data = {
        'quantity_tons': st.number_input('Quantity (tons)', value=0.0),
        'customer': st.text_input('Customer'),
        'country': st.text_input('Country'),
        'item_type': st.text_input('Item Type'),
        'application': st.text_input('Application'),
        'thickness': st.number_input('Thickness', value=0.0),
        'width': st.number_input('Width', value=0.0),
        'material_ref': st.text_input('Material Reference'),
        'product_ref': st.text_input('Product Reference'),
        'delivery_date': st.date_input('Delivery Date')
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input
    input_df['customer'] = label_encoder.transform([input_df['customer'][0]])
    input_df['country'] = label_encoder.transform([input_df['country'][0]])
    input_df['material_ref'] = input_df['material_ref'].replace('Unknown', np.nan)
    input_df = input_df.fillna('Unknown')
    
    input_df_scaled = scaler.transform(input_df)

    if st.button('Classify Status'):
        prediction = clf_model.predict(input_df_scaled)
        status = 'WON' if prediction[0] == 1 else 'LOST'
        st.write(f'Predicted Status: {status}')
