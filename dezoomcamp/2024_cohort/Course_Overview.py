import streamlit as st
from st_pages import add_page_title, hide_pages
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load the model
model = joblib.load('files/house/model.pkl')
encoders = joblib.load('files/house/encoders.pkl')


# Load the new dataset
data = pd.read_csv('files/house/updated_house_info.csv')  
df = pd.read_csv('files/house/updated_house_info.csv')  


add_page_title(layout="wide")

hide_pages(["Thank you"])

st.markdown("### 🏠 HOUSING PRICE PREDICTION AND VISUALIZATION APP")

#st.video("https://www.youtube.com/watch?v=AtRhA-NfS24")

# Streamlit form for user input
st.title('House Price Prediction')

def preprocess_input(data):
    # Your preprocessing steps here
    data['Parking Status'] = data['Parking Status'].str.replace('Covered,', 'Covered')
    for col, encoder in encoders.items():
        data[col] = encoder.transform(data[col])
    data.drop(columns=['Status'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def predict_price(input_data):
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return prediction
  
  
  
st.write('This app predicts the price of a house based on input features.')

st.header('Input Features')
  
input_data = pd.DataFrame({
  'Price (in rupees)': [0],
  'location': [''],
  'Status': [''],
  'Transaction': [''],
  'Furnishing': [''],
  'overlooking': [''],
  'Bathroom': [0],
  'Balcony': [0],
  'Ownership': [''],
  'Carpet Area (in sqft)': [0],
  'Super Area (in sqft)': [0],
  'Floor Level': [0],
  'Total Floors': [0],
  'Number of Parking': [0],
  'Parking Status': ['']
})


input_data['location'] = st.selectbox('Location', sorted(df['location'].unique()))
input_data['Transaction'] = st.selectbox('Transaction', sorted(df['Transaction'].unique()))
input_data['Furnishing'] = st.selectbox('Furnishing', sorted(df['Furnishing'].unique()))
input_data['overlooking'] = st.selectbox('Overlooking', sorted(df['overlooking'].unique()))
input_data['Ownership'] = st.selectbox('Ownership', sorted(df['Ownership'].unique()))
input_data['Bathroom'] = st.number_input('Number of Bathrooms', min_value=0)
input_data['Balcony'] = st.number_input('Number of Balconies', min_value=0)
input_data['Carpet Area (in sqft)'] = st.number_input('Carpet Area (in sqft)', min_value=0)
input_data['Super Area (in sqft)'] = st.number_input('Super Area (in sqft)', min_value=0)
input_data['Floor Level'] = st.number_input('Floor Level', min_value=0)
input_data['Total Floors'] = st.number_input('Total Floors', min_value=0)
input_data['Number of Parking'] = st.number_input('Number of Parking', min_value=0)
input_data['Parking Status'] = st.radio('Parking Status', ['Covered', 'Open'])


if st.button('Predict Price'):
  prediction = predict_price(input_data)
  st.write('Predicted Price:', prediction)



hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
