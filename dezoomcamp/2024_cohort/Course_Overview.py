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

add_page_title(layout="wide")

hide_pages(["Thank you"])

st.markdown("### 🏠 HOUSING PRICE PREDICTION AND VISUALIZATION APP")

#st.video("https://www.youtube.com/watch?v=AtRhA-NfS24")

# Streamlit form for user input
st.title('House Price Prediction')


# Function to convert object columns to integers
def convert_object_to_int(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')
    return df

# Convert object columns
data = convert_object_to_int(data)

# Convert float columns to integers, excluding 'Price (in rupees)'
float_cols = data.select_dtypes(include='float64').columns
float_cols = float_cols[float_cols != 'Price (in rupees)']
data[float_cols] = data[float_cols].fillna(0).astype('int32')

# Function to preprocess data
def preprocess_data(data):
    # Encode categorical columns
    for col, encoder in encoders.items():
        data[col] = encoder.transform(data[col])
    return data

# Preprocess the data
data = preprocess_data(data)

# Streamlit app
st.title('House Price Prediction App')

# Display the dataset
st.subheader('Preprocessed Dataset')
st.write(data)

# Sidebar for user input
st.sidebar.header('User Input')

# Function to get user input
def get_user_input():
    user_input = {}
    for col in data.columns:
        user_input[col] = st.sidebar.number_input(f'Enter {col}', value=0)
    return user_input

# Get user input
user_input = get_user_input()

# Function to predict house price
def predict_price(user_input):
    user_df = pd.DataFrame([user_input])
    user_df = preprocess_data(user_df)
    X = user_df.drop('Price (in rupees)', axis=1)
    model = joblib.load('model.pkl')
    prediction = model.predict(X)
    return prediction[0]

# Predict house price
predicted_price = predict_price(user_input)

# Display predicted price
st.subheader('Predicted Price')
st.write(f'The predicted price of the house is: {predicted_price}')


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
