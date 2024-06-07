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


###########################



# Form for user input
st.header("Enter the Housing Features")

# Define input fields
location = st.selectbox("Location", sorted(data['location'].unique()))
transaction = st.selectbox("Transaction", sorted(data['Transaction'].unique()))
furnishing = st.selectbox("Furnishing", sorted(data['Furnishing'].unique()))
overlooking = st.selectbox("Overlooking", sorted(data['overlooking'].unique()))
ownership = st.selectbox("Ownership", sorted(data['Ownership'].unique()))
bathroom = st.number_input("Bathroom", min_value=1, step=1)
balcony = st.number_input("Balcony", min_value=0, step=1)
carpet_area = st.number_input("Carpet Area (in sqft)", min_value=0.0, step=0.1)
super_area = st.number_input("Super Area (in sqft)", min_value=0.0, step=0.1)
floor_level = st.number_input("Floor Level", min_value=0, step=1)
total_floors = st.number_input("Total Floors", min_value=0, step=1)
number_of_parking = st.number_input("Number of Parking", min_value=0, step=1)
parking_status = st.selectbox("Parking Status", sorted(data['Parking Status'].unique()))

# Prediction button
if st.button("Predict"):
    # Encode the input data
    input_data = pd.DataFrame({
        'location': [location],
        'Transaction': [transaction],
        'Furnishing': [furnishing],
        'overlooking': [overlooking],
        'Ownership': [ownership],
        'Bathroom': [bathroom],
        'Balcony': [balcony],
        'Carpet Area (in sqft)': [carpet_area],
        'Super Area (in sqft)': [super_area],
        'Floor Level': [floor_level],
        'Total Floors': [total_floors],
        'Number of Parking': [number_of_parking],
        'Parking Status': [parking_status]
    })

    for col, encoder in encoders.items():
        input_data[col] = encoder.transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.success(f"The predicted price is: {prediction:.2f} rupees")




############################



hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
