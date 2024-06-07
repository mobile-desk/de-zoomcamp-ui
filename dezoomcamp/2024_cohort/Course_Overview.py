import streamlit as st
from st_pages import add_page_title, hide_pages
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

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


# Transform function
def transform_input(input_data, encoders):
    for col, encoder in encoders.items():
        if col in input_data:
            input_data[col] = encoder.transform([input_data[col]])[0]
        else:
            st.error(f"Key {col} not found in input data")
    return input_data

# User input form
location = st.selectbox('Location', encoders['location'].classes_)
transaction = st.selectbox('Transaction', encoders['Transaction'].classes_)
furnishing = st.selectbox('Furnishing', encoders['Furnishing'].classes_)
ownership = st.selectbox('Ownership', encoders['Ownership'].classes_)
parking_status = st.selectbox('Parking Status', encoders['Parking Status'].classes_)
overlooking = st.selectbox('Overlooking', encoders['overlooking'].classes_)

carpet_area = st.number_input('Carpet Area (in sqft)', min_value=0.0)
super_area = st.number_input('Super Area (in sqft)', min_value=0.0)
floor_level = st.number_input('Floor Level', min_value=0)
total_floors = st.number_input('Total Floors', min_value=0)
number_of_parking = st.number_input('Number of Parking', min_value=0)

# Handle submit button
if st.button('Predict'):
    # Collect user input into a dictionary
    user_input = {
        'location': location,
        'Transaction': transaction,
        'Furnishing': furnishing,
        'Ownership': ownership,
        'Parking Status': parking_status,
        'overlooking': overlooking,
        'Carpet Area (in sqft)': carpet_area,
        'Super Area (in sqft)': super_area,
        'Floor Level': floor_level,
        'Total Floors': total_floors,
        'Number of Parking': number_of_parking
    }
    
    # Transform the input data using the saved encoders
    transformed_input = transform_input(user_input, encoders)
    
    # Convert to numpy array and reshape
    input_data = np.array(list(transformed_input.values())).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    st.write(f'Predicted Price: {prediction} rupees')

# Example of a simple chart
if st.button('Show Distribution of Prices'):
    data = pd.read_csv('files/house/updated_house_info.csv')  # Adjust path as needed
    fig = px.histogram(data, x='Price (in rupees)', nbins=50, title='Distribution of House Prices')
    st.plotly_chart(fig)

    # Example of a scatter plot
    fig = px.scatter(data, x='Carpet Area (in sqft)', y='Price (in rupees)', title='Carpet Area vs Price')
    st.plotly_chart(fig)




hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
