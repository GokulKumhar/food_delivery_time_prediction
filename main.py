import streamlit as st
import pandas as pd
import pickle

# =========================
# Load Model
# =========================

with open('best_random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# =========================
# Streamlit Page Config
# =========================

st.set_page_config(
    page_title='Food Delivery Time Predictor',
    page_icon='🍔',
    layout='wide'
)

# =========================
# Title
# =========================

st.title('🍔 Food Delivery Time Prediction App')
st.write('Predict food delivery time using Machine Learning 🚀')

st.markdown('---')

# =========================
# Create Columns
# =========================

col1, col2 = st.columns(2)

# =========================
# Input Fields
# =========================

with col1:

    distance_km = st.number_input(
        'Distance (KM)',
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.1
    )

    weather = st.selectbox(
        'Weather',
        ['Clear', 'Foggy', 'Rainy', 'Windy']
    )

    traffic_level = st.selectbox(
        'Traffic Level',
        ['Low', 'Medium', 'High']
    )

    time_of_day = st.selectbox(
        'Time Of Day',
        ['Morning', 'Afternoon', 'Evening', 'Night']
    )

with col2:

    vehicle_type = st.selectbox(
        'Vehicle Type',
        ['Bike', 'Scooter']
    )

    preparation_time = st.number_input(
        'Preparation Time (Minutes)',
        min_value=0,
        max_value=120,
        value=15
    )

    courier_experience = st.number_input(
        'Courier Experience (Years)',
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.5
    )

# =========================
# Manual Encoding
# =========================

weather_mapping = {
    'Clear': 0,
    'Foggy': 1,
    'Rainy': 2,
    'Windy': 3
}

traffic_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}

time_mapping = {
    'Morning': 0,
    'Afternoon': 1,
    'Evening': 2,
    'Night': 3
}

vehicle_mapping = {
    'Bike': 0,
    'Scooter': 1
}

weather_encoded = weather_mapping[weather]
traffic_encoded = traffic_mapping[traffic_level]
time_encoded = time_mapping[time_of_day]
vehicle_encoded = vehicle_mapping[vehicle_type]

# =========================
# Create Input DataFrame
# =========================

input_data = pd.DataFrame({

    'Distance_km': [distance_km],
    'Weather': [weather_encoded],
    'Traffic_Level': [traffic_encoded],
    'Time_of_Day': [time_encoded],
    'Vehicle_Type': [vehicle_encoded],
    'Preparation_Time_min': [preparation_time],
    'Courier_Experience_yrs': [courier_experience]

})

# =========================
# Prediction Button
# =========================

if st.button('Predict Delivery Time'):

    prediction = model.predict(input_data)[0]

    st.success(
        f'⏰ Estimated Delivery Time: {prediction:.2f} Minutes'
    )

    st.balloons()

# =========================
# Footer
# =========================

st.markdown('---')

st.caption(
    'Built using Streamlit & Random Forest Machine Learning Model 🚀'
)