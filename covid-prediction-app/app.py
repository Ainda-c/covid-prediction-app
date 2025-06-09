import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="COVID-19 Outbreak Prediction",
    page_icon="🦠",
    layout="wide"
)

# Title and description
st.title("COVID-19 Outbreak Prediction Model")
st.markdown("""
This application uses machine learning to predict COVID-19 outbreaks based on historical data.
The model helps address SDG 3 (Good Health and Well-being) by providing insights for public health planning.
""")

# Sidebar
st.sidebar.header("Model Parameters")

# Load the model and scaler (you'll need to save these after training)
@st.cache_resource
def load_model():
    try:
        with open('covid_new_cases_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        st.error("Model files not found. Please train the model first.")
        return None, None

# Main content
st.header("Model Predictions")

# Input features
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Parameters")
    new_cases_lag_1 = st.number_input("New Cases (Previous Day)", min_value=0)
    new_cases_lag_7 = st.number_input("New Cases (7 Days Ago)", min_value=0)
    vaccination_rate = st.slider("Vaccination Rate", 0.0, 1.0, 0.5)
    hospital_beds = st.number_input("Hospital Beds per Thousand", min_value=0.0)

with col2:
    st.subheader("Additional Parameters")
    mortality_rate = st.slider("Mortality Rate", 0.0, 0.1, 0.02)
    cases_per_population = st.slider("Cases per Population", 0.0, 0.1, 0.01)
    life_expectancy = st.number_input("Life Expectancy", min_value=0.0, max_value=100.0, value=75.0)

# Create feature array
features = np.array([[
    new_cases_lag_1,
    new_cases_lag_7,
    vaccination_rate,
    mortality_rate,
    cases_per_population,
    hospital_beds,
    life_expectancy
]])

# Make prediction
if st.button("Predict"):
    model, scaler = load_model()
    if model is not None and scaler is not None:
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Display prediction
        st.success(f"Predicted New Cases: {int(prediction)}")
        
        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Predicted New Cases"},
            gauge={'axis': {'range': [0, max(prediction * 2, 1000)]}}
        ))
        st.plotly_chart(fig)

# Ethical Considerations
st.header("Ethical Considerations")
st.markdown("""
### Data Quality and Bias
- The model's predictions may be affected by varying data quality across regions
- Some countries may have better reporting systems than others
- Testing capacity varies significantly between regions

### Fairness and Sustainability
- The model should be regularly retrained with new data
- Performance should be monitored across different regions
- Regular validation against ground truth data is essential
""")

# Footer
st.markdown("---")
st.markdown("""
This application is part of a project addressing SDG 3 (Good Health and Well-being).
The model is designed to help public health officials make informed decisions about COVID-19 response.
""")