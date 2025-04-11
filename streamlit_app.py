import streamlit as st 
import pickle
import numpy as np

# Load a model
model = pickle.load(open('ann_model.pkl', 'rb'))

st.title("Renewable Energy Prediction")

# Example inputs
input_1 = st.number_input("Feature 1")
input_2 = st.number_input("Feature 2")

if st.button("Predict"):
    data = np.array([[input_1, input_2]])
    prediction = model.predict(data)
    st.success(f"Predicted Output: {prediction}")
