# app.py

import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title of the app
st.title("Titanic Survival Prediction App")

# Input fields for user data
st.sidebar.header("Passenger Input Parameters")

def user_input_features():
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    age = st.sidebar.slider("Age", 0, 100, 29)  # default to 29, typical age in dataset
    sibsp = st.sidebar.slider("Number of Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.sidebar.slider("Number of Parents/Children Aboard", 0, 10, 0)
    embark = st.sidebar.selectbox("Port of Embarkation", ("0", "1", "2"))
    pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", (1, 2, 3))

    # Convert categorical inputs to numerical values
    sex = 1 if sex == "Male" else 0
    embark = int(embark)  # Convert embark to integer

    # Create input array including pclass
    data = np.array([[pclass, sex, age, sibsp, parch, embark]])
    return data

# Get user input
input_data = user_input_features()

# Display user input
st.subheader("Passenger Input Parameters")
st.write(f"Passenger Class: {input_data[0][0]}")
st.write(f"Sex: {'Male' if input_data[0][1] == 1 else 'Female'}")
st.write(f"Age: {input_data[0][2]}")
st.write(f"Siblings/Spouses Aboard: {input_data[0][3]}")
st.write(f"Parents/Children Aboard: {input_data[0][4]}")
st.write(f"Port of Embarkation: {input_data[0][5]}")

# Predict the survival probability
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # probability of survival

    # Display prediction
    if prediction[0] == 1:
        st.success(f"Survived! Probability: {probability:.2f}")
    else:
        st.error(f"Did not survive. Probability: {probability:.2f}")
