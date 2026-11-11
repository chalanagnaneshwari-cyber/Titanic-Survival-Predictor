import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢"
)

# Title
st.title("🚢 Titanic Survival Predictor")
st.write("Predict whether a Titanic passenger may have survived.")

# Feature order - MUST match the training data
FEATURES = ['pclass', 'sex', 'age', 'fare']


# Load the model only once
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


# Check whether model file exists
if not os.path.exists("model.joblib"):
    st.error("⚠️ Model file is missing. Please add model.joblib to this folder.")
    st.stop()

try:
    model = load_model()
except Exception:
    st.error("⚠️ Unable to load the model. Please check the model file.")
    st.stop()


# User inputs
st.subheader("Enter Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    options=[1, 2, 3],
    index=2
)

sex_text = st.radio(
    "Sex",
    options=["Female", "Male"]
)

# Convert sex to the same format used during training
sex = 0 if sex_text == "Female" else 1

age = st.slider(
    "Age",
    min_value=0,
    max_value=80,
    value=25
)

fare = st.slider(
    "Fare",
    min_value=0.0,
    max_value=550.0,
    value=32.0
)


# Prediction button
if st.button("Predict Survival"):

    try:
        # Validate input
        if age < 0 or fare < 0:
            st.warning("Please enter valid values.")
        else:
            # Create DataFrame in EXACT training feature order
            input_data = pd.DataFrame(
                [[pclass, sex, age, fare]],
                columns=FEATURES
            )

            # Get prediction and probability
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            # Find survival probability
            survival_probability = probability[1] * 100

            st.divider()

            # Display result
            if prediction == 1:
                st.success("🎉 Prediction: Survived")
            else:
                st.error("Prediction: Did Not Survive")

            # Display probability
            st.metric(
                label="Predicted Survival Probability",
                value=f"{survival_probability:.1f}%"
            )

    except Exception:
        st.error("⚠️ Something went wrong while making the prediction. Please check your inputs.")


# Information about the project
st.divider()

st.caption(
    "This app uses a Decision Tree model trained on the Titanic dataset "
    "using passenger class, sex, age, and fare. "
    "This is an educational machine learning project and not professional advice."
)