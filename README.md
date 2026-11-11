# Titanic Survival Predictor

A simple machine learning web application that predicts whether a Titanic passenger may have survived.

# Features

The application uses a trained **Decision Tree Classifier** and takes the following inputs:

- Passenger Class
- Sex
- Age
- Fare

#Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Joblib

#How to Run

1. Clone this repository.
2. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
python -m streamlit run app.py
```

# About the Model

The model was trained on the Titanic dataset to predict passenger survival. It uses passenger class, sex, age, and fare as input features.

> **Note:** This is an educational machine learning project. The predictions are based on historical Titanic data and should not be considered real-world advice.
