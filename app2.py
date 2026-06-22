import joblib
import pandas as pd
import streamlit as st

# Load the saved end-to-end processing pipeline model
model_filename = "pipeline_model.pkl"
model = joblib.load(model_filename)

# Define the web interface header text layout
st.title("Customer Churn Prediction Web App")
st.write(
    "Enter customer health metrics below to predict risk using the Scikit-Learn Pipeline."
)

# Create input fields inside the browser interface for the user
age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
monthly_spend = st.number_input(
    "Monthly Spend ($)", min_value=10.0, max_value=500.0, value=75.0
)
subscription_type = st.selectbox(
    "Subscription Plan Type", ["Basic", "Standard", "Premium"]
)
gender = st.selectbox("Customer Gender", ["Male", "Female"])

# Create a trigger button to execute predictions
if st.button("Predict Churn Status"):
    # Group the browser input data into a structural DataFrame match
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Monthly_Spend": [monthly_spend],
            "Subscription_Type": [subscription_type],
            "Gender": [gender],
        }
    )

    # Pass the raw messy data directly into the pipeline
    # The pipeline automatically runs imputation and encoding internally
    prediction = model.predict(input_data)
    prediction_probability = model.predict_proba(input_data)[0][1]

    # Show the results back onto the web page interface
    st.subheader("Prediction Results:")
    if prediction[0] == 1:
        st.error(
            f"High Risk: Customer is likely to churn! (Probability: {prediction_probability:.2%})"
        )
    else:
        st.success(
            f"Low Risk: Customer is likely to stay active. (Probability of churn: {prediction_probability:.2%})"
        )