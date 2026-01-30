import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="pkansal83/mlops_tourism_project", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of customer purchasing the newly introduced Wellness Tourism Package before contacting them.
Please enter the costumer and interaction data below to get a prediction.
""")

# User input
age = st.number_input("Age", min_value=18, max_value=70, value=21, step=1)
TypeOfContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.radio("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of people accompnying", min_value=0, max_value=10, value=1, step=1)
PreferredPropertyStar = st.number_input("Preferred hotel rating", min_value=1, max_value=5, value=3, step=1)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Unmarried", "Married", "Divorced"])
NumberOfTrips = st.number_input("Average number of trips annually", min_value=1, max_value=50, value=3, step=1)
passport = 1 if st.radio("Hold valid passport:", ["No", "Yes"]) == "Yes" else 0
OwnCar = 1 if st.radio("Owns a Car:", ["No", "Yes"]) == "Yes" else 0
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying", min_value=0, max_value=5, value=1, step=1)
Designation = st.selectbox("Designation in current organization", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Gross monthly income", min_value=1, value=25000)
PitchSatisfactionScore = st.number_input("Satisfaction score with the sales pitch", min_value=1, max_value=5, value=5, step=1)
ProductPitched = st.selectbox("Type of product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("Total number of follow-ups by the salesperson after the sales pitch", min_value=1, max_value=10, value=2, step=1)
DurationOfPitch = st.number_input("Duration of the sales pitch delivered", min_value=5, value=15)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': TypeOfContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    result = "Purchase" if prediction == 1 else "Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts that the customer will **{result}** new Wellness Tourism Package")
