import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# The pipeline's preprocessor was fit on label-encoded categorical columns
# (see prep.py), so raw text inputs from the widgets below must be mapped
# to the same integer codes before being passed to the model.
CATEGORY_MAPS = {
    "TypeofContact": {"Company Invited": 0, "Self Enquiry": 1},
    "Occupation": {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3},
    "Gender": {"Female": 0, "Male": 1},
    "ProductPitched": {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4},
    "MaritalStatus": {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3},
    "Designation": {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4},
}

# Streamlit UI
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to **purchase the tourism
package** being pitched, based on their profile and interaction details.
Please enter the customer details below to get a prediction.
""")

# User input
type_of_contact = st.selectbox("Type of Contact", list(CATEGORY_MAPS["TypeofContact"].keys()))
occupation = st.selectbox("Occupation", list(CATEGORY_MAPS["Occupation"].keys()))
gender = st.selectbox("Gender", list(CATEGORY_MAPS["Gender"].keys()))
product_pitched = st.selectbox("Product Pitched", list(CATEGORY_MAPS["ProductPitched"].keys()))
marital_status = st.selectbox("Marital Status", list(CATEGORY_MAPS["MaritalStatus"].keys()))
designation = st.selectbox("Designation", list(CATEGORY_MAPS["Designation"].keys()))

age = st.number_input("Age", min_value=18, max_value=100, value=35)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=120.0, value=15.0, step=1.0)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
number_of_followups = st.number_input("Number of Followups", min_value=0.0, max_value=10.0, value=3.0, step=1.0)
preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
number_of_trips = st.number_input("Number of Trips (per year)", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
passport = st.selectbox("Has Passport?", ["Yes", "No"])
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=5.0, value=0.0, step=1.0)
monthly_income = st.number_input("Monthly Income (USD)", min_value=0.0, max_value=100000.0, value=20000.0, step=100.0)

# Assemble input into DataFrame, mapping categorical text to the encoded
# integer codes the pipeline was trained on
input_data = pd.DataFrame([{
    'Age': age,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': number_of_trips,
    'Passport': 1 if passport == "Yes" else 0,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': 1 if own_car == "Yes" else 0,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'MonthlyIncome': monthly_income,
    'TypeofContact': CATEGORY_MAPS["TypeofContact"][type_of_contact],
    'Occupation': CATEGORY_MAPS["Occupation"][occupation],
    'Gender': CATEGORY_MAPS["Gender"][gender],
    'ProductPitched': CATEGORY_MAPS["ProductPitched"][product_pitched],
    'MaritalStatus': CATEGORY_MAPS["MaritalStatus"][marital_status],
    'Designation': CATEGORY_MAPS["Designation"][designation],
}])

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"Likely to **purchase** the package (confidence: {probability:.1%})")
    else:
        st.info(f"Unlikely to purchase the package (confidence: {1 - probability:.1%})")
