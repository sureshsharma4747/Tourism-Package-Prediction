import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load model
# --------------------------------------------------

MODEL_PATH = "model/tourism_package_model.pkl"

model = joblib.load(MODEL_PATH)

st.title("Tourism Package Prediction")

st.write(
    "Predict whether a customer is likely to purchase the tourism package."
)

# --------------------------------------------------
# Customer inputs
# --------------------------------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=1,
    max_value=60,
    value=10
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Free Lancer",
        "Small Business",
        "Large Business",
        "Government"
    ]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

number_of_followups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=20,
    value=3
)

product_pitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Deluxe",
        "Standard",
        "Super Deluxe",
        "King"
    ]
)

preferred_property_star = st.number_input(
    "Preferred Property Star",
    min_value=1,
    max_value=5,
    value=3
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Married",
        "Unmarried",
        "Single",
        "Divorced"
    ]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=30,
    value=2
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

pitch_satisfaction_score = st.number_input(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=25000
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age": [age],
        "TypeofContact": [type_of_contact],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "NumberOfFollowups": [number_of_followups],
        "ProductPitched": [product_pitched],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "Designation": [designation],
        "MonthlyIncome": [monthly_income]
    })

    # Make sure columns are in exactly the same order
    input_data = input_data[
        model.named_steps["preprocessor"].feature_names_in_
    ]

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.success(
            f"Customer is likely to purchase the package. "
            f"Probability: {probability:.2%}"
        )
    else:
        st.warning(
            f"Customer is unlikely to purchase the package. "
            f"Probability: {probability:.2%}"
        )