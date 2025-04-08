import streamlit as st
import joblib
import numpy as np

# Load model and encoder
model = joblib.load("lr_model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("🎓 Student Final Score Predictor")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])
extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
career_aspiration = st.selectbox("Career Aspiration", ["Engineering", "Medicine", "Commerce", "Arts", "Other"])

absence_days = st.number_input("Absence Days", min_value=0, max_value=30)
weekly_self_study_hours = st.number_input("Weekly Self-Study Hours", min_value=0, max_value=40)
math_score = st.slider("Math Score", 0, 100)
history_score = st.slider("History Score", 0, 100)
physics_score = st.slider("Physics Score", 0, 100)
chemistry_score = st.slider("Chemistry Score", 0, 100)
biology_score = st.slider("Biology Score", 0, 100)
english_score = st.slider("English Score", 0, 100)
geography_score = st.slider("Geography Score", 0, 100)

if st.button("Predict Final Score"):
    # Step 1: Categorical input
    categorical_input = [[
        gender,
        part_time_job,
        extracurricular_activities,
        career_aspiration
    ]]

    # Step 2: Numerical input
    numerical_input = [[
        absence_days,
        weekly_self_study_hours,
        math_score,
        history_score,
        physics_score,
        chemistry_score,
        biology_score,
        english_score,
        geography_score
    ]]

    # Step 3: Encode + Stack
    encoded_categorical = encoder.transform(categorical_input)
    final_input = np.hstack((encoded_categorical, numerical_input))

    # Step 4: Predict
    prediction = model.predict(final_input)
    st.success(f"🎯 Predicted Final Score: {round(prediction[0], 2)}")
