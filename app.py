import streamlit as st
import pickle
import numpy as np

# Load models
kidney_model = pickle.load(open("models/kidney_model.pkl", "rb"))
liver_model = pickle.load(open("models/liver_model.pkl", "rb"))
parkinsons_model = pickle.load(open("models/parkinsons_model.pkl", "rb"))

st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Multiple Disease Prediction System")

selected = st.selectbox(
    "Select Disease",
    (
        "Kidney Disease",
        "Liver Disease",
        "Parkinson's Disease"
    )
)

# =========================
# Liver Disease
# =========================
if selected == "Liver Disease":

    st.header("Liver Disease Prediction")

    age = st.number_input("Age", 0, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])

    total_bilirubin = st.number_input("Total Bilirubin", value=1.0)
    direct_bilirubin = st.number_input("Direct Bilirubin", value=0.2)
    alkaline = st.number_input("Alkaline Phosphotase", value=200)
    alamine = st.number_input("Alamine Aminotransferase", value=30)
    aspartate = st.number_input("Aspartate Aminotransferase", value=30)
    proteins = st.number_input("Total Proteins", value=6.5)
    albumin = st.number_input("Albumin", value=3.5)
    agr = st.number_input("Albumin and Globulin Ratio", value=1.0)

    gender = 1 if gender == "Male" else 0

    if st.button("Predict Liver Disease"):

        prediction = liver_model.predict([[
            age,
            gender,
            total_bilirubin,
            direct_bilirubin,
            alkaline,
            alamine,
            aspartate,
            proteins,
            albumin,
            agr
        ]])

        st.success(f"Prediction: {prediction[0]}")


# =========================
# Parkinson's Disease
# =========================
elif selected == "Parkinson's Disease":

    st.header("Parkinson's Disease Prediction")

    st.info("Enter 22 feature values from Parkinson's dataset")

    values = []

    feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
        'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
        'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    for feature in feature_names:
        values.append(st.number_input(feature, value=0.0))

    if st.button("Predict Parkinson's Disease"):

        prediction = parkinsons_model.predict([values])

        st.success(f"Prediction: {prediction[0]}")


# =========================
# Kidney Disease
# =========================
else:

    st.header("Kidney Disease Prediction")

    st.info(
        "Kidney dataset contains many features. "
        "For project demonstration, model loading is verified."
    )

    st.success("Kidney Model Loaded Successfully")