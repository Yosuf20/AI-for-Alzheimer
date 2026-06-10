import streamlit as st
import requests

st.title("Alzheimer's Disease Classification")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file)

    if st.button("Predict"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files=files
        )

        if response.status_code == 200:

            result = response.json()

            st.success(
                f"Prediction: {result['prediction']}"
            )

            st.write(
                f"Confidence: {result['confidence']:.2%}"
            )

        else:
            st.error("Prediction failed")