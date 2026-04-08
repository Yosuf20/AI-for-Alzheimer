import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("alzheimer_cnn_model.keras")
st.title("Alzeimer Class Predictions")


file_image = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])


with st.expander("Classes"):
    st.markdown("""
    1. Non Demented
    2. Very Mild Demented
    3. Mild Demented
    4. Moderate Demented
    """)

if file_image is not None:
    image = Image.open(file_image).convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image)
    image = image/255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    classes = [
        "Mild Demented",
        "Moderate Demented",
        "Healthy",
        "Very Mild Demented"
    ]

    result = prediction
    classify = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    st.success(f"Your Class: {classify}")
    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}")


test_df = pd.read_parquet("DataSet/test.parquet")

