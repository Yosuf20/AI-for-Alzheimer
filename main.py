from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import io
import os

app = FastAPI()



model = tf.keras.models.load_model('model/alzheimer_cnn_model.keras')

STATIC_DIR = os.path.join(os.path.dirname(__file__), '.', "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

predic_class = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

def Preprocess(image):
    image = image.resize((128, 128))
    image = np.array(image)
    image = image/255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get('/')
def show():
    return FileResponse("static/index.html")

@app.post('/predict')
def predict(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    age: Optional[str] = Form(None),
    dob : Optional[str] = Form(None),
    ):

    try:
        image = Image.open(file.file).convert('RGB')
    except Exception:
        raise HTTPException(status_code=422, detail={"message" : "Could not read the image"})
    

    processed_image = Preprocess(image)

    try:
        prediction = model.predict(processed_image)
        predicted_class = predic_class[np.argmax(prediction)]
        all_confidences = {cls: float(prediction[0][i]) for i, cls in enumerate(predic_class)}
        confidence = float(np.max(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Inference Failed: {str(e)}")

    return JSONResponse(status_code=200, content={
        "prediction":       predicted_class,
        "confidence":       confidence,
        "all_confidences":  all_confidences,
        "patient_name":     name,
        "patient_age":      age,
        "patient_dob":      dob,
    })
    

