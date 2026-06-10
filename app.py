from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model('alzheimer_cnn_model.keras')

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
    return {'message' : 'alzheimer prediction testing'}

@app.post('/predict')
def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert('RGB')

    processed_image = Preprocess(image)

    prediction = model.predict(processed_image)

    predicted_class = predic_class[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return JSONResponse(status_code=200, content= {"prediction": predicted_class, "confidence": confidence})
    

