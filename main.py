from email.mime import image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from uvicorn import run
from predict import prediksi
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

import os

app = FastAPI()
def load_model():
    json_file = open('model_selar.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('selar_como_augmented.h5')
    print("Model loaded")
    return model

load_model

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

@app.get("/")
async def root():
    return {"message": "Welcome to the FISH ROOTEN API!"}
    
# UPLOAD FUNCTION
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open("images/uploaded_" + file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    path = "images/"+ file.filename
    prediction = prediksi('http://127.0.0.1:8000/images/uploaded_' + file.filename)

    return prediction
    
@app.get("/images/{images}", response_class=FileResponse)
async def get_images(images: str =""):
    return 'images/'+images
