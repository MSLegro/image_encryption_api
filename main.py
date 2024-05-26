from fastapi import FastAPI
import keras
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from src.routes.route import create_api_route

app = FastAPI()

origins: list[str] = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


decoder_model_path = os.path.join('src', 'models', 'decoder1.h5')
encoder_model_path = os.path.join('src', 'models', 'encoder1.h5')

# Verifica si el archivo existe
if not os.path.exists(decoder_model_path or encoder_model_path):
    raise FileNotFoundError(f"No se encontró el modelo en la ruta")

# Cargar el encoder y el decoder
encoder = keras.models.load_model('src/models/encoder1.h5')
decoder = keras.models.load_model('src/models/decoder1.h5')

app.include_router(create_api_route(encoder, decoder))

@app.get('/')
def read_root():
    return {"Welcome": "Hello World"}

if __name__ == "__main__":
    uvicorn.run("main:app",
                host='0.0.0.0', port=8080, reload=True)