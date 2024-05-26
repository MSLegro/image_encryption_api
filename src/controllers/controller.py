from click import File
from fastapi import UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import keras
import os
import numpy as np
import io
from PIL import Image

def procesing_image_for_encoded(image):
    with open("temp_image.jpg", "wb") as buffer:
        buffer.write(image.file.read())
    # Cargar la imagen
    img = keras.preprocessing.image.load_img("temp_image.jpg", color_mode='grayscale', target_size=(28, 28))
    # Convertir la imagen a un array numpy
    img_array = keras.preprocessing.image.img_to_array(img)
    # Normalizar la imagen
    img_array = img_array.astype('float32') / 255
    # Aplanar la imagen (28x28 a 784)
    img_array = img_array.reshape((1, 28*28))
    os.remove("temp_image.jpg")
    return img_array


async def post_image_for_encoded(encoder, image):

    img_array = procesing_image_for_encoded(image)
    encoded_img = encoder.predict(img_array)
    np.save('encoded_img.npy',encoded_img)

    data_array = np.array(encoded_img)

    # Normaliza los datos para que estén en el rango [0, 255]
    data_array_normalized = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array)) * 255

    # Convierte el array normalizado a tipo uint8
    data_array_uint8 = data_array_normalized.astype(np.uint8)

    # Convierte el array a una imagen
    image = Image.fromarray(data_array_uint8)

    # Guarda la imagen en un objeto BytesIO
    img_io = io.BytesIO()
    image.save(img_io,'PNG')
    img_io.seek(0)

    return StreamingResponse(img_io, media_type="image/png")


async def get_encoded_image():
    try:
        return FileResponse('encoded_img.npy', media_type='application/octet-stream', filename='encoded_img.npy')
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


async def post_image_for_decoded(decoder, image: UploadFile = File(...)):

    with open("encoded_img.npy", "wb") as buffer:
        buffer.write(image.file.read())

    encoded_img = np.load("encoded_img.npy")

    decoded_img = decoder.predict(encoded_img)

    data_array = np.array(decoded_img)

    # Normaliza los datos para que estén en el rango [0, 255]
    data_array_normalized = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array)) * 255

    # Convierte el array normalizado a tipo uint8
    data_array_uint8 = data_array_normalized.astype(np.uint8)

    # Convierte el array a una imagen
    image = Image.fromarray(data_array_uint8)

    # Guarda la imagen en un objeto BytesIO
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return StreamingResponse(img_io, media_type="image/png")



