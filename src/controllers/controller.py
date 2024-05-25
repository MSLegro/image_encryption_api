import keras
import os
import json

def post_image_for_encryption(encoder, decoder, image):

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

    encoded_img = encoder.predict(img_array)
    decoded_img = decoder.predict(encoded_img)

    encoded_img_list = encoded_img.tolist()
    decoded_img_list = decoded_img.tolist()
    
    response= {
        "encoded": encoded_img_list,
        "decoded": decoded_img_list
    }

    return json.dumps(response)

