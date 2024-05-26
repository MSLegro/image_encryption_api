from fastapi import APIRouter, UploadFile
from src.controllers import controller

BASE_URL = '/api/v1'

router = APIRouter()

def create_api_route(encoder, decoder):
  @router.post(f"{BASE_URL}/upload-image-encoded")
  async def post_image_for_encoded(image: UploadFile):
    return await controller.post_image_for_encoded(encoder, image)
  
  @router.post(f"{BASE_URL}/upload-image-decoded")
  async def post_image_for_decoded(image: UploadFile):
    return await controller.post_image_for_decoded(decoder, image)
  
  @router.get(f"{BASE_URL}/download-image")
  async def get_encoded_image():
    return await controller.get_encoded_image()
  return router