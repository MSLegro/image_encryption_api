from fastapi import APIRouter, UploadFile
from src.controllers import controller

BASE_URL = '/api/v1'

router = APIRouter()

def create_api_route(encoder, decoder):
  @router.post(f"{BASE_URL}/upload-image")
  def post_image_for_encryption(image: UploadFile):
    return controller.post_image_for_encryption(encoder,decoder,image)
  
  return router