from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from models import *
from middleware import executable_time
from functions import ImagePigimageBase64Generate, imageBase64Generate, imageGenerate, resolve_token, fetchImagePig

import logging
import tempfile
import os 
import io
import base64
import shutil
import uuid
import asyncio

app = FastAPI()

load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
if not HUGGING_FACE_API_KEY:
    raise EnvironmentError("Please check environment variables or .env file.")
IMAGE_PIG = os.getenv("IMAGE_PIG")
if not IMAGE_PIG:
    raise EnvironmentError("Please check environment variables or .env file.")

app.mount("/images", StaticFiles(directory="temp_images"), name="images")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def delete_file_later(path: str, delay_seconds: int = 600):
    await asyncio.sleep(delay_seconds)
    if os.path.exists(path):
        os.remove(path)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

app.middleware("http")(executable_time)

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

@app.get("/healthz")
async def health():
    return {"status": "ok"}

@app.post("/img64Hugging", tags=["Hugging Face Hub"])
async def img_base64_hugging(data: HuggingPromptRequest, authorization: str = Header(default = None)):
    """Output image data in base64 format"""
    # Check model available
    model = data.model.strip()
    token = resolve_token(authorization)
    if token is None:
        token = HUGGING_FACE_API_KEY
    if data.model is None:
        model = "openfree/flux-chatgpt-ghibli-lora"
    try:
        image_data = await imageBase64Generate(
            prompt=data.prompt, 
            model=model,
            token=token,
        )
        return JSONResponse(content={"image_base64": image_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/imgFileHugging", tags=["Hugging Face Hub"])
async def img_file_hugging(
    data: HuggingPromptRequest, 
    background_tasks: BackgroundTasks,
    authorization: str = Header(default = None)
):
    """Output image file"""
    model = data.model.strip()
    token = resolve_token(authorization)
    if token is None:
        token = HUGGING_FACE_API_KEY
    # if not await is_model_available(model=model, token=token):
    if data.model is None:
        model = "openfree/flux-chatgpt-ghibli-lora"
    try:
        image_data = await imageGenerate(
            prompt=data.prompt, 
            model=model, 
            token=token
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode="wb") as image_file:
            image_file.write(image_data)
            image_path = image_file.name

        background_tasks.add_task(os.unlink, image_path)
        return FileResponse(image_path, media_type="image/png", filename="output.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/imgHugging", tags=["Hugging Face Hub"])
async def img_raw_hugging(data: HuggingPromptRequest, authorization: str = Header(default = None)):
    """Output image directly"""
    model = data.model.strip()
    token = resolve_token(authorization)
    if token is None:
        token = HUGGING_FACE_API_KEY 
    if data.model is None:
        model = "openfree/flux-chatgpt-ghibli-lora"
    try:
        image_data = await imageGenerate(
            prompt=data.prompt,
            model=model,
            token=token
        )
        image_stream = io.BytesIO(image_data)
        return StreamingResponse(image_stream, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/img64ImagePig", tags=["Image Pig"])
async def img_base64_imagepig(
    data: ImagePigPromptRequest, 
    authorization: str = Header(default = None)):
    """Output image data in base64 format"""
    # return {"data": "Hehe"}
    
    if authorization is None:
        token = IMAGE_PIG
    else:
        token = authorization
    try:
        # image_data = await ImagePigimageBase64Generate(prompt=data.prompt, token=token)
        image_data = await fetchImagePig(prompt=data.prompt, token=token)
        return JSONResponse(content={"image_base64": str(image_data)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/imgFileImagePig", tags=["Image Pig"])
async def img_file_imagepig(
    data: ImagePigPromptRequest, 
    request: Request,
    background_tasks: BackgroundTasks,
    # download: bool = Query(default=False),
    authorization: str = Header(default = None),
):
    """Output image file"""
    if authorization is None:
        token = IMAGE_PIG
    else:
        token = authorization 
    try:
        image_data = await fetchImagePig(prompt=data.prompt, token=token)
        # Use this to store file temporary after response
        unique_id = f"{uuid.uuid4().hex}.png"
        file_path = f"temp_images/{unique_id}"
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        background_tasks.add_task(delete_file_later, file_path, 600)
        url = str(request.base_url) + f"/images/{unique_id}"
        return JSONResponse(content={"url": url})
        # Use this to delete file after response
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode="wb") as image_file:
        #     image_file.write(base64.b64decode(image_data))
        #     image_path = image_file.name
        # background_tasks.add_task(os.unlink, image_path)
        # if download:
        #     return FileResponse(image_path, media_type="image/png", filename="output.png")
        # else:
        #     return FileResponse(image_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
@app.post("/imgImagePig", tags=["Image Pig"])
async def img_raw_imagepig(data: ImagePigPromptRequest, authorizatino: str = Header(default = None)):
    """Output image directly"""
    if authorizatino is None:
        token = IMAGE_PIG
    else:
        token = authorizatino
    try:
        image_data = await fetchImagePig(prompt=data.prompt, token=token)
        decoded_img = base64.b64decode(image_data)
        image_stream = io.BytesIO(decoded_img)
        return StreamingResponse(image_stream, media_type = "image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

if __name__ == "__main__":
    import uvicorn
    port = 8080
    uvicorn.run(app, host="0.0.0.0", port=port)

