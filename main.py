from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from typing import Optional, Annotated

from middleware import executable_time

import functions
import models

import asyncio
import base64
import httpx
import io
import logging
import os 
import requests
import shutil
import tempfile
import uuid

app = FastAPI()

bearer_scheme = HTTPBearer()

load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
if not HUGGING_FACE_API_KEY:
    raise EnvironmentError("Please check environment variables or .env file.")
IMAGE_PIG = os.getenv("IMAGE_PIG")
if not IMAGE_PIG:
    raise EnvironmentError("Please check environment variables or .env file.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_IMAGES_PATH = os.path.join(BASE_DIR, "temp_images")
app.mount("/images", StaticFiles(directory=TEMP_IMAGES_PATH), name="images")

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
async def img_base64_hugging(
    data: models.HuggingPromptRequest, 
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
):
    """Output image data in base64 format"""
    # Check model available
    model = data.model.strip()
    if data.model is None:
        model = "openfree/flux-chatgpt-ghibli-lora"
    try:
        image_data = await functions.imageBase64Generate(
            prompt=data.prompt, 
            model=model,
            token=credentials.credentials,
        )
        return JSONResponse(content={"image_base64": image_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/imgFileHugging", tags=["Hugging Face Hub"])
async def img_file_hugging(
    data: models.HuggingPromptRequest, 
    background_tasks: BackgroundTasks,
    # authorization: str = Header(default = None)
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
):
    """Output image file"""
    model = data.model.strip()
    # if not await is_model_available(model=model, token=token):
    if data.model is None:
        model = "openfree/flux-chatgpt-ghibli-lora"
    try:
        image_data = await functions.imageGenerate(
            prompt=data.prompt, 
            model=model, 
            token=credentials.credentials,
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode="wb") as image_file:
            image_file.write(image_data)
            image_path = image_file.name

        background_tasks.add_task(os.unlink, image_path)
        return FileResponse(image_path, media_type="image/png", filename="output.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/imgHugging", tags=["Hugging Face Hub"])
async def img_raw_hugging(
    data: models.HuggingPromptRequest, 
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
):
    """Output image directly"""
    model = data.model.strip()
    if data.model is None:
        model = "openfree/flux-chatgpt-ghibli-lora"
    try:
        image_data = await functions.imageGenerate(
            prompt=data.prompt,
            model=model,
            token=credentials.credentials,
        )
        image_stream = io.BytesIO(image_data)
        return StreamingResponse(image_stream, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/img64ImagePig", tags=["Image Pig"])
async def img_base64_imagepig(
    data: models.ImagePigPromptRequest, 
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]):
    """Output image data in base64 format"""
    try:
        # image_data = await ImagePigimageBase64Generate(prompt=data.prompt, token=token)
        image_data = await functions.fetchImagePig(prompt=data.prompt, token=credentials.credentials)
        return JSONResponse(content={"image_base64": str(image_data)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fileImagePig", tags=["Image Pig"])
async def img_file_imagepig(
    data: models.ImagePigPromptRequest, 
    background_tasks: BackgroundTasks, 
    request: Request, 
    # auth: Optional[str] = Header(None), 
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
):
    """Output image filepath for temperory (default: 10 minutes)"""
    # return JSONResponse(content={"test": str(token)})
    try:
        image_data = await functions.fetchImagePig(prompt=data.prompt, token=credentials.credentials)
        unique_id = f"{uuid.uuid4().hex}.png"
        file_path = f"temp_images/{unique_id}"
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        logger.info("Saved to:", os.path.abspath(file_path))
        background_tasks.add_task(delete_file_later, file_path, 600)
        # url = str(request.base_url).rstrip("/") + f"/{file_path}"
        url = str(request.base_url).replace("http://", "https://").rstrip("/") + f"/images/{unique_id}"
        return JSONResponse(content={"url": url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/imgImagePig", tags=["Image Pig"])
async def img_raw_imagepig(
    data: models.ImagePigPromptRequest, 
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
):
    """Output image directly"""
    try:
        image_data = await functions.fetchImagePig(prompt=data.prompt, token=credentials.credentials)
        decoded_img = base64.b64decode(image_data)
        image_stream = io.BytesIO(decoded_img)
        return StreamingResponse(image_stream, media_type = "image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/imgOpenAI", tags=["Open AI"])
async def img_openai(
    data: models.OpenAIPromptRequest,
    request: Request,
    background_task: BackgroundTasks, 
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
):
    """Output image via url using openai model"""
    try:
        client = OpenAI(api_key=credentials.credentials)
        response = client.images.generate(
            model = data.model, 
            prompt = data.prompt, 
            n = 1, 
            size = "1024x1024"
        )
        if response is not None:
            url = response.data[0].url #This is a signed url
            signed_response = requests.get(url)
            if signed_response.status_code == 200:
                image_data = signed_response.content
            else:
                raise HTTPException(status_code=500, detail=f"Failed to donwload image from {signed_response}")
            unique_id = f"{uuid.uuid4().hex}.png"
            file_path = f"temp_images/{unique_id}"
            with open(file_path, "wb") as f:
                f.write(image_data)
            background_task.add_task(delete_file_later, file_path, 600)
            new_url = str(request.base_url).replace("http://", "https://").rstrip("/") + f"/images/{unique_id}"
            return JSONResponse(content={"url": str(new_url)})
        raise HTTPException(status_code=500, detail="Failed to create image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/imgFileOpenAI", tags=["Open AI"])
async def img_file_openai(
    data: models.OpenAIPromptRequest, 
    background_tasks: BackgroundTasks,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)], 
):
    """Output image via file response"""
    try:
        client = OpenAI(api_key=credentials.credentials)
        response = client.images.generate(
            model = data.model, 
            prompt = data.prompt, 
            n = 1, 
            size = "1024x1024", 
        )
        if response is not None: 
            url = response.data[0].url
            # temp_path = functions.download_img_tmp(url=url)
            async with httpx.AsyncClient(timeout=10) as client: 
                img_response = await client.get(url=url)
                img_response.raise_for_status()
                content_type = img_response.headers.get("Content-Type", "").lower()
                if "image" not in content_type:
                    raise ValueError(f"URL did not return an image, {content_type}")
            try:
                encoded_str = base64.b64encode(img_response.content).decode('utf-8')
                unique_id = f"{uuid.uuid4().hex}.png"
                file_path = f"temp_images/{unique_id}"
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(encoded_str))
                background_tasks.add_task(delete_file_later, file_path, 600)
                return FileResponse(path=file_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail={"error": str(e)})
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/base64ImgOpenAI", tags=["Open AI"])
async def img_base64_openai(
    data: models.OpenAIPromptRequest, 
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
):
    """Output image encoded base64 using openai model"""
    try:
        client = OpenAI(api_key=credentials.credentials)
        response = client.images.generate(
            model = "dall-e-2", 
            prompt = data.prompt, 
            n = 1, 
            size = "256x256" 
        )
        if response is not None:
            image_response = await functions.download_and_encode(url=response.data[0].url)
            return JSONResponse(
                content= {
                    "image_base64": image_response, 
                }
            )
        raise HTTPException(status_code=500, detail="Failed to create image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/imgtmpFileImagine", tags=["Imagine"])
async def img_tmpfile_imagine(
    data: models.ImaginePromptRequest, 
    request: Request,
    background_task: BackgroundTasks, 
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)], 
):
    """Output image via url"""
    try:
        BASE_URL = "https://api.vyro.ai/v2/image/generations"
        headers = {"Authorization": f"Bearer {credentials.credentials}"}
        payload = {"prompt": (None, data.prompt), "style": (None, data.style), "aspect_ratio": (None, data.aspect)}
        response = requests.post(BASE_URL, headers = headers, files = payload)
        if response.status_code == 200:
            image_data = response.content
        else:
            response.raise_for_status()
        unique_id = f"{uuid.uuid4().hex}.png"
        file_path = f"temp_images/{unique_id}"
        with open(file_path, "wb") as f:
            f.write(image_data)
        background_task.add_task(delete_file_later, file_path, 600)
        url = str(request.base_url).replace("http://", "https://").rstrip("/") + f"/images/{unique_id}"
        return JSONResponse(content={"url": url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/imgFileImagine", tags=["Imagine"])
async def img_file_imagine(
    data: models.ImaginePromptRequest, 
    request: Request, 
    background_task: BackgroundTasks, 
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
):
    """Output image via file response"""
    try:
        BASE_URL = "https://api.vyro.ai/v2/image/generations"
        headers = {"Authorization": f"Bearer {credentials.credentials}"}
        payload = {"prompt": (None, data.prompt), "style": (None, data.style), "aspect_ratio": (None, data.aspect)}        
        response = requests.post(BASE_URL, headers=headers, files=payload)
        if response.status_code == 200:
            image_data = response.content
        else:
            response.raise_for_status()
        filename = "output.png"
        with open(filename, "wb") as f:
            f.write(image_data)
        return FileResponse(filename, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

if __name__ == "__main__":
    import uvicorn
    port = 8080
    uvicorn.run(app, host="0.0.0.0", port=port)

