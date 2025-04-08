from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login

from models import PromptRequest, ImageResponse
from middleware import executable_time
from functions import is_model_available, imageBase64Generate, imageGenerate

import time 
import datetime
import logging
import tempfile
import os 

app = FastAPI()

load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
if not HUGGING_FACE_API_KEY:
    raise EnvironmentError("HUGGING_FACE_API_KEY is not set. Please check environment vairables or .env file.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.post("/img64", response_model=ImageResponse)
async def img_base64(data: PromptRequest):
    """Output image data in base64 format"""
    # Check model available
    model = data.model.strip()
    if not await is_model_available(model=model, token=HUGGING_FACE_API_KEY):
        raise HTTPException(status_code=400, detail=f"Model {model} is not avaiable or may be privated")
    try:
        image_data = await imageBase64Generate(
            prompt=data.prompt, 
            model=model,
            token=HUGGING_FACE_API_KEY,
        )
        return JSONResponse(content={"image_base64": image_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/img", response_model=ImageResponse)
async def img(data: PromptRequest):
    """Output image data raw"""
    model = data.model.strip()
    if not await is_model_available(model=model, token=HUGGING_FACE_API_KEY):
        raise HTTPException(status_code=400, detail=f"Model {model} is not avaiable or may be privated")
    try:
        image_data = await imageGenerate(
            prompt=data.prompt, 
            model=model, 
            token=HUGGING_FACE_API_KEY
        )
        image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image_path.write(image_data)
        image_path.close()
        return FileResponse(image_path.name, media_type="image/png", filename="output.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = 8080
    uvicorn.run(app, host="0.0.0.0", port=port)

