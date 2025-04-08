from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv

from models import PromptRequest
from middleware import executable_time
from functions import is_model_available, imageBase64Generate, imageGenerate, resolve_token

import logging
import tempfile
import os 

app = FastAPI()

load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
if not HUGGING_FACE_API_KEY:
    raise EnvironmentError("Please check environment variables or .env file.")

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

@app.get("/healthz")
async def health():
    return {"status": "ok"}

@app.post("/img64", tags=["Image Generation"])
async def img_base64(data: PromptRequest, authorization: str = Header(default = None)):
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

@app.post("/img")
async def img(
    data: PromptRequest, 
    background_tasks: BackgroundTasks,
    authorization: str = Header(default = None)
):
    """Output image data raw"""
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

if __name__ == "__main__":
    import uvicorn
    port = 8080
    uvicorn.run(app, host="0.0.0.0", port=port)

