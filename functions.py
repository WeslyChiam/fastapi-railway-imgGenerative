from pathlib import Path

import base64
import httpx
import asyncio
import logging
import requests
import shutil
import tempfile
import mimetypes
import os

TIMEOUT = 60.0 
RETRY_DELY = 2.5
RETRIES = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def resolve_token(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    if auth_header.lower().startswith("bearer "):
        # return auth_header.strip()
        token = auth_header[7:].strip()
    else:
        token = auth_header.strip()
    if token.startswith("hf_"):
        return token
    return None

async def fetch_image_bytes(prompt, model, token, retries=RETRIES, delay=RETRY_DELY):
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": prompt}
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout = TIMEOUT) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    image_bytes = response.content 
                    return image_bytes
                elif response.status_code in [500, 502, 503, 504]:
                    logger.warning(f"Attempt {attempt+1}: Temporary error {response.status_code}, retrying...")
                else:
                    logger.error(f"Attempt {attempt+1}: Unexpected status {response.status_code}, retrying...")
                    response.raise_for_status()
        except httpx.RequestError as e:
            logger.error(f"Attempt {attempt+1} Request error: {e}")
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Unexpected error: {e}")
        await asyncio.sleep(min(delay * (2 ** attempt), 10.0))
    logger.error("All retry attempts failed.")
    raise RuntimeError("Failed to get image from Hugging Face API after retries.")

async def is_model_available(model: str, token: str) -> bool:
    """Check if the Hugging Face Model is available"""
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {
        "Authorization": f"Bearer {token}", 
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout = TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                logger.info(f"Model {model} is available!")
                return True
            logger.error(f"Model {model} not available. Status: {resp.status_code}")
            return False
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    return False

async def imageBase64Generate(
        prompt: str, 
        model: str, 
        token: str, 
        retries: int = RETRIES, 
        delay: float = RETRY_DELY, 
):
    image_bytes = await fetch_image_bytes(
        prompt=prompt, 
        model=model, 
        token=token, 
        retries=retries, 
        delay=delay, 
    )
    return base64.b64encode(image_bytes).decode("utf-8")

async def imageGenerate(
        prompt: str, 
        model: str, 
        token: str, 
        retries: int = RETRIES, 
        delay: float = RETRY_DELY, 
):
    image_bytes = await fetch_image_bytes(
        prompt=prompt, 
        model=model, 
        token=token, 
        retries=retries, 
        delay=delay, 
    )
    return image_bytes

async def fetchImagePig(
        prompt: str, 
        token: str,
):
    response = requests.post(
        "https://api.imagepig.com/",
        headers = {"Api-Key": token}, 
        json = {"prompt": prompt}
    )
    if response.status_code == 200:
        return response.json()["image_data"]
    else:
        response.raise_for_status()

async def ImagePigimageBase64Generate(
        prompt: str, 
        token: str, 
):
    image_data = await fetchImagePig(
        prompt=prompt, 
        token=token, 
    )
    return image_data

async def download_img_tmp(url: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url=url)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "image" not in content_type:
                raise ValueError(f"URL did not return an image, {content_type}")
            
            mime = content_type.split(";")[0]
            exts = mimetypes.guess_all_extensions(mime)
            ext = exts[0] if exts else ".jpg"

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            with open(temp_file.name, 'wb') as output:
                output.write(response.content)
            temp_file.close()
            return temp_file.name
    except Exception as e:
        raise ValueError(f"Unable to download {url}: {e}")
    
async def download_and_encode(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url = url)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "image" not in content_type:
                raise ValueError(f"URL did not return an image, {content_type}")
            encoded_str = base64.b64encode(response.content).decode("utf-8")
            return encoded_str
    except Exception as e:
        raise ValueError(f"Unable to download and encode image: {e}")




