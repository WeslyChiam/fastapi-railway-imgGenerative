from diffusers import AutoPipelineForText2Image
from io import BytesIO
import torch
import base64
import httpx
import asyncio
import os
import logging

TIMEOUT = 60.0 
RETRY_DELY = 2.5
RETRIES = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {
        "Authorization": f"Bearer {token}", 
        "Content-Type": "application/json",
    }
    payload = {"inputs": prompt}

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    image_bytes = response.content
                    return base64.b64encode(image_bytes).decode("utf-8")
                elif response.status_code in [500, 502, 503, 504]:
                    logger.warning(f"Attempt {attempt+1}: Temporary error {response.status_code}, retrying...")
                else:
                    logger.error(f"Attempt {attempt+1}: Unexpected status {response.status_code}, retrying...")
                    response.raise_for_status()
        except httpx.RequestError as e:
            logger.error(f"Attempt {attempt+1} Request error: {e}")
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Unexpected error: {e}")
        await asyncio.sleep(delay)
    raise RuntimeError("Failed to get image from Hugging Face API after retries.")

async def imageGenerate(
        prompt: str, 
        model: str, 
        token: str, 
        retries: int = RETRIES, 
        delay: float = RETRY_DELY, 
):
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {
        "Authorization": f"Bearer {token}", 
        "Content-Type": "application/json",
    }
    payload = {"inputs": prompt}

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    # image_bytes = response.content
                    # return base64.b64encode(image_bytes).decode("utf-8")
                    return response.content
                elif response.status_code in [500, 502, 503, 504]:
                    logger.warning(f"Attempt {attempt+1}: Temporary error {response.status_code}, retrying...")
                else:
                    logger.error(f"Attempt {attempt+1}: Unexpected status {response.status_code}, retrying...")
                    response.raise_for_status()
        except httpx.RequestError as e:
            logger.error(f"Attempt {attempt+1} Request error: {e}")
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Unexpected error: {e}")
        await asyncio.sleep(delay)
    raise RuntimeError("Failed to get image from Hugging Face API after retries.")
