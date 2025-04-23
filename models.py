from pydantic import BaseModel, Field, constr, HttpUrl
from typing import Optional, Literal

SAMPLE_TEXT = "Astronat riding a horse"

class HuggingPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example = SAMPLE_TEXT) # type: ignore
    model: Optional[Literal[
        "openfree/flux-chatgpt-ghibli-lora"
    ]] = Field(default="openfree/flux-chatgpt-ghibli-lora", example="openfree/flux-chatgpt-ghibli-lora")

class ImagePigPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example = SAMPLE_TEXT) # type: ignore

class OpenAIPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example = SAMPLE_TEXT) # type: ignore
    model: Optional[Literal["dall-e-2"]] = Field(default="dall-e-2")

class OpenAIEditPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(...) # type: ignore
    img_url: HttpUrl 
    mask_url: Optional[HttpUrl] = None
    model: Optional[Literal["dall-e-2"]] = Field(default="dall-e-2")
    
class ImaginePromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example = SAMPLE_TEXT) # type: ignore 
    style: Literal["realistic", "anime", "flux-schnell", "flux-dev-fast", "flux-dev", "imagine-turbo"]
    aspect: Literal["1:1", "3:2", "4:3", "3:4", "16:9", "9:16"]

