from pydantic import BaseModel, Field, constr, HttpUrl
from typing import Optional, Literal

class HuggingPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example="Astronaunt riding a horse") # type: ignore
    # model: Optional[str] = "openfree/flux-chatgpt-ghibli-lora"
    model: Optional[Literal[
        "openfree/flux-chatgpt-ghibli-lora"
    ]] = Field(default="openfree/flux-chatgpt-ghibli-lora", example="openfree/flux-chatgpt-ghibli-lora")

class ImagePigPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example="Astronaunt riding a horse") # type: ignore

class OpenAIPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example="Astronaunt riding a horse") # type: ignore
    model: Optional[Literal["dall-e-2"]] = Field(default="dall-e-2")

class OpenAIEditPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(...) # type: ignore
    img_url: HttpUrl 
    mask_url = Optional[HttpUrl] = None
    model: Optional[Literal["dall-e-2"]] = Field(default="dall-e-2")
    


