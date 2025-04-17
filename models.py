from pydantic import BaseModel, Field, constr, validator
from typing import Optional, Literal

class HuggingPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example="Astronaunt riding a horse") # type: ignore
    # model: Optional[str] = "openfree/flux-chatgpt-ghibli-lora"
    model: Optional[Literal[
        "openfree/flux-chatgpt-ghibli-lora"
    ]] = Field(default="openfree/flux-chatgpt-ghibli-lora", example="openfree/flux-chatgpt-ghibli-lora")

    @validator("prompt")
    def prompt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt must not be empty.")
        return v

class ImagePigPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example="Astronaunt riding a horse") # type: ignore
    @validator("prompt")
    def promt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt must not be empty.")
        return v

class OpenAIPromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example="Astronaunt riding a horse") # type: ignore
    model: Optional[Literal["dall-e-2"]] = Field(default="dall-e-2")
    


