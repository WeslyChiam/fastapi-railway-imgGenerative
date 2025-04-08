from pydantic import BaseModel, Field, constr, validator
from typing import Optional, Literal

class PromptRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(..., example="Astronaunt riding a horse")
    # model: Optional[str] = "openfree/flux-chatgpt-ghibli-lora"
    model: Optional[Literal[
        "openfree/flux-chatgpt-ghibli-lora"
    ]] = Field(default="openfree/flux-chatgpt-ghibli-lora", example="openfree/flux-chatgpt-ghibli-lora")

    @validator("prompt")
    def prompt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt must not be empty.")
        return v

