from pydantic import BaseModel 
from typing import Optional

class PromptRequest(BaseModel):
    prompt: str
    model: Optional[str] = "openfree/flux-chatgpt-ghibli-lora"



