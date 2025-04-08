from fastapi import FastAPI, Request, HTTPException

from classes import PromptRequest

import time 
import datetime

app = FastAPI()

promptRequest = PromptRequest


@app.middleware("http")
async def executable_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Execution-Time"] = str(datetime.timedelta(seconds=process_time))
    return response

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

