# middleware.py

import time
import datetime
from fastapi import Request
from fastapi.responses import Response

async def executable_time(request: Request, call_next) -> Response:
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Execution-Time"] = str(datetime.timedelta(seconds=process_time))
    return response
