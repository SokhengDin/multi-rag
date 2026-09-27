import time

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from app.schemas.base_schema import RESPONSE_SCHEMA
from app.core.logger import color_status_code
from app import logger


class HttpMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        start   = time.perf_counter()
        client  = request.client.host if request.client else "unknown"

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(f"[UNHANDLED_ERROR] {request.method} {request.url.path} from {client}")
            response = JSONResponse(
                status_code = 500,
                content     = RESPONSE_SCHEMA(status=500, message="Internal server error").model_dump(),
            )

        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

        logger.info(
            f"{client} - {request.method} {request.url.path} "
            f"{color_status_code(response.status_code)} {duration_ms:.2f}ms"
        )

        return response
