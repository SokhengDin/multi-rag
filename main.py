import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from contextlib import asynccontextmanager

from app.api import api_router
from app.schemas.base_schema import RESPONSE_SCHEMA
from app.core.config import settings
from app.core.device import check_device
from app.providers.embedder import load_dense_embedder
from app.utils import minio_storage as storage
from app.db import session as db
from app.middleware.http_middleware import HttpMiddleware
from app import logger


@asynccontextmanager
async def lifespan(
    app: FastAPI
):
    logger.info("Starting application ...")

    app.state.device = check_device()
    logger.info(f"Detected device: {app.state.device}")

    async def check_storage():
        storage_ok, storage_detail = await asyncio.to_thread(storage.check_connection)
        if storage_ok:
            logger.info(f"Storage connection established: {storage_detail}")
        else:
            logger.error(f"Storage connection failed: {storage_detail}")

    async def check_db():
        db_ok, db_detail = await asyncio.to_thread(db.check_connection)
        if db_ok:
            logger.info(f"Database connection established: {db_detail}")
        else:
            logger.error(f"Database connection failed: {db_detail}")

    await asyncio.gather(check_storage(), check_db())

    await asyncio.to_thread(load_dense_embedder)

    yield

    logger.info("Shutting application ...")



app = FastAPI(
    lifespan=lifespan
    , title="Multi Rag API"
)

app.add_middleware(HttpMiddleware)

app.add_middleware(
    CORSMiddleware
    , allow_origins=["*"]
    , allow_credentials=True
    , allow_methods=["*"]
    , allow_headers=["*"]
)

app.include_router(api_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code = exc.status_code,
        content     = RESPONSE_SCHEMA(status=exc.status_code, message=str(exc.detail)).model_dump(),
        headers     = exc.headers,
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors      = []
    for error in exc.errors():
        errors.append({
            "field"     : " -> ".join(str(x) for x in error["loc"]),
            "message"   : error["msg"],
            "type"      : error["type"]
        })
    
    client = request.client.host if request.client else "unknown"
    logger.error(f"[VALIDATION_ERROR] {request.method} {request.url.path} from {client}: {errors}")

    error_response = RESPONSE_SCHEMA(
        status  = 422,
        message = "Validation error",
        data    = {"errors": errors}
    )
    
    return JSONResponse(
        status_code = 422,
        content     = error_response.model_dump()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app"
        , host = "0.0.0.0"
        , port = settings.API_PORT
    )