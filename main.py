import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.schemas.base_schema import RESPONSE_SCHEMA
from app.core.config import settings
from app.core.device import check_device
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

    storage_task = asyncio.create_task(check_storage())
    db_task      = asyncio.create_task(check_db())

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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors      = []
    for error in exc.errors():
        errors.append({
            "field"     : " -> ".join(str(x) for x in error["loc"]),
            "message"   : error["msg"],
            "type"      : error["type"]
        })
    
    logger.error(f"[VALIDATION_ERROR] Request from {request.client.host if request.client else 'unknown'}")
    logger.error(f"[VALIDATION_ERROR] Headers: {dict(request.headers)}")
    logger.error(f"[VALIDATION_ERROR] Errors: {errors}")
    
    error_response = RESPONSE_SCHEMA(
        status  = 422,
        message = "Validation error",
        data    = {
            "errors": errors,
            "body"  : exc.body
        }
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