import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.rag.retrieval.service import search, search_by_image
from app.schemas.base_schema import RESPONSE_SCHEMA
from app.schemas.search_schema import SearchFilters, SearchIn, SearchOut

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=RESPONSE_SCHEMA[SearchOut])
def search_chunks(request: SearchIn, session: Session = Depends(get_session)):
    """dense: vector search. hybrid: dense + full-text (+ sparse) fused with RRF. rerank: cross-encoder on the top results."""
    return RESPONSE_SCHEMA(status=200, message="OK", data=search(session, request))


@router.post("/image", response_model=RESPONSE_SCHEMA[SearchOut])
def search_chunks_by_image(
    file       : UploadFile  = File(...),
    query      : str | None  = Form(None, max_length=2000, description="Optional text, embedded together with the image"),
    top_k      : int         = Form(10, ge=1, le=100),
    candidates : int         = Form(100, ge=1, le=1000),
    filters    : str | None  = Form(None, description='SearchFilters as JSON, e.g. {"modalities": ["image"]}'),
    session    : Session     = Depends(get_session),
):
    """Find chunks similar to an image: figures, tables, pages, or text that describes it."""
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except OSError:
        raise HTTPException(status_code=400, detail="File is not a readable image")

    try:
        parsed = SearchFilters.model_validate_json(filters) if filters else SearchFilters()
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Invalid filters: {e.errors(include_url=False)}")

    result = search_by_image(session, image, query, parsed, top_k, candidates)
    return RESPONSE_SCHEMA(status=200, message="OK", data=result)
