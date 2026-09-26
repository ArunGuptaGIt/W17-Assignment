import shutil
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.models.schemas import IngestResponse
from app.services.ingestion_service import IngestionService

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api", tags=["Ingestion"])

ingestion_service = IngestionService()

@router.post("/ingest", response_model=IngestResponse)
@limiter.limit("5/minute")
async def ingest_endpoint(request: Request, file: UploadFile = File(...)):
    """
    Ingest PDF, TXT, or Markdown document.
    Executes page-aware loading, stable chunk ID generation, dual indexing, and cache invalidation.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Only .pdf, .txt, and .md allowed.")

    dest_path = settings.DOCUMENTS_DIR / file.filename

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        res = await ingestion_service.ingest_file(dest_path)
        return res

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion processing failed: {str(e)}")
