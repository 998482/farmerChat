from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    vector_store_loaded = getattr(request.app.state, "vector_store", None) is not None
    return HealthResponse(status="ok", vector_store_loaded=vector_store_loaded)
