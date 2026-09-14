import logging

from fastapi import APIRouter, Request, HTTPException

from app.models.schemas import KrishiAgentRequest, KrishiAgentResponse
from app.core.llm_chain import answer_query

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/krishi-agent/ask", response_model=KrishiAgentResponse)
def ask_krishi_agent(payload: KrishiAgentRequest, request: Request):
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store load nahi hua. Pehle ingestion chalao: python -m app.ingestion.ingest_docs",
        )

    try:
        result = answer_query(
            vector_store,
            query=payload.query,
            crop=payload.crop,
            district=payload.district,
        )
    except Exception as e:
        logger.exception("Krishi agent query failed")
        raise HTTPException(status_code=502, detail=f"LLM providers dono fail ho gaye: {e}")

    return KrishiAgentResponse(**result)
