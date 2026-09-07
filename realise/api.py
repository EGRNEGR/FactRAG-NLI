"""FastAPI entrypoint for the on-premise RAG pipeline.

Наш прежний API: ``uvicorn api:app --host 127.0.0.1 --port 8000``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)


class IngestRequest(BaseModel):
    path: str = Field(min_length=1, description="Path visible to the local service")
    backend: str = Field(default="auto", pattern="^(auto|text|pymupdf|unstructured|docling)$")


class TextIngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(default="api")
    document_id: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=100)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    """Create an app with dependency injection for tests and multiple local tenants."""
    rag = pipeline or RAGPipeline.from_environment()
    service = FastAPI(title="On-premise Technical RAG", version="0.1.0")

    def health() -> dict[str, Any]:
        return {"status": "ok", "documents": len(rag.documents), "chunks": rag.retriever.size}

    def ingest(request: IngestRequest) -> dict[str, Any]:
        try:
            document = rag.ingest_path(request.path, backend=request.backend)
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "document_id": document.document_id,
            "source": document.source,
            "elements": len(document.elements),
            "chunks": len(document.chunks),
        }

    def ingest_text(request: TextIngestRequest) -> dict[str, Any]:
        document = rag.ingest_text(
            request.text, source=request.source, document_id=request.document_id
        )
        return {
            "document_id": document.document_id,
            "elements": len(document.elements),
            "chunks": len(document.chunks),
        }

    def query(request: QueryRequest) -> dict[str, Any]:
        try:
            response = rag.query(request.query, top_k=request.top_k)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return response.model_dump(mode="json")

    service.add_api_route("/health", health, methods=["GET"])
    service.add_api_route("/ingest", ingest, methods=["POST"])
    service.add_api_route("/ingest/text", ingest_text, methods=["POST"])
    service.add_api_route("/query", query, methods=["POST"])
    return service


app = create_app()
