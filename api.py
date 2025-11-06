# api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import rag_core # Import our refactored logic

# --- Pydantic Models for API Validation ---
class IngestRequest(BaseModel):
    texts: List[str] = Field(..., description="A list of raw text documents to ingest.")
    metadatas: List[Dict] = Field(..., description="A parallel list of metadata dictionaries. Each dict must contain a 'source'.")

class IngestResponse(BaseModel):
    message: str
    num_chunks_added: int
    total_chunks_in_db: int
    embedding_dimensions: int
    new_chunk_ids: List[str]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    source_filter: Optional[str] = None

class ContextChunk(BaseModel):
    id: str
    text: str
    metadata: Dict
    score: float

class QueryResponse(BaseModel):
    answer: str
    context_chunks: List[ContextChunk]
    llm_model_used: Optional[str] = None
    llm_usage: Optional[Dict] = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAG System API",
    description="API for ingesting documents and querying them with a RAG pipeline.",
    version="1.0.0"
)

@app.post("/ingest", response_model=IngestResponse, tags=["1. Ingestion"])
async def ingest_documents(request: IngestRequest):
    """
    Ingests and processes a list of texts into the vector database.
    """
    try:
        result = rag_core.ingest_texts(texts=request.texts, metadatas=request.metadatas)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@app.post("/query", response_model=QueryResponse, tags=["2. Querying"])
async def query_documents(request: QueryRequest):
    """
    Queries the RAG system to get a generated answer based on stored documents.
    """
    try:
        context_chunks = rag_core.query_vector_store(
            query=request.query, 
            top_k=request.top_k, 
            source_filter=request.source_filter
        )
        if not context_chunks:
            return QueryResponse(answer="Could not find relevant context for this query.", context_chunks=[])
        
        generation_result = await rag_core.generate_answer(
            query=request.query, 
            context_chunks=context_chunks
        )
        return QueryResponse(
            answer=generation_result['answer'], 
            context_chunks=context_chunks,
            llm_model_used=generation_result.get('llm_model_used'),
            llm_usage=generation_result.get('llm_usage')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

@app.post("/query/stream", tags=["2. Querying"])
async def query_documents_stream(request: QueryRequest):
    """
    Queries the RAG system and streams the answer token by token.
    """
    context_chunks = rag_core.query_vector_store(
        query=request.query, 
        top_k=request.top_k, 
        source_filter=request.source_filter
    )
    if not context_chunks:
        async def empty_stream():
            yield "Could not find relevant context for this query."
        return StreamingResponse(empty_stream(), media_type="text/plain")
        
    stream_generator = rag_core.generate_answer_stream(request.query, context_chunks)
    return StreamingResponse(stream_generator, media_type="text/plain")