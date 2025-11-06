# rag_core.py
import os
import json
import uuid
import numpy as np
import httpx
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from dotenv import load_dotenv # <--- ADD THIS LINE

# --- Load Environment Variables ---
load_dotenv() # <--- ADD THIS LINE to explicitly load from .env

# --- Configuration ---
CHROMA_PERSIST_DIR = "vector_store"
CHROMA_COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# --- Startup Check ---
if not TOGETHER_API_KEY: # <--- ADD THIS CHECK
    raise ValueError(
        "TOGETHER_API_KEY is not set in the environment. "
        "Please create a .env file and add your key."
    )

# --- Global Resources ---
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

def chunk_text_by_sentences(text: str, window_size: int = 4, overlap: int = 1) -> list[str]:
    # (This function is unchanged)
    import nltk
    if not text: return []
    sentences = nltk.sent_tokenize(text)
    if not sentences: return []
    chunks = []
    step = window_size - overlap
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i:i + window_size]
        if chunk_sentences:
            chunks.append(" ".join(chunk_sentences))
    return chunks

def ingest_texts(texts: List[str], metadatas: List[Dict]) -> Dict:
    """Chunks, embeds, and stores a list of texts in ChromaDB."""
    if len(texts) != len(metadatas):
        raise ValueError("The number of texts must match the number of metadatas.")

    all_chunk_texts = []
    all_metadatas = []
    all_ids = []
    
    for i, text in enumerate(texts):
        chunks = chunk_text_by_sentences(text)
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            all_chunk_texts.append(chunk)
            # Add chunk-specific info to the provided metadata
            chunk_metadata = metadatas[i].copy()
            chunk_metadata["ingestion_date"] = datetime.now().isoformat()
            all_metadatas.append(chunk_metadata)
            all_ids.append(chunk_id)

    if not all_chunk_texts:
        return {"message": "No valid chunks were created from the provided texts."}

    # ChromaDB can take a list of documents, metadatas, and ids
    collection.add(
        documents=all_chunk_texts,
        metadatas=all_metadatas,
        ids=all_ids
    )

    return {
        "message": "Ingestion successful.",
        "num_chunks_added": len(all_chunk_texts),
        "total_chunks_in_db": collection.count(),
        "embedding_dimensions": embedding_model.get_sentence_embedding_dimension(),
        "new_chunk_ids": all_ids
    }

def query_vector_store(query: str, top_k: int, source_filter: Optional[str] = None) -> List[Dict]:
    """Queries ChromaDB for relevant chunks with metadata filtering."""
    query_embedding = embedding_model.encode(query).tolist()

    where_clause = {}
    if source_filter:
        where_clause = {"source": source_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause
    )

    # Format the results to be more user-friendly
    formatted_chunks = []
    if results and results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            l2_distance = results['distances'][0][i]
            # Convert L2 distance to a 0-1 similarity score (for normalized embeddings)
            similarity_score = 1.0 - (l2_distance**2) / 2
            
            formatted_chunks.append({
                "id": results['ids'][0][i],
                "text": doc,
                "metadata": results['metadatas'][0][i],
                "score": max(0.0, min(1.0, similarity_score)) # Clamp score
            })
    return formatted_chunks

async def generate_answer(query: str, context_chunks: List[Dict]) -> Dict:
    """Generates a final answer using the LLM."""
    context_str = "\n\n---\n\n".join(
        f"Source: {chunk['metadata'].get('source', 'N/A')}\nContent: {chunk['text']}" for chunk in context_chunks
    )
    system_prompt = "You are an expert AI assistant..." # (Same as before)
    user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}\n\nANSWER:"
    
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    payload = {"model": LLM_MODEL, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "max_tokens": 1024}

    async with httpx.AsyncClient(timeout=90) as client:
        try:
            r = await client.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return {
                "answer": data['choices'][0]['message']['content'].strip(),
                "llm_model_used": data.get('model'),
                "llm_usage": data.get('usage')
            }
        except Exception as e:
            return {"answer": f"Error generating answer: {e}", "llm_model_used": LLM_MODEL}

async def generate_answer_stream(query: str, context_chunks: List[Dict]):
    """Yields LLM response tokens as they arrive."""
    # (Implementation is the same, just with "stream": True)
    context_str = "\n\n---\n\n".join(
        f"Source: {chunk['metadata'].get('source', 'N/A')}\nContent: {chunk['text']}" for chunk in context_chunks
    )
    system_prompt = "You are an expert AI assistant..."
    user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}\n\nANSWER:"
    
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    payload = {"model": LLM_MODEL, "messages": [{"role": "system", "content": user_prompt}], "stream": True}

    async with httpx.AsyncClient(timeout=90) as client:
        async with client.stream("POST", "https://api.together.xyz/v1/chat/completions", headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    line_data = line[5:].strip()
                    if line_data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line_data)
                        token = chunk['choices'][0]['delta'].get('content', '')
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue