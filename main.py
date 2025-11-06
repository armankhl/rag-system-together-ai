import os
import json
import asyncio
import numpy as np
import httpx
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# --- Configuration (must match ingestion.py) ---
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "docs.index")
CHUNKS_JSON_PATH = os.path.join(VECTOR_STORE_DIR, "docs_chunks.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Global cache for loaded resources
_embedding_model_cache = None
_faiss_index_cache = None
_chunk_data_cache = None

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is not set.")

def load_retrieval_resources():
    """Loads all necessary resources for retrieval into a global cache."""
    global _embedding_model_cache, _faiss_index_cache, _chunk_data_cache
    
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Vector store not found. Run 'python ingestion.py' first.")

    if _embedding_model_cache is None:
        _embedding_model_cache = SentenceTransformer(EMBEDDING_MODEL)
    if _faiss_index_cache is None:
        _faiss_index_cache = faiss.read_index(FAISS_INDEX_PATH)
    if _chunk_data_cache is None:
        with open(CHUNKS_JSON_PATH, 'r', encoding='utf-8') as f:
            _chunk_data_cache = json.load(f)

def retrieve_context(query: str, top_k: int = 5, source_filter: str = None) -> str:
    """
    Retrieves relevant context using a two-stage process:
    1. Filter chunks by metadata (if a filter is provided).
    2. Perform a vector search on the filtered (or full) set of chunks.
    """
    # --- Stage 1: Metadata Filtering ---
    candidate_indices = list(range(len(_chunk_data_cache))) # Start with all possible indices
    if source_filter:
        print(f"Applying filter: source='{source_filter}'")
        candidate_indices = [
            i for i, chunk in enumerate(_chunk_data_cache)
            if chunk['metadata']['source'].lower() == source_filter.lower()
        ]
        if not candidate_indices:
            print("No chunks found matching the metadata filter.")
            return "I could not find any documents matching your filter criteria."

    # --- Stage 2: Vector Search ---
    print(f"Performing vector search on {len(candidate_indices)} candidate chunks...")
    query_embedding = _embedding_model_cache.encode([query]).astype('float32')

    # If we have a filtered list, we need to search on a subset of vectors
    if len(candidate_indices) != len(_chunk_data_cache):
        # Create a temporary index with only the vectors of the candidate chunks
        filtered_vectors = np.array([_faiss_index_cache.reconstruct(i) for i in candidate_indices])
        temp_index = faiss.IndexFlatL2(filtered_vectors.shape[1])
        temp_index.add(filtered_vectors)
        # Search this temporary index
        distances, temp_indices = temp_index.search(query_embedding, min(top_k, len(candidate_indices)))
        # Map the temporary indices back to the original FAISS indices
        final_indices = [candidate_indices[i] for i in temp_indices[0]]
    else:
        # If no filter, search the full index
        distances, final_indices_array = _faiss_index_cache.search(query_embedding, top_k)
        final_indices = final_indices_array[0]

    retrieved_chunks = [_chunk_data_cache[i] for i in final_indices]
    
    # Add metadata to the context for better LLM understanding
    context_with_metadata = [
        f"Source: {chunk['metadata']['source']}, Page: {chunk['metadata']['page_number']}\nContent: {chunk['text']}"
        for chunk in retrieved_chunks
    ]
    
    return "\n\n---\n\n".join(context_with_metadata)


async def generate_answer(query: str, context: str) -> str:
    """Generates an answer using a direct call to the Together AI API."""
    # (This function remains unchanged)
    print("Generating answer with LLM...")
    system_prompt = (
        "You are an expert AI assistant. Answer the question based *only* on the provided context. "
        "The context may include metadata like 'Source' and 'Page'. Use this to be more precise if needed. "
        "If the information is not in the context, say that you cannot find the answer in the provided documents."
    )
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
    
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    payload = {"model": LLM_MODEL, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "max_tokens": 1024, "temperature": 0.7}

    async with httpx.AsyncClient(timeout=90) as client:
        try:
            r = await client.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"An error occurred during LLM call: {e}"

async def chat_loop():
    """Main async function to run the interactive chat loop with filtering."""
    load_retrieval_resources()
    print(f"\n✅ RAG system is ready. Ask your questions!")
    print("   (Type 'exit' to quit)")

    while True:
        question = input("\n> Enter your question: ")
        if question.lower() == 'exit': break

        source_filter = input("> Enter source to filter by (e.g., 'bitcoin.pdf'), or press Enter to skip: ")
        source_filter = source_filter.strip() if source_filter.strip() else None

        context = retrieve_context(question, source_filter=source_filter)
        answer = await generate_answer(question, context)
        
        print("\n💡 Answer:")
        print(answer)

if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")