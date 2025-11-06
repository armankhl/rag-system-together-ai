import os
import json
import nltk
import faiss
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

# --- Configuration ---
DATA_PATH = "data/"
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "docs.index")
CHUNKS_JSON_PATH = os.path.join(VECTOR_STORE_DIR, "docs_chunks.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def download_nltk_punkt():
    """Downloads the NLTK 'punkt' tokenizer if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' model...")
        nltk.download('punkt')

def chunk_text_by_sentences(text: str, window_size: int = 4, overlap: int = 1) -> list[str]:
    """Splits text into sentences and creates overlapping chunks of sentences."""
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

def run_ingestion():
    """
    Full pipeline to ingest PDFs, extract metadata, chunk them, create embeddings,
    and save everything for the retrieval stage.
    """
    download_nltk_punkt()

    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)

    # 1. Load Documents
    print(f"Loading documents from '{DATA_PATH}'...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    # The loader returns a list of Document objects, each with page_content and metadata
    documents = loader.load()
    if not documents:
        print(f"No documents found in '{DATA_PATH}'.")
        return

    # 2. Process and Chunk Documents with Metadata
    print("Processing and chunking documents with metadata...")
    all_chunks_with_metadata = []
    chunk_texts_for_embedding = []

    # Keep a unique ID for each chunk
    chunk_id_counter = 0

    for doc in documents:
        chunks = chunk_text_by_sentences(doc.page_content, window_size=4, overlap=1)
        
        for chunk_text in chunks:
            # Create a structured dictionary for each chunk
            chunk_data = {
                "id": f"chunk_{chunk_id_counter}",
                "text": chunk_text,
                "metadata": {
                    "source": os.path.basename(doc.metadata.get('source', 'Unknown')),
                    "page_number": doc.metadata.get('page', -1),
                    "ingestion_date": datetime.now().isoformat()
                    # You could add other metadata here, like a 'subject' label
                    # "subject": "Technology" 
                }
            }
            all_chunks_with_metadata.append(chunk_data)
            chunk_texts_for_embedding.append(chunk_text)
            chunk_id_counter += 1

    if not all_chunks_with_metadata:
        print("No chunks were created from the documents.")
        return
    
    print(f"Created a total of {len(all_chunks_with_metadata)} chunks.")

    # 3. Create Embeddings
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Creating embeddings for all chunks... (This may take a while)")
    embeddings = embedding_model.encode(chunk_texts_for_embedding, show_progress_bar=True)
    
    # 4. Build and Save FAISS Index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    # The index position corresponds directly to the position in `all_chunks_with_metadata`
    index.add(embeddings.astype('float32'))
    
    print(f"Saving FAISS index to '{FAISS_INDEX_PATH}'")
    faiss.write_index(index, FAISS_INDEX_PATH)

    # 5. Save Structured Chunk Data
    print(f"Saving structured chunk data with metadata to '{CHUNKS_JSON_PATH}'")
    with open(CHUNKS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks_with_metadata, f, indent=4)
    
    print("\n✅ Ingestion pipeline completed successfully!")

if __name__ == "__main__":
    run_ingestion()