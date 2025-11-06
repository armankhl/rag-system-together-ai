# ingestion_cli.py
import os
import nltk
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
import rag_core # Import the core logic

# Load environment variables
load_dotenv()

# --- Configuration ---
DATA_PATH = "data/"

def download_nltk_punkt():
    """Downloads the NLTK 'punkt' tokenizer if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' model...")
        nltk.download('punkt')
        print("'punkt' model downloaded successfully.")

def run_cli_ingestion():
    """
    Command-line interface for ingesting PDF documents from a local directory.
    This function now acts as a client to the rag_core.ingest_texts function.
    """
    # Ensure NLTK model for sentence tokenization is available
    download_nltk_punkt()

    # 1. Load Documents from the data directory
    print(f"Loading documents from '{DATA_PATH}'...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    if not documents:
        print(f"No documents found in '{DATA_PATH}'. Please add PDF files to this directory.")
        return

    print(f"Loaded {len(documents)} document(s).")

    # 2. Prepare texts and metadatas for the core ingestion function
    # The format must match what rag_core.ingest_texts expects: List[str] and List[Dict]
    texts_to_ingest = []
    metadatas_to_ingest = []

    for doc in documents:
        texts_to_ingest.append(doc.page_content)
        # We create the initial metadata dictionary here
        metadatas_to_ingest.append({
            "source": os.path.basename(doc.metadata.get('source', 'Unknown')),
            "page_number": doc.metadata.get('page', -1)
            # rag_core will add chunk-specific info like ingestion_date
        })

    # 3. Call the core ingestion logic
    print("Starting ingestion process using the core RAG logic...")
    try:
        result = rag_core.ingest_texts(
            texts=texts_to_ingest,
            metadatas=metadatas_to_ingest
        )
        
        # 4. Print the summary from the result
        print("\n--- Ingestion Summary ---")
        print(f"Message: {result.get('message')}")
        print(f"Chunks Added: {result.get('num_chunks_added')}")
        print(f"Total Chunks in DB: {result.get('total_chunks_in_db')}")
        print(f"Embedding Dimensions: {result.get('embedding_dimensions')}")
        print("--------------------------")
        print("\n✅ CLI ingestion complete.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during ingestion: {e}")

if __name__ == "__main__":
    # Before running, you might want to clear the old database if you're re-ingesting everything
    # This is a manual step for the CLI user.
    print("NOTE: This script will ADD documents to the existing vector store.")
    print("If you want to start fresh, please delete the 'vector_store' directory before running.")
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    run_cli_ingestion()