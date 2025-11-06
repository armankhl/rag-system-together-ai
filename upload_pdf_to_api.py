# upload_pdf_to_api.py
import os
import requests
from langchain_community.document_loaders import PyPDFLoader

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
PDF_FILE_PATH = "data/bitcoin.pdf" # <-- The PDF you want to ingest

def upload_pdf(file_path: str):
    """
    Reads a PDF, extracts its text, and sends it to the /ingest API endpoint.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Loading text from '{file_path}'...")
    # Use PyPDFLoader to read the PDF and get its text content
    loader = PyPDFLoader(file_path)
    # The loader returns a list of Document objects, one for each page
    pages = loader.load()

    # We can ingest page by page for better metadata, or combine them.
    # Let's ingest the whole document as a single text for simplicity here.
    full_text = "\n".join([page.page_content for page in pages])
    
    # The file name will be our 'source' metadata
    source_name = os.path.basename(file_path)

    # --- Prepare the JSON payload for the API ---
    # The API expects a list of texts and a parallel list of metadatas
    payload = {
        "texts": [full_text],
        "metadatas": [{"source": source_name}]
    }

    print(f"Sending text from '{source_name}' to the ingestion API...")
    try:
        # Make the POST request to your running FastAPI server
        response = requests.post(f"{API_BASE_URL}/ingest", json=payload)
        
        # Raise an exception if the request was not successful
        response.raise_for_status()

        # Print the success response from the API
        print("\n--- API Response ---")
        print(response.json())
        print("--------------------")
        print(f"✅ Successfully ingested '{source_name}'.")

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Failed to connect to the API server at {API_BASE_URL}.")
        print(f"Please make sure your FastAPI server is running: uvicorn api:app --reload")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    upload_pdf(PDF_FILE_PATH)