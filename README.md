# Command-Line RAG System with Together AI

This is a functional command-line RAG (Retrieval-Augmented Generation) system that answers questions based on PDF documents. It uses a local sentence-transformer model for embeddings and makes direct API calls to Together AI for language generation.

## Features

-   **Offline Ingestion Pipeline (`ingestion.py`):**
    -   Loads PDF documents from a `data/` directory.
    -   Chunks text using a sentence-windowing strategy for semantic coherence.
    -   Extracts and stores metadata (source file, page number).
    -   Creates vector embeddings locally using `sentence-transformers`.
    -   Builds and saves a FAISS vector index for efficient retrieval.
-   **Online Inference CLI (`main.py`):**
    -   Loads the pre-built vector store.
    -   Accepts a user query and an optional metadata filter (e.g., source document).
    -   Performs a two-stage retrieval (metadata filter -> vector search).
    -   Generates a final answer using a powerful LLM from Together AI.

## Setup

1.  **Clone the repository and navigate into it.**
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up your API key:**
    -   Copy the example environment file: `cp .env.example .env`
    -   Edit the `.env` file and add your `TOGETHER_API_KEY`.

## Usage

The system operates in two stages:

**1. Run the Ingestion Pipeline (Run this once):**
Place your PDF files in the `data/` directory and run the ingestion script. This will create a `vector_store/` directory with your indexed data.
```bash
python ingestion.py
```
**2. Start the Interactive Chat:
Once ingestion is complete, run the main script to start asking questions.
```bash
python main.py```