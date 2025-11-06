# chat_cli.py
import asyncio
from dotenv import load_dotenv
import rag_core # Import the core logic

# Load environment variables from .env file
load_dotenv()

async def chat_loop():
    """
    Main async function to run the interactive command-line chat loop.
    This function now acts as a client to the rag_core functions.
    """
    print("\n✅ RAG Command-Line Chat Interface is ready.")
    print("   (Type 'exit' to quit)")

    while True:
        # 1. Get user input
        question = input("\n> Enter your question: ")
        if question.lower() == 'exit':
            break

        source_filter = input("> Enter a source to filter by (e.g., 'bitcoin.pdf'), or press Enter to skip: ")
        # Ensure that an empty string becomes None
        source_filter = source_filter.strip() if source_filter.strip() else None

        # 2. Query the vector store using the core logic
        print("Retrieving relevant context...")
        try:
            context_chunks = rag_core.query_vector_store(
                query=question,
                top_k=5,
                source_filter=source_filter
            )

            if not context_chunks:
                print("\n💡 Answer:\nCould not find any relevant context to answer your question. Please try a different query or check your filters.")
                continue

            # 3. Generate an answer using the core logic
            print("Generating answer...")
            generation_result = await rag_core.generate_answer(
                query=question,
                context_chunks=context_chunks
            )
            
            # 4. Print the final answer
            print("\n💡 Answer:")
            print(generation_result.get('answer', 'No answer was generated.'))

            # Optionally, show the sources that were used
            print("\n--- Sources Used ---")
            sources = list(set(chunk['metadata']['source'] for chunk in context_chunks))
            for source in sources:
                print(f"- {source}")
            print("--------------------")

        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")


if __name__ == "__main__":
    # The rag_core module initializes its resources (like the ChromaDB client)
    # when it's imported, so we don't need an explicit setup call here.
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\nExiting chat.")
    except Exception as e:
        print(f"\nA critical error occurred: {e}")