import os
from pathlib import Path
from dotenv import load_dotenv
from memvid import MemvidEncoder
from memvid.chat import MemvidChat

load_dotenv()

CODEC = "mp4v"
LLM_MODEL = "gemma3:1b"
MIN_RESPONSE_LENGTH = 50
CONTAINER_NAME = "memvid-test"
OLLAMA_BASE_URL = "http://vs-ollama-server.westeurope.cloudapp.azure.com"
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")


def upload_memory_to_blob():
    """Upload memory video and index to Azure Blob Storage"""
    print("=== STEP 1: Upload Memory to Blob Storage ===\n")

    datasets_dir = Path(__file__).parent / "datasets"
    csv1_path = datasets_dir / "articles_1.csv"
    csv2_path = datasets_dir / "articles_2.csv"

    if not csv1_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv1_path}")
    if not csv2_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv2_path}")

    encoder = MemvidEncoder(storage_connection=AZURE_STORAGE_CONNECTION_STRING)

    print("Loading")
    encoder.add_csv(str(csv1_path), text_column="text")
    encoder.add_csv(str(csv2_path), text_column="text")

    blob_video_path = f"blob://{CONTAINER_NAME}/articles_memory.mp4"
    blob_index_path = f"blob://{CONTAINER_NAME}/articles_memory"

    print(f"\nBuilding video and uploading to Azure")
    stats = encoder.build_video(
        output_file=blob_video_path,
        index_file=blob_index_path,
        codec=CODEC,
        show_progress=True,
    )

    print(f"\n✓ Upload complete!")
    print(f"  Container: {CONTAINER_NAME}")
    print(f"  Video: articles_memory.mp4 ({stats['video_size_mb']:.2f} MB)")
    print(f"  Index: articles_memory.faiss + articles_memory.json")
    print(f"  Total chunks: {stats['total_chunks']}")

    return blob_video_path, blob_index_path


def chat_with_blob_memory():
    """Chat using memory retrieved from Azure Blob Storage"""
    print("\n=== STEP 2: Chat with Blob Memory ===\n")

    blob_video_path = f"blob://{CONTAINER_NAME}/articles_memory.mp4"
    blob_index_path = f"blob://{CONTAINER_NAME}/articles_memory"

    print("Initializing chat (downloading index from blob)...")
    chat = MemvidChat(
        video_file=blob_video_path,
        index_file=blob_index_path,
        llm_provider="ollama",
        llm_model=LLM_MODEL,
        llm_base_url=OLLAMA_BASE_URL,
        storage_connection=AZURE_STORAGE_CONNECTION_STRING,
    )

    chat.start_session()
    print()

    queries = [
        "¿Qué es el CIFA?",
        "¿En qué puedo usar los residuos de tomate?",
        "Qué sabes decirme de la cochinilla acanalada?",
        "¿Cuál es la situación de la gripe aviar en España?",
        "¿Cuál es la situación del aceite de oliva?",
    ]

    failed_queries = []

    for i, query in enumerate(queries, 1):
        print(f"{query}")

        context = chat.search_context(query)

        if not context:
            print("    No context found")
            failed_queries.append((query, "No context retrieved"))
            continue

        try:
            response = chat.chat(query, stream=False)

            if not response or len(response) < MIN_RESPONSE_LENGTH:
                print("    Response too short or empty")
                failed_queries.append((query, "Response too short"))
                continue

            print(f"    Response: {response}\n")

        except Exception as e:
            print(f"    Error: {e}\n")
            failed_queries.append((query, str(e)))

    print("-" * 60)
    if failed_queries:
        print("FAILED: Some queries failed:")
        for query, reason in failed_queries:
            print(f"  - '{query}': {reason}")
        return False
    else:
        print("SUCCESS: All queries succeeded!")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    conversation_path = str(output_dir / "blob_chat_conversation.json")
    chat.export_conversation(conversation_path)

    return True


if __name__ == "__main__":
    try:
        success = chat_with_blob_memory()

        print("\n" + "=" * 60)
        if success:
            print("OK")
        else:
            print("NOT OK")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
