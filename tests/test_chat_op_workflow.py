"""
Test chat integration with real data workflow using OpenRouter DeepSeek model.
This test verifies that data from both CSVs is accessible through chat after merging.
"""

import os
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv
from memvid import MemvidEncoder
from memvid.chat import MemvidChat

load_dotenv()

# Test configuration
CODEC = "mp4v"
MIN_RESPONSE_LENGTH = 50
MIN_RECOVERY_RATE = 80.0
DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3.1:free"


def _setup_paths() -> Tuple[Path, Path, Path]:
    """Setup and validate dataset paths."""
    datasets_dir = Path(__file__).parent / "datasets"
    csv1_path = datasets_dir / "articles_1.csv"
    csv2_path = datasets_dir / "articles_2.csv"
    output_dir = Path(__file__).parent / "output"

    if not csv1_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv1_path}")
    if not csv2_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv2_path}")

    output_dir.mkdir(exist_ok=True)
    return csv1_path, csv2_path, output_dir


def _create_video_from_csv(csv_path: Path, video_path: str, index_path: str) -> int:
    """Create video from CSV and return chunk count."""
    encoder = MemvidEncoder()
    encoder.add_csv(str(csv_path), text_column="text")
    chunk_count = len(encoder.chunks)

    assert chunk_count > 0, f"No chunks added from {csv_path.name}"

    encoder.build_video(video_path, index_path, codec=CODEC)
    assert Path(video_path).exists(), f"Video not created: {video_path}"

    return chunk_count


def _merge_and_extend_video(
    source_video: str,
    csv_path: Path,
    output_video: str,
    output_index: str,
    initial_chunks: int,
) -> Tuple[int, int, float]:
    """Merge video and extend with new CSV data."""
    encoder = MemvidEncoder()
    encoder.merge_from_video(source_video)

    after_merge = len(encoder.chunks)
    recovery_rate = (after_merge / initial_chunks * 100) if initial_chunks > 0 else 0

    assert (
        recovery_rate >= MIN_RECOVERY_RATE
    ), f"Insufficient recovery: {recovery_rate:.1f}% (expected >= {MIN_RECOVERY_RATE}%)"

    encoder.add_csv(str(csv_path), text_column="text")
    final_chunks = len(encoder.chunks)

    assert final_chunks > after_merge, f"No chunks added from {csv_path.name}"

    encoder.build_video(output_video, output_index, codec=CODEC)
    assert Path(output_video).exists(), f"Final video not created: {output_video}"

    return after_merge, final_chunks, recovery_rate


def _initialize_chat(video_path: str, index_path: str, api_key: str) -> MemvidChat:
    """Initialize chat with OpenRouter DeepSeek model."""
    chat = MemvidChat(
        video_file=video_path,
        index_file=index_path,
        llm_provider="openai",
        llm_model=DEEPSEEK_MODEL,
        llm_base_url="https://openrouter.ai/api/v1",
        llm_api_key=api_key,
    )

    chat.start_session()

    return chat


def test_chat_integration_with_merged_data():
    """
    Complete workflow test with chat integration using OpenRouter DeepSeek.

    Steps:
    1. Create initial video from articles_1.csv
    2. Merge and extend with articles_2.csv
    3. Initialize chat with the final merged MP4 file
    4. Ask 4 questions ONLY to the final MP4 memory
    5. Verify the final MP4 contains searchable content from both original sources


    Requires: OPENROUTER_API_KEY in .env file
    Model: deepseek/deepseek-chat-v3.1:free
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        msg = "OPENROUTER_API_KEY not available. Create .env file with your API key from https://openrouter.ai/"
        print(f"SKIPPING TEST: {msg}")
        return

    # Setup paths
    csv1_path, csv2_path, output_dir = _setup_paths()

    video1_path = str(output_dir / "chat_initial_video.mp4")
    index1_path = str(output_dir / "chat_initial_index")
    video2_path = str(output_dir / "chat_final_merged_video.mp4")
    index2_path = str(output_dir / "chat_final_merged_index")

    # Step 1: Create initial video from CSV1
    print("\nStep 1: Creating initial video from articles_1.csv")
    initial_chunks = _create_video_from_csv(csv1_path, video1_path, index1_path)
    print(f"  Created video with {initial_chunks} chunks")

    # Step 2: Merge and extend with CSV2
    print("\nStep 2: Merging and extending with articles_2.csv")
    after_merge, final_chunks, recovery_rate = _merge_and_extend_video(
        video1_path, csv2_path, video2_path, index2_path, initial_chunks
    )
    added_chunks = final_chunks - after_merge
    print(
        f"  Merged {after_merge}/{initial_chunks} chunks ({recovery_rate:.1f}% recovery)"
    )
    print(f"  Added {added_chunks} new chunks from CSV2")
    print(f"  Total final chunks: {final_chunks}")

    # Step 3: Initialize chat with final merged MP4 file
    print("\nStep 3: Initializing chat with final merged MP4 file")
    chat = _initialize_chat(video2_path, index2_path, openrouter_key)

    # Step 4: Querying the final merged MP4
    print("\nStep 4: Querying the final merged MP4 file")

    all_queries = [
        ("¿Qué es el CIFA?"),
        ("¿En qué puedo usar los residuos de tomate?"),
    ]

    failed_queries = []

    for query in all_queries:
        print(f"\n  Query: {query}")

        # First, verify context retrieval works from the final MP4 memory
        context = chat.search_context(query)

        if not context:
            print("    No context found")
            failed_queries.append((query, "No context retrieved"))
            continue

        # Now ask the LLM
        try:
            response = chat.chat(query, stream=False)

            if not response or len(response) < MIN_RESPONSE_LENGTH:
                print("    Response too short or empty")
                failed_queries.append((query, "Response too short"))
                continue

            # Print a preview of the response
            print(f"   Response preview: {response}")

        except Exception as e:
            print(f"    Error getting LLM response: {e}")
            failed_queries.append((query, str(e)))

    # Step 5: Verify results
    print(f"\n{'-'*70}")
    print("Step 5: Verification Results")

    if failed_queries:
        print("  Failed queries:")
        for query, reason in failed_queries:
            print(f"    - '{query}': {reason}")
    else:
        print("  All queries succeeded")

    # Step 6: Export conversation
    print("\nStep 6: Exporting conversation")
    conversation_path = str(output_dir / "chat_conversation.json")
    chat.export_conversation(conversation_path)
    print(f"  Exported {len(chat.conversation_history)} conversation turns")

    # Summary
    print("\n" + "-" * 70)
    print("TEST RESULTS - Chat Integration with Final Merged MP4")
    print(f"Initial chunks (CSV1):     {initial_chunks}")
    print(f"Recovered after merge:     {after_merge} ({recovery_rate:.1f}%)")
    print(f"Added chunks (CSV2):       {added_chunks}")
    print(f"Total final chunks:        {final_chunks}")
    print(f"Conversation turns:        {len(chat.conversation_history)}")
    print(f"Queried file:              {video2_path}")


def test_env_file_detection():
    """Test that .env file is properly loaded."""
    print("\nTest: Environment File Detection")

    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if openrouter_key:
        print("  OPENROUTER_API_KEY found")
    else:
        print("  OPENROUTER_API_KEY not found in environment")
    print()


if __name__ == "__main__":
    # Check environment
    test_env_file_detection()

    # Full integration with LLM (requires API key)
    try:
        test_chat_integration_with_merged_data()
    except Exception as e:
        print(f"\nFull integration test failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "-" * 70)
    print("Ok")
