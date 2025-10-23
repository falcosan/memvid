"""
Test chat integration with real data workflow using OpenRouter DeepSeek model.
This test verifies that data from both CSVs is accessible through chat after merging.
"""
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from memvid import MemvidEncoder
from memvid.chat import MemvidChat

load_dotenv()


# Test configuration
CHUNK_SIZE = 200
OVERLAP = 20
CODEC = "mp4v"
MIN_RESPONSE_LENGTH = 50
MIN_RECOVERY_RATE = 90.0
SEARCH_TOP_K = 5
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
    encoder.add_csv(str(csv_path), text_column="text", chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    chunk_count = len(encoder.chunks)
    
    assert chunk_count > 0, f"No chunks added from {csv_path.name}"
    
    encoder.build_video(video_path, index_path, codec=CODEC, show_progress=False)
    assert Path(video_path).exists(), f"Video not created: {video_path}"
    
    return chunk_count


def _merge_and_extend_video(
    source_video: str, 
    csv_path: Path, 
    output_video: str, 
    output_index: str,
    initial_chunks: int
) -> Tuple[int, int, float]:
    """Merge video and extend with new CSV data."""
    encoder = MemvidEncoder()
    encoder.merge_from_video(source_video, show_progress=False)
    
    after_merge = len(encoder.chunks)
    recovery_rate = (after_merge / initial_chunks * 100) if initial_chunks > 0 else 0
    
    assert recovery_rate >= MIN_RECOVERY_RATE, \
        f"Insufficient recovery: {recovery_rate:.1f}% (expected >= {MIN_RECOVERY_RATE}%)"
    
    encoder.add_csv(str(csv_path), text_column="text", chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    final_chunks = len(encoder.chunks)
    added_chunks = final_chunks - after_merge
    
    assert final_chunks > after_merge, f"No chunks added from {csv_path.name}"
    
    encoder.build_video(output_video, output_index, codec=CODEC, show_progress=False)
    assert Path(output_video).exists(), f"Final video not created: {output_video}"
    
    return after_merge, final_chunks, recovery_rate


def _initialize_chat(video_path: str, index_path: str, api_key: str) -> MemvidChat:
    """Initialize chat with OpenRouter DeepSeek model."""
    try:
        from openai import OpenAI
    except ImportError as e:
        error_msg = f"OpenAI library not available: {e}"
        print(f"SKIPPING: {error_msg}")
        raise ImportError(error_msg)
    
    openrouter_client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    chat = MemvidChat(
        video_file=video_path,
        index_file=index_path,
        llm_provider='openai',
        llm_model=DEEPSEEK_MODEL,
        llm_api_key=api_key
    )
    
    chat.llm_client.provider.client = openrouter_client
    chat.llm_client.provider.model = DEEPSEEK_MODEL
    chat.start_session()
    
    return chat


def _execute_queries(chat: MemvidChat, queries: List[str], print_responses: bool = True) -> List[str]:
    """Execute multiple queries and return responses."""
    responses = []
    for i, query in enumerate(queries, 1):
        try:
            print(f"\n  Query {i}: {query}")
            response = chat.chat(query, stream=False)
            
            if response is None:
                raise ValueError(f"Received None response for query: '{query}'")
            if len(response) <= MIN_RESPONSE_LENGTH:
                raise ValueError(f"Response too short for query '{query}': {len(response)} chars")
            
            if print_responses:
                print(f"  Response: {response}")
            
            responses.append(response)
        except Exception as e:
            print(f"\n  Error executing query '{query}': {e}")
            raise
    return responses


def _verify_keywords(context_chunks: List[str], expected_keywords: List[str]) -> List[str]:
    """Verify expected keywords are present in context chunks."""
    all_context = " ".join(context_chunks).lower()
    found_keywords = [kw for kw in expected_keywords if kw in all_context]
    
    assert len(found_keywords) >= 1, \
        f"Insufficient keywords found. Expected: {expected_keywords}, Found: {found_keywords}"
    
    return found_keywords


def test_chat_integration_with_merged_data():
    """
    Complete workflow test with chat integration using OpenRouter DeepSeek.
    
    Steps:
    1. Create video from articles_1.csv
    2. Merge video and extend with articles_2.csv
    3. Use chat (DeepSeek model) to query both datasets
    4. Verify data from both CSVs is accessible via chat
    
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
    video2_path = str(output_dir / "chat_final_video.mp4")
    index2_path = str(output_dir / "chat_final_index")
    
    # Step 1: Create initial video
    print("\nStep 1: Creating video from articles_1.csv")
    initial_chunks = _create_video_from_csv(csv1_path, video1_path, index1_path)
    print(f"  Created video with {initial_chunks} chunks")
    
    # Step 2: Merge and extend
    print("\nStep 2: Merging video and extending with articles_2.csv")
    after_merge, final_chunks, recovery_rate = _merge_and_extend_video(
        video1_path, csv2_path, video2_path, index2_path, initial_chunks
    )
    added_chunks = final_chunks - after_merge
    print(f"  Merged {after_merge}/{initial_chunks} chunks ({recovery_rate:.1f}% recovery)")
    print(f"  Added {added_chunks} new chunks, total: {final_chunks}")
    
    # Step 3: Initialize chat
    print("\nStep 3: Initializing chat with DeepSeek model")
    chat = _initialize_chat(video2_path, index2_path, openrouter_key)
    print(f"  Chat initialized with {DEEPSEEK_MODEL}")
    
    # Step 4 & 5: Execute queries
    print("\nStep 4: Testing queries from CSV1")
    queries_csv1 = [
        "¿Qué información hay sobre la miel?",
        "¿Qué se menciona sobre laboratorios?",
    ]
    responses_csv1 = _execute_queries(chat, queries_csv1)
    print(f"\n  Executed {len(queries_csv1)} queries successfully")
    
    print("\nStep 5: Testing queries from CSV2")
    queries_csv2 = [
        "¿Qué información hay sobre la sequía?",
        "¿Qué se menciona sobre microorganismos?",
    ]
    responses_csv2 = _execute_queries(chat, queries_csv2)
    print(f"\n  Executed {len(queries_csv2)} queries successfully")
    
    # Step 6 & 7: Verify context retrieval
    print("\nStep 6: Verifying context retrieval from both datasets")
    context_csv1 = chat.search_context("miel laboratorio", top_k=SEARCH_TOP_K)
    context_csv2 = chat.search_context("sequía microorganismos", top_k=SEARCH_TOP_K)
    
    assert len(context_csv1) > 0, "No context found for CSV1 keywords"
    assert len(context_csv2) > 0, "No context found for CSV2 keywords"
    print(f"  Found {len(context_csv1)} chunks for CSV1, {len(context_csv2)} chunks for CSV2")
    
    print("\nStep 7: Verifying keywords presence")
    keywords_csv1 = ["laboratorio", "miel"]
    keywords_csv2 = ["sequía", "microorganismos"]
    
    found_csv1 = _verify_keywords(context_csv1, keywords_csv1)
    found_csv2 = _verify_keywords(context_csv2, keywords_csv2)
    print(f"  CSV1 keywords: {', '.join(found_csv1)}")
    print(f"  CSV2 keywords: {', '.join(found_csv2)}")
    
    # Step 8: Export conversation
    print("\nStep 8: Exporting conversation")
    conversation_path = str(output_dir / "chat_conversation.json")
    chat.export_conversation(conversation_path)
    print(f"  Exported {len(chat.conversation_history)} conversation turns")
    
    # Summary
    print("\n" + "="*80)
    print("TEST PASSED - Chat Integration with Merged Data")
    print("="*80)
    print(f"Initial chunks (CSV1):     {initial_chunks}")
    print(f"Recovered after merge:     {after_merge} ({recovery_rate:.1f}%)")
    print(f"Added chunks (CSV2):       {added_chunks}")
    print(f"Total final chunks:        {final_chunks}")
    print(f"Total queries executed:    {len(queries_csv1) + len(queries_csv2)}")
    print(f"Conversation turns:        {len(chat.conversation_history)}")
    print("="*80)


def test_chat_context_only_mode():
    """
    Test chat in context-only mode without LLM.
    Verifies that semantic search works independently of LLM availability.
    """
    csv1_path, _, output_dir = _setup_paths()
    
    video_path = str(output_dir / "chat_context_only_video.mp4")
    index_path = str(output_dir / "chat_context_only_index")
    
    print("\nTest: Chat Context-Only Mode (No LLM Required)")
    
    # Create video
    chunk_count = _create_video_from_csv(csv1_path, video_path, index_path)
    print(f"  Created video with {chunk_count} chunks")
    
    # Initialize chat without valid API key
    chat = MemvidChat(
        video_file=video_path,
        index_file=index_path,
        llm_provider='openai',
        llm_api_key='invalid_test_key'
    )
    
    # Test semantic search
    results = chat.search_context("miel laboratorio", top_k=3)
    assert len(results) > 0, "No context found"
    
    all_text = " ".join(results).lower()
    assert "miel" in all_text or "laboratorio" in all_text, \
        "Expected keywords not found in context"
    
    print(f"  Found {len(results)} context chunks with expected keywords")
    print("  TEST PASSED - Context-only mode works without LLM\n")


def test_env_file_detection():
    """Test that .env file is properly loaded."""
    print("\nTest: Environment File Detection")
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if openrouter_key:
        masked_key = openrouter_key[:8] + "..." + openrouter_key[-4:] if len(openrouter_key) > 12 else "***"
        print(f"  OPENROUTER_API_KEY found: {masked_key}")
        print(f"  Key length: {len(openrouter_key)} characters")
        print("  Ready to run full chat integration test")
    else:
        print("  OPENROUTER_API_KEY not found in environment")
        print("  To enable full testing, create a .env file with:")
        print("    OPENROUTER_API_KEY=your-api-key-here")
        print("  Get your free key at: https://openrouter.ai/")
    print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MEMVID CHAT INTEGRATION TEST SUITE")
    print("="*80)
    
    # Test 1: Check environment
    test_env_file_detection()
    
    # Test 2: Context-only mode (always runs)
    try:
        test_chat_context_only_mode()
    except Exception as e:
        print(f"\nContext-only test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Full integration with LLM (requires API key)
    try:
        test_chat_integration_with_merged_data()
    except Exception as e:
        print(f"\nFull integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Test suite completed")
    print("="*80 + "\n")

