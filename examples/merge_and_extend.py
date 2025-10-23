#!/usr/bin/env python3
"""
Example: Merge existing videos and add new CSV data

This example demonstrates the complete workflow for:
1. Creating an initial video memory
2. Merging it with new video data
3. Adding CSV data
4. Building a combined video
5. Chatting with the combined memory

The key insight: You must build a NEW video file that contains ALL the data.
You cannot modify an existing video in-place.
"""

import sys
import os
from pathlib import Path
import csv
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid import MemvidEncoder, MemvidChat

# Use mp4v codec for maximum compatibility (works without FFmpeg)
VIDEO_CODEC = "mp4v"

def create_sample_csv(csv_path: Path, data: list):
    """Helper to create sample CSV file"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows(data)
    print(f"✓ Created sample CSV: {csv_path}")


def example_basic_workflow():
    """Example 1: Basic merge and extend workflow"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Merge and Extend Workflow")
    print("=" * 70)
    print()
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Create initial video with some data
    print("Step 1: Creating initial video memory...")
    print("-" * 70)
    
    encoder1 = MemvidEncoder()
    encoder1.add_chunks([
        "Python is a high-level programming language.",
        "JavaScript is used for web development.",
        "Java is platform-independent."
    ])
    
    video1 = output_dir / "initial.mp4"
    index1 = output_dir / "initial_index.json"
    
    encoder1.build_video(str(video1), str(index1), codec="mp4v", show_progress=True)
    print(f"✓ Created initial video: {video1}")
    print()
    
    # Step 2: Create a CSV with new data
    print("Step 2: Creating CSV with new data...")
    print("-" * 70)
    
    csv_path = output_dir / "programming_facts.csv"
    csv_data = [
        {"id": "1", "text": "C++ supports object-oriented programming."},
        {"id": "2", "text": "Ruby is known for elegant syntax."},
        {"id": "3", "text": "Go is designed for concurrent programming."}
    ]
    create_sample_csv(csv_path, csv_data)
    print()
    
    # Step 3: Create new encoder, merge video, add CSV
    print("Step 3: Merging video and CSV data...")
    print("-" * 70)
    
    encoder2 = MemvidEncoder()
    
    # Merge from existing video
    print(f"  Merging from: {video1}")
    encoder2.merge_from_video(str(video1), show_progress=True)
    
    # Add CSV data
    print(f"  Adding CSV: {csv_path}")
    encoder2.add_csv(str(csv_path), text_column="text")
    
    print(f"  Total chunks: {len(encoder2.chunks)}")
    print()
    
    # Step 4: Build combined video
    print("Step 4: Building combined video...")
    print("-" * 70)
    
    video2 = output_dir / "combined.mp4"
    index2 = output_dir / "combined_index.json"
    
    stats = encoder2.build_video(str(video2), str(index2), codec="mp4v", show_progress=True)
    print(f"✓ Created combined video: {video2}")
    print(f"  Total chunks in video: {stats['total_chunks']}")
    print(f"  Video size: {stats['video_size_mb']:.2f} MB")
    print()
    
    # Step 5: Test retrieval
    print("Step 5: Testing data retrieval...")
    print("-" * 70)
    
    chat = MemvidChat(str(video2), str(index2))
    
    # Query for old data
    print("  Searching for OLD data (Python)...")
    results_old = chat.search_context("Python programming", top_k=2)
    for i, result in enumerate(results_old, 1):
        print(f"    {i}. {result}")
    
    print()
    
    # Query for new data
    print("  Searching for NEW data (C++)...")
    results_new = chat.search_context("C++ object oriented", top_k=2)
    for i, result in enumerate(results_new, 1):
        print(f"    {i}. {result}")
    
    print()
    print("✓ Both old and new data are accessible!")
    print()


def example_convenience_method():
    """Example 2: Using the convenience method"""
    print("=" * 70)
    print("EXAMPLE 2: Using extend_and_rebuild() Convenience Method")
    print("=" * 70)
    print()
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare: Create sample video and CSV
    print("Preparation: Creating sample files...")
    print("-" * 70)
    
    # Create sample video
    encoder_sample = MemvidEncoder()
    encoder_sample.add_text("Machine learning uses algorithms to learn from data.")
    sample_video = output_dir / f"ml_basics.mp4"
    sample_index = output_dir / "ml_basics_index.json"
    encoder_sample.build_video(str(sample_video), str(sample_index), show_progress=False)
    print(f"✓ Created: {sample_video}")
    
    # Create sample CSV
    csv_path = output_dir / "ai_facts.csv"
    csv_data = [
        {"id": "1", "text": "Deep learning uses neural networks with many layers."},
        {"id": "2", "text": "Natural language processing helps computers understand text."}
    ]
    create_sample_csv(csv_path, csv_data)
    print()
    
    # Now use the convenience method
    print("Using extend_and_rebuild() method...")
    print("-" * 70)
    
    encoder = MemvidEncoder()
    encoder.add_text("Artificial intelligence simulates human intelligence.")
    
    # Single method call to merge, add, and build!
    combined_video = output_dir / f"ai_complete.mp4"
    combined_index = output_dir / "ai_complete_index.json"
    
    stats = encoder.extend_and_rebuild(
        output_video=str(combined_video),
        output_index=str(combined_index),
        video_files=[str(sample_video)],
        csv_files=[str(csv_path)],
        text_column="text",
        show_progress=True
    )
    
    print(f"✓ Created combined video: {combined_video}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print()
    
    # Test retrieval
    print("Testing retrieval from combined video...")
    print("-" * 70)
    
    chat = MemvidChat(str(combined_video), str(combined_index))
    results = chat.search_context("neural networks deep learning", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result[:100]}...")
    
    print()
    print("✓ All data sources merged successfully!")
    print()


def example_multiple_videos():
    """Example 3: Merging multiple videos"""
    print("=" * 70)
    print("EXAMPLE 3: Merging Multiple Videos")
    print("=" * 70)
    print()
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create multiple source videos
    print("Creating multiple source videos...")
    print("-" * 70)
    
    videos = []
    
    # Video 1: Programming languages
    enc1 = MemvidEncoder()
    enc1.add_text("Python, Java, and C++ are popular programming languages.")
    vid1 = output_dir / f"langs.mp4"
    idx1 = output_dir / "langs_index.json"
    enc1.build_video(str(vid1), str(idx1), show_progress=False)
    videos.append(str(vid1))
    print(f"✓ Created: {vid1}")
    
    # Video 2: Databases
    enc2 = MemvidEncoder()
    enc2.add_text("PostgreSQL, MySQL, and MongoDB are database systems.")
    vid2 = output_dir / f"databases.mp4"
    idx2 = output_dir / "databases_index.json"
    enc2.build_video(str(vid2), str(idx2), show_progress=False)
    videos.append(str(vid2))
    print(f"✓ Created: {vid2}")
    
    # Video 3: Frameworks
    enc3 = MemvidEncoder()
    enc3.add_text("Django, Flask, and FastAPI are Python web frameworks.")
    vid3 = output_dir / f"frameworks.mp4"
    idx3 = output_dir / "frameworks_index.json"
    enc3.build_video(str(vid3), str(idx3), show_progress=False)
    videos.append(str(vid3))
    print(f"✓ Created: {vid3}")
    
    print()
    
    # Merge all videos using from_videos classmethod
    print("Merging all videos into one...")
    print("-" * 70)
    
    # Method 1: Using from_videos classmethod
    merged_encoder = MemvidEncoder.from_videos(videos, show_progress=True)
    
    merged_video = output_dir / f"tech_complete.mp4"
    merged_index = output_dir / "tech_complete_index.json"
    
    stats = merged_encoder.build_video(str(merged_video), str(merged_index), show_progress=True)
    
    print(f"✓ Created merged video: {merged_video}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print()
    
    # Test comprehensive search
    print("Testing search across all merged videos...")
    print("-" * 70)
    
    chat = MemvidChat(str(merged_video), str(merged_index))
    
    queries = [
        "programming languages",
        "database systems",
        "web frameworks"
    ]
    
    for query in queries:
        print(f"  Query: '{query}'")
        results = chat.search_context(query, top_k=1)
        if results:
            print(f"    → {results[0][:80]}...")
        print()
    
    print("✓ All videos merged and searchable!")
    print()


def main():
    """Run all examples"""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  MEMVID: Merge Videos and Add CSV Data Examples".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("This script demonstrates how to:")
    print("  • Merge data from existing videos")
    print("  • Add new data from CSV files")
    print("  • Build combined videos with all data")
    print("  • Query the combined memories")
    print()
    
    try:
        # Run examples
        example_basic_workflow()
        example_convenience_method()
        example_multiple_videos()
        
        print("=" * 70)
        print("✓✓✓ ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓✓✓")
        print("=" * 70)
        print()
        print("Key Takeaways:")
        print("  1. Videos are immutable - you must build a NEW video with ALL data")
        print("  2. Use merge_from_video() to load chunks from existing videos")
        print("  3. Use add_csv() to add data from CSV files")
        print("  4. Use build_video() to create the final combined video")
        print("  5. Use extend_and_rebuild() for a convenient one-call workflow")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
