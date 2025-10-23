#!/usr/bin/env python3
"""
Test script to reproduce the merge_from_video and add_csv issue
"""

import os
import sys
import tempfile
from pathlib import Path
import csv

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from memvid import MemvidEncoder, MemvidChat

def test_workflow():
    """Test the complete workflow: create video -> merge -> add CSV -> build -> chat"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print("=" * 60)
        print("STEP 1: Create initial video with some data")
        print("=" * 60)
        
        # Create initial video
        encoder1 = MemvidEncoder()
        initial_chunks = [
            "The capital of France is Paris.",
            "The capital of Germany is Berlin.",
            "The capital of Italy is Rome."
        ]
        encoder1.add_chunks(initial_chunks)
        
        video1_path = temp_path / "initial.mp4"
        index1_path = temp_path / "initial_index.json"
        
        print(f"Building initial video with {len(encoder1.chunks)} chunks...")
        stats1 = encoder1.build_video(str(video1_path), str(index1_path), show_progress=False)
        print(f"✓ Initial video created: {stats1['total_chunks']} chunks")
        print()
        
        print("=" * 60)
        print("STEP 2: Create CSV with new data")
        print("=" * 60)
        
        # Create CSV file
        csv_path = temp_path / "cities.csv"
        csv_data = [
            {"city": "Madrid", "info": "The capital of Spain is Madrid."},
            {"city": "Lisbon", "info": "The capital of Portugal is Lisbon."},
            {"city": "Athens", "info": "The capital of Greece is Athens."}
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["city", "info"])
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"✓ Created CSV with {len(csv_data)} rows at {csv_path}")
        print()
        
        print("=" * 60)
        print("STEP 3: Merge video + add CSV data")
        print("=" * 60)
        
        # Create new encoder and merge old video + CSV
        encoder2 = MemvidEncoder()
        
        print(f"Chunks before merge: {len(encoder2.chunks)}")
        encoder2.merge_from_video(str(video1_path), show_progress=False)
        print(f"Chunks after merge: {len(encoder2.chunks)}")
        
        encoder2.add_csv(str(csv_path), text_column="info", chunk_size=200, overlap=0)
        print(f"Chunks after CSV: {len(encoder2.chunks)}")
        print()
        
        # Debug: print what chunks we have
        print("Chunks in encoder2:")
        for i, chunk in enumerate(encoder2.chunks[:10]):  # Show first 10
            print(f"  {i+1}. {chunk[:80]}...")
        print()
        
        print("=" * 60)
        print("STEP 4: Build final video")
        print("=" * 60)
        
        video2_path = temp_path / "merged.mp4"
        index2_path = temp_path / "merged_index.json"
        
        print(f"Building merged video with {len(encoder2.chunks)} chunks...")
        stats2 = encoder2.build_video(str(video2_path), str(index2_path), show_progress=False)
        print(f"✓ Merged video created: {stats2['total_chunks']} chunks")
        print()
        
        print("=" * 60)
        print("STEP 5: Test retrieval from merged video")
        print("=" * 60)
        
        # Test if we can retrieve data from the merged video
        from memvid import MemvidRetriever
        
        retriever = MemvidRetriever(str(video2_path), str(index2_path))
        stats = retriever.get_stats()
        print(f"Retriever loaded: {stats['index_stats']['total_chunks']} chunks in index")
        print()
        
        # Search for old data (from initial video)
        print("Testing search for OLD data (Paris):")
        results_old = retriever.search("capital of France", top_k=2)
        for i, result in enumerate(results_old):
            print(f"  {i+1}. {result}")
        print()
        
        # Search for new data (from CSV)
        print("Testing search for NEW data (Madrid):")
        results_new = retriever.search("capital of Spain", top_k=2)
        for i, result in enumerate(results_new):
            print(f"  {i+1}. {result}")
        print()
        
        print("=" * 60)
        print("STEP 6: Test chat with merged video")
        print("=" * 60)
        
        # Try chat (without LLM, just context retrieval)
        chat = MemvidChat(
            video_file=str(video2_path),
            index_file=str(index2_path),
            llm_provider='google',  # Will fail gracefully if no API key
        )
        
        print("Testing context retrieval for OLD data:")
        context_old = chat.search_context("What is the capital of France?", top_k=3)
        print(f"Found {len(context_old)} context chunks")
        for i, ctx in enumerate(context_old):
            print(f"  {i+1}. {ctx}")
        print()
        
        print("Testing context retrieval for NEW data:")
        context_new = chat.search_context("What is the capital of Spain?", top_k=3)
        print(f"Found {len(context_new)} context chunks")
        for i, ctx in enumerate(context_new):
            print(f"  {i+1}. {ctx}")
        print()
        
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        total_expected = len(initial_chunks) + len(csv_data)
        
        print(f"Expected total chunks: {total_expected}")
        print(f"  - From initial video: {len(initial_chunks)}")
        print(f"  - From CSV: {len(csv_data)}")
        print()
        print(f"Actual chunks in encoder: {len(encoder2.chunks)}")
        print(f"Actual chunks in built video: {stats2['total_chunks']}")
        print(f"Actual chunks in index: {stats['index_stats']['total_chunks']}")
        print()
        
        # Check if we found the right data
        old_data_found = any("Paris" in r for r in results_old)
        new_data_found = any("Madrid" in r for r in results_new)
        
        print(f"OLD data (Paris) searchable: {'✓ YES' if old_data_found else '✗ NO'}")
        print(f"NEW data (Madrid) searchable: {'✓ YES' if new_data_found else '✗ NO'}")
        print()
        
        if len(encoder2.chunks) == total_expected and old_data_found and new_data_found:
            print("✓✓✓ SUCCESS: All data is accessible! ✓✓✓")
            return True
        else:
            print("✗✗✗ FAILURE: Data is missing or not accessible! ✗✗✗")
            return False

if __name__ == "__main__":
    print("Testing merge_from_video + add_csv workflow...")
    print()
    
    try:
        success = test_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗✗✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
