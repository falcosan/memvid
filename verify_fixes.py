#!/usr/bin/env python3
"""
Quick verification test for merge_from_video and add_csv fixes
"""

import sys
import os
import tempfile
from pathlib import Path
import csv

sys.path.insert(0, str(Path(__file__).parent))

from memvid import MemvidEncoder, MemvidRetriever

def run_verification():
    """Quick test to verify the fixes work"""
    
    print("🧪 Running verification test...")
    print()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Create initial video
        print("1. Creating initial video...")
        encoder1 = MemvidEncoder()
        encoder1.add_chunks([
            "The Eiffel Tower is in Paris, France.",
            "The Colosseum is in Rome, Italy.",
        ])
        
        video1 = temp_path / "initial.mp4"
        index1 = temp_path / "initial_index.json"
        encoder1.build_video(str(video1), str(index1), codec="mp4v", show_progress=False)
        print(f"   ✓ Created with {len(encoder1.chunks)} chunks")
        
        # Step 2: Create CSV
        print("2. Creating CSV with new data...")
        csv_file = temp_path / "landmarks.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["name", "info"])
            writer.writeheader()
            writer.writerows([
                {"name": "Big Ben", "info": "Big Ben is a clock tower in London, England."},
                {"name": "Sagrada Familia", "info": "Sagrada Familia is a basilica in Barcelona, Spain."},
            ])
        print(f"   ✓ Created CSV with 2 rows")
        
        # Step 3: Merge + Add + Build
        print("3. Merging video and adding CSV...")
        encoder2 = MemvidEncoder()
        encoder2.merge_from_video(str(video1), show_progress=False)
        print(f"   ✓ Merged: {len(encoder2.chunks)} chunks loaded")
        
        encoder2.add_csv(str(csv_file), text_column="info")
        print(f"   ✓ Added CSV: {len(encoder2.chunks)} total chunks")
        
        video2 = temp_path / "combined.mp4"
        index2 = temp_path / "combined_index.json"
        stats = encoder2.build_video(str(video2), str(index2), codec="mp4v", show_progress=False)
        print(f"   ✓ Built video: {stats['total_chunks']} chunks encoded")
        
        # Step 4: Verify retrieval
        print("4. Verifying data retrieval...")
        retriever = MemvidRetriever(str(video2), str(index2))
        
        # Test old data
        results_old = retriever.search("Eiffel Tower Paris", top_k=1)
        old_found = any("Eiffel" in r and "Paris" in r for r in results_old)
        print(f"   {'✓' if old_found else '✗'} OLD data (Eiffel Tower): {'Found' if old_found else 'NOT FOUND'}")
        
        # Test new data
        results_new = retriever.search("Big Ben London", top_k=1)
        new_found = any("Big Ben" in r and "London" in r for r in results_new)
        print(f"   {'✓' if new_found else '✗'} NEW data (Big Ben): {'Found' if new_found else 'NOT FOUND'}")
        
        # Step 5: Additional retrieval tests with different queries
        print("5. Additional retrieval verification...")
        
        # Test with more specific queries
        results_italy = retriever.search("Colosseum Rome Italy", top_k=2)
        italy_found = any("Colosseum" in r and "Rome" in r for r in results_italy)
        print(f"   {'✓' if italy_found else '✗'} OLD data (Colosseum): {'Found' if italy_found else 'NOT FOUND'}")
        
        results_spain = retriever.search("Sagrada Familia Barcelona", top_k=2)
        spain_found = any("Sagrada Familia" in r and "Barcelona" in r for r in results_spain)
        print(f"   {'✓' if spain_found else '✗'} NEW data (Sagrada Familia): {'Found' if spain_found else 'NOT FOUND'}")
        
        # Verify index stats
        stats = retriever.get_stats()
        chunks_in_index = stats['index_stats']['total_chunks']
        print(f"   {'✓' if chunks_in_index == 4 else '✗'} Index contains {chunks_in_index} chunks (expected 4)")
        
        # Final verdict
        print()
        all_ok = old_found and new_found and italy_found and spain_found and chunks_in_index == 4
        
        if all_ok:
            print("✅ ✅ ✅  VERIFICATION PASSED  ✅ ✅ ✅")
            print()
            print("The fixes are working correctly:")
            print("  • merge_from_video() properly loads chunks from existing videos")
            print("  • add_csv() properly adds new data from CSV files")
            print("  • build_video() includes all chunks in video+index")
            print("  • Retriever can access both old and new data")
            print(f"  • Total chunks: {chunks_in_index}/4 ✓")
            return 0
        else:
            print("❌ ❌ ❌  VERIFICATION FAILED  ❌ ❌ ❌")
            print()
            print("Issues detected:")
            if not old_found:
                print("  ✗ OLD data (Eiffel Tower) not found in retriever")
            if not new_found:
                print("  ✗ NEW data (Big Ben) not found in retriever")
            if not italy_found:
                print("  ✗ OLD data (Colosseum) not found in retriever")
            if not spain_found:
                print("  ✗ NEW data (Sagrada Familia) not found in retriever")
            if chunks_in_index != 4:
                print(f"  ✗ Index has {chunks_in_index} chunks, expected 4")
            return 1

if __name__ == "__main__":
    try:
        sys.exit(run_verification())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
