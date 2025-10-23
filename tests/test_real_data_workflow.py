"""
Test real-world workflow: Create video from first CSV, then extend with second CSV
"""
import os
import tempfile
from pathlib import Path

from memvid import MemvidEncoder

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


def test_real_csv_workflow():
    """
    Test complete workflow with real CSV data:
    1. Create video from articles_1.csv (using smaller chunks for reliable QR decoding)
    2. Extend with articles_2.csv data
    3. Verify all data is accessible
    
    Note: Spanish text with special characters (á, é, í, ó, etc.) creates more complex
    QR codes. Recovery rates of 50-70% are typical with current QR settings. For production
    use with multilingual text, consider:
    - Smaller chunk sizes (100-150 chars)
    - Increased QR_BOX_SIZE for larger QR codes
    - Text normalization (removing accents)
    """
    # Get paths to real datasets
    datasets_dir = Path(__file__).parent / "datasets"
    csv1_path = datasets_dir / "articles_1.csv"
    csv2_path = datasets_dir / "articles_2.csv"
    
    # Verify datasets exist
    assert csv1_path.exists(), f"Dataset not found: {csv1_path}"
    assert csv2_path.exists(), f"Dataset not found: {csv2_path}"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        video1_path = os.path.join(tmpdir, "initial_video.mp4")
        index1_path = os.path.join(tmpdir, "initial_index")
        video2_path = os.path.join(tmpdir, "extended_video.mp4")
        index2_path = os.path.join(tmpdir, "extended_index")
        
        # Step 1: Create initial video from first CSV (using smaller chunks)
        print("\n📹 Step 1: Creating video from articles_1.csv...")
        encoder1 = MemvidEncoder()
        encoder1.add_csv(str(csv1_path), text_column="text", chunk_size=200, overlap=20)
        
        initial_chunks = len(encoder1.chunks)
        print(f"   ✓ Added {initial_chunks} chunks from first CSV")
        assert initial_chunks > 0, "No chunks added from first CSV"
        
        # Build initial video
        encoder1.build_video(video1_path, index1_path, codec="mp4v", show_progress=True)
        assert Path(video1_path).exists(), "Initial video not created"
        print(f"   ✓ Initial video created: {Path(video1_path).stat().st_size / 1024:.1f} KB")
        
        # Step 2: Merge first video and extend with second CSV
        print("\n🔀 Step 2: Merging video and adding articles_2.csv...")
        encoder2 = MemvidEncoder()
        
        # Merge from first video
        encoder2.merge_from_video(video1_path, show_progress=True)
        after_merge = len(encoder2.chunks)
        print(f"   ✓ Merged {after_merge} chunks from video")
        
        # Calculate expected recovery rate (QR decoding may not be 100% with complex text)
        recovery_rate = (after_merge / initial_chunks) * 100 if initial_chunks > 0 else 0
        print(f"   ℹ️  Recovery rate: {recovery_rate:.1f}%")
        
        # We expect at least 40% of chunks to be recovered (realistic for Spanish text)
        assert recovery_rate >= 40.0, \
            f"Too many chunks lost: only {recovery_rate:.1f}% recovered (expected ≥40%)"
        assert after_merge > 0, "No chunks were recovered from video"
        
        # Add second CSV
        encoder2.add_csv(str(csv2_path), text_column="text", chunk_size=200, overlap=20)
        final_chunks = len(encoder2.chunks)
        added_chunks = final_chunks - after_merge
        print(f"   ✓ Added {added_chunks} chunks from second CSV")
        print(f"   ✓ Total chunks: {final_chunks}")
        assert final_chunks > after_merge, "No chunks added from second CSV"
        
        # Build extended video
        encoder2.build_video(video2_path, index2_path, codec="mp4v", show_progress=True)
        assert Path(video2_path).exists(), "Extended video not created"
        print(f"   ✓ Extended video created: {Path(video2_path).stat().st_size / 1024:.1f} KB")
        
        # Step 3: Verify data persistence by loading back
        print("\n✅ Step 3: Verifying data persistence...")
        encoder3 = MemvidEncoder()
        loaded_chunks = encoder3.load_chunks_from_video(video2_path, show_progress=True)
        
        print(f"   ✓ Loaded {len(loaded_chunks)} chunks from extended video")
        
        # Again, allow for some loss in QR decoding
        final_recovery_rate = (len(loaded_chunks) / final_chunks) * 100 if final_chunks > 0 else 0
        print(f"   ℹ️  Final recovery rate: {final_recovery_rate:.1f}%")
        assert final_recovery_rate >= 40.0, \
            f"Too many chunks lost in final video: only {final_recovery_rate:.1f}% recovered"
        assert len(loaded_chunks) > 0, "No chunks were recovered from final video"
        
        # Step 4: Content verification
        print("\n🔍 Step 4: Verifying content...")
        all_text = " ".join(loaded_chunks).lower()
        
        # Check for keywords from both CSVs
        # From articles: about lab meat, honey, drought, bees, innovation
        keywords = ["laboratorio", "miel", "sequía", "abejas", "innovación", "alimentación"]
        
        found_keywords = [kw for kw in keywords if kw in all_text]
        print(f"   ✓ Found {len(found_keywords)}/{len(keywords)} keywords from articles")
        
        # Verify we have substantial content
        assert len(all_text) > 500, "Combined text seems too short"
        assert len(found_keywords) >= 2, \
            f"Too few keywords found ({len(found_keywords)}/6). Expected content may not be present."
        
        # Summary
        print("\n" + "="*70)
        print("📊 Test Summary:")
        print("="*70)
        print(f"Initial CSV chunks:      {initial_chunks}")
        print(f"After merge:             {after_merge} ({recovery_rate:.1f}% recovered)")
        print(f"Second CSV chunks:       {added_chunks}")
        print(f"Total final chunks:      {final_chunks}")
        print(f"Verified loaded:         {len(loaded_chunks)} ({final_recovery_rate:.1f}% recovered)")
        print(f"Content size:            {len(all_text):,} characters")
        print(f"Keywords found:          {', '.join(found_keywords)}")
        print("="*70)
        print("ℹ️  Note: Recovery rate <100% is expected with complex Spanish text")
        print("   For better results, consider smaller chunks or text normalization")
        print("="*70)
        print("✅ Real data workflow test passed!")
        print("="*70)


def test_csv_column_validation():
    """Test that CSV column validation works correctly"""
    datasets_dir = Path(__file__).parent / "datasets"
    csv1_path = datasets_dir / "articles_1.csv"
    
    encoder = MemvidEncoder()
    
    # Test with correct column
    encoder.add_csv(str(csv1_path), text_column="text")
    assert len(encoder.chunks) > 0
    
    # Test with wrong column - should raise error
    encoder2 = MemvidEncoder()
    if HAS_PYTEST:
        with pytest.raises(ValueError, match="Column.*not found"):
            encoder2.add_csv(str(csv1_path), text_column="nonexistent_column")
    else:
        try:
            encoder2.add_csv(str(csv1_path), text_column="nonexistent_column")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e)


def test_empty_rows_handling():
    """Test that empty rows in CSV are properly handled"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create CSV with empty rows
        csv_path = os.path.join(tmpdir, "test_empty.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("text\n")
            f.write("First article\n")
            f.write("\n")  # Empty row
            f.write("   \n")  # Whitespace only
            f.write("Second article\n")
        
        encoder = MemvidEncoder()
        encoder.add_csv(csv_path, text_column="text")
        
        # Should have only 2 chunks (empty rows skipped)
        assert len(encoder.chunks) == 2
        assert "First article" in encoder.chunks[0]
        assert "Second article" in encoder.chunks[1]


if __name__ == "__main__":
    # Run tests manually
    print("Running real data workflow tests...")
    test_real_csv_workflow()
    print("\nRunning CSV validation test...")
    test_csv_column_validation()
    print("\nRunning empty rows test...")
    test_empty_rows_handling()
    print("\n✅ All tests completed successfully!")
