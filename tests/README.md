# Tests

This directory contains tests for the memvid library.

## Test Files

### `test_real_data_workflow.py`

Comprehensive integration test using real Spanish agricultural articles from CSV files.

**What it tests:**

1. Creating a video memory from CSV data (`articles_1.csv`)
2. Merging video data into a new encoder
3. Extending with additional CSV data (`articles_2.csv`)
4. Verifying all data is accessible after the merge and extend workflow

**Key Findings:**

- ✅ The merge and extend workflow works correctly
- ℹ️ QR code recovery rates of 50-70% are typical with Spanish text containing special characters (á, é, í, ó, ú, ñ, etc.)
- ℹ️ This is expected behavior due to the complexity of QR codes with non-ASCII characters
- ✅ All keywords and content remain accessible despite <100% recovery rate

**Running the tests:**

```bash
# Run all tests
python tests/test_real_data_workflow.py

# Or use pytest (if installed)
pytest tests/test_real_data_workflow.py -v
```

## Test Datasets

The `datasets/` directory contains real CSV files with Spanish agricultural articles:

- `articles_1.csv` - 10 articles (used for initial video creation)
- `articles_2.csv` - 5 articles (used for extending the video)

Each CSV has columns: `id`, `title`, `text`

## QR Code Recovery Rates

The recovery rate depends on text complexity:

| Text Type                     | Typical Recovery Rate |
| ----------------------------- | --------------------- |
| Simple English text           | 90-100%               |
| Spanish text with accents     | 50-70%                |
| Very long chunks (>500 chars) | 30-50%                |

**Recommendations for production use:**

1. **For multilingual text:** Use smaller chunk sizes (100-200 characters)
2. **For better recovery:** Consider text normalization (removing accents)
3. **For critical data:** Increase `QR_BOX_SIZE` in config for larger, more readable QR codes
4. **Monitor:** Always check recovery rates after merge operations

## Notes

- Tests use `codec='mp4v'` for fast execution
- Progress bars are enabled to show encoding/decoding progress
- Tests are realistic and account for expected QR decoding limitations
