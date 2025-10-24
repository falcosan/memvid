# Memvid

A Python library for encoding text data into QR code videos and retrieving it using semantic search. Memvid allows you to store large text corpora as video files and query them using natural language.

## Overview

Memvid converts text documents into QR code frames, encodes them as video files, and builds a semantic search index. This enables efficient storage and retrieval of text data through video encoding, with support for conversational AI interfaces.

The library consists of three main components:

- **Encoder**: Converts text into QR code videos
- **Retriever**: Searches and extracts text from videos
- **Chat**: Provides conversational interface with LLM integration

## Installation

```bash
pip install lib-memvid
```

For LLM support (OpenAI, Google, Anthropic):

```bash
pip install lib-memvid[llm]
```

For EPUB support:

```bash
pip install lib-memvid[epub]
```

## Quick Start

### Encoding Text to Video

```python
from memvid import MemvidEncoder

# Create encoder and add text
encoder = MemvidEncoder()
encoder.add_text("Your text content here")

# Build video and index
encoder.build_video(
    output_file="output/memory.mp4",
    index_file="output/memory.json"
)
```

### Searching the Video

```python
from memvid import MemvidRetriever

# Initialize retriever
retriever = MemvidRetriever(
    video_file="output/memory.mp4",
    index_file="output/memory.json"
)

# Search for relevant content
results = retriever.search("your search query", top_k=5)
for result in results:
    print(result)
```

### Chat Interface

```python
from memvid import MemvidChat

# Initialize chat with LLM provider
chat = MemvidChat(
    video_file="output/memory.mp4",
    index_file="output/memory.json",
    llm_provider="google",  # or "openai", "anthropic"
    llm_api_key="your-api-key"
)

# Start interactive chat
chat.interactive_chat()
```

## Features

### Data Input

Load text from multiple sources:

```python
encoder = MemvidEncoder()

# From text file
encoder.add_text(open("document.txt").read())

# From PDF
encoder.add_pdf("document.pdf")

# From EPUB
encoder.add_epub("book.epub")

# From CSV
encoder.add_csv("data.csv", text_column="content")

# Merge from existing videos
encoder.merge_from_video("existing.mp4")
```

### Video Codecs

Memvid supports multiple video codecs with configurable quality settings:

```python
# H.265 (HEVC) - best compression
encoder.build_video("output.mkv", "index.json", codec="h265")

# H.264 (AVC) - wide compatibility
encoder.build_video("output.mkv", "index.json", codec="h264")

# MP4V - fast encoding
encoder.build_video("output.mp4", "index.json", codec="mp4v")
```

### Search Capabilities

Retrieve text with semantic search:

```python
retriever = MemvidRetriever("memory.mp4", "memory.json")

# Basic search
results = retriever.search("machine learning", top_k=5)

# Search with metadata
results = retriever.search_with_metadata("neural networks", top_k=3)
for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")

# Get specific chunk
chunk = retriever.get_chunk_by_id(42)

# Get context window around a chunk
context = retriever.get_context_window(chunk_id=42, window_size=2)
```

### LLM Integration

Chat interface supports multiple LLM providers:

```python
chat = MemvidChat(
    video_file="memory.mp4",
    index_file="memory.json",
    llm_provider="google",  # "openai", "anthropic"
    llm_model="gemini-2.0-flash-exp"
)

# Single query
response = chat.chat("What is the main topic?")

# Streaming response
chat.chat("Explain the concept", stream=True)

# Export conversation
chat.export_conversation("conversation.json")
```

### Configuration

Customize encoding and retrieval settings:

```python
from memvid import MemvidEncoder
from memvid.config import get_default_config

config = get_default_config()
config["chunking"]["chunk_size"] = 1024
config["chunking"]["overlap"] = 64
config["retrieval"]["top_k"] = 10

encoder = MemvidEncoder(config=config)
```

## License

MIT License
