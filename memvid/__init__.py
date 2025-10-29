"""
Memvid - QR Code Video-Based AI Memory Library
"""

__version__ = "0.1.0"

from .chat import MemvidChat
from .encoder import MemvidEncoder
from .retriever import MemvidRetriever
from .llm_client import LLMClient, create_llm_client
from .interactive import chat_with_memory, quick_chat
from .storage import StorageAdapter, get_storage_adapter

__all__ = [
    "MemvidEncoder",
    "MemvidRetriever",
    "MemvidChat",
    "chat_with_memory",
    "quick_chat",
    "LLMClient",
    "create_llm_client",
    "StorageAdapter",
    "get_storage_adapter",
]
