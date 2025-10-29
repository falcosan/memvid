"""
MemvidChat - Enhanced conversational interface with multi-provider LLM support
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from .llm_client import LLMClient
from .retriever import MemvidRetriever
from .config import get_default_config
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class MemvidChat:
    """Enhanced MemvidChat with multi-provider LLM support"""

    def __init__(
        self,
        video_file: str,
        index_file: str,
        llm_provider: str = "google",
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """Initialize MemvidChat with flexible LLM provider support"""
        self.video_file = video_file
        self.index_file = index_file
        self.config = config or get_default_config()

        # Initialize retriever
        self.retriever = MemvidRetriever(video_file, index_file, self.config)

        # Initialize LLM client
        try:
            self.llm_client = LLMClient(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
            )
            self.llm_provider = llm_provider
            logger.info(f"✓ Initialized {llm_provider} LLM client")
        except Exception as e:
            logger.error(f"✗ Failed to initialize LLM client: {e}")
            self.llm_client = None
            self.llm_provider = None

        # Configuration
        self.context_chunks = self.config.get("chat", {}).get("context_chunks", 10)
        self.max_history = self.config.get("chat", {}).get("max_history", 10)

        # Session state
        self.conversation_history = []
        self.session_id = None
        self.system_prompt = "You are a helpful AI assistant with access to a knowledge base stored in video format. You must answer always in the same language as the question."

    def start_session(self, system_prompt: Optional[str] = None):
        """Start a new chat session"""
        self.conversation_history = []
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if system_prompt:
            self.system_prompt = system_prompt

        logger.info(f"Chat session started: {self.session_id}")
        provider_msg = (
            f"Using {self.llm_provider} for responses."
            if self.llm_provider
            else "LLM not available - will return context only."
        )
        print(f"{provider_msg}\n{'-' * 50}")

    def chat(
        self, message: str, stream: bool = False, use_history: bool = False
    ) -> str:
        """Send a message and get a response using retrieved context"""
        if not self.session_id:
            self.start_session()

        # Handle non-LLM case
        if not self.llm_client:
            return self._context_only_response(message)

        # Build messages with context
        messages = self._build_messages(message, use_history)

        # Store user message
        self.conversation_history.append({"role": "user", "content": message})

        # Get and store response
        try:
            if stream:
                print("Assistant: ", end="", flush=True)
                full_response = ""
                for chunk in self.llm_client.chat_stream(messages):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()
                response = full_response
            else:
                response = self.llm_client.chat(messages)

            if response:
                self.conversation_history.append(
                    {"role": "assistant", "content": response}
                )
                return response
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logger.error(error_msg)
            return error_msg

        return "Sorry, I encountered an error generating a response."

    def _build_messages(self, message: str, use_history: bool) -> List[Dict[str, str]]:
        """Build message list with context and optional history"""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history if requested
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history[-(self.max_history * 2) :])

        # Retrieve and add context
        try:
            chunks = self.retriever.search(message, top_k=self.context_chunks)
            context = "\n\n".join(
                [f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(chunks)]
            )

            # Limit context length (rough estimate: 4 chars ≈ 1 token)
            if len(context) > 8000:
                context = context[:8000] + "..."

            if context.strip():
                enhanced_message = f"Context from knowledge base:\n{context}\n\nUser question: {message}"
            else:
                enhanced_message = message
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            enhanced_message = message

        messages.append({"role": "user", "content": enhanced_message})
        return messages

    def _context_only_response(self, query: str) -> str:
        """Generate response without LLM (fallback)"""
        try:
            chunks = self.retriever.search(query, top_k=self.context_chunks)
            if not chunks:
                return "I couldn't find any relevant information in the knowledge base."

            # Check relevance
            if sum(len(chunk) for chunk in chunks) / len(chunks) < 50:
                return "I couldn't find any relevant information about that topic."

            response = "Based on the knowledge base:\n\n"
            for i, chunk in enumerate(chunks[:3], 1):
                excerpt = chunk[:200] + "..." if len(chunk) > 200 else chunk
                response += f"{i}. {excerpt}\n\n"
            return response.strip()
        except Exception as e:
            return f"Error searching knowledge base: {e}"

    def export_conversation(self, path: str):
        """Export conversation history to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "llm_provider": self.llm_provider,
            "conversation": self.conversation_history,
            "video_file": self.video_file,
            "index_file": self.index_file,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Conversation exported to {path}")

    def load_session(self, session_file: str):
        """Load session from file"""
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.session_id = data.get("session_id")
        self.conversation_history = data.get("conversation", [])
        logger.info(f"Loaded session: {self.session_id}")

    def search_context(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search for context without generating a response

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant text chunks from the knowledge base
        """
        try:
            return self.retriever.search(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Error in search_context: {e}")
            return []

    def interactive_chat(self):
        """Start an interactive chat session"""
        if not self.llm_client:
            print(
                "Warning: LLM client not initialized. Will return context-only responses."
            )

        self.start_session()
        print("Commands: quit/exit | clear | stats | + prefix for follow-up questions")
        print("=" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                # Handle commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    if self.conversation_history:
                        self.export_conversation(
                            f"output/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        )
                    print("Goodbye!")
                    break

                if user_input.lower() == "clear":
                    self.conversation_history = []
                    print("Conversation history cleared.")
                    continue

                if user_input.lower() == "stats":
                    print(
                        f"Session: {self.session_id}, Messages: {len(self.conversation_history)}, Provider: {self.llm_provider}"
                    )
                    continue

                if not user_input:
                    continue

                # Check for follow-up mode
                use_history = user_input.startswith("+")
                if use_history:
                    user_input = user_input[1:].strip()

                # Get response
                if self.llm_client:
                    self.chat(user_input, stream=True, use_history=use_history)
                else:
                    print(
                        f"Assistant: {self.chat(user_input, use_history=use_history)}"
                    )

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


# Convenience functions for backwards compatibility
def chat_with_memory(video_file: str, index_file: str, **kwargs):
    """Quick interactive chat function"""
    chat = MemvidChat(
        video_file=video_file,
        index_file=index_file,
        llm_provider=kwargs.get("provider", "google"),
        llm_model=kwargs.get("model"),
        llm_api_key=kwargs.get("api_key"),
    )
    chat.interactive_chat()


def quick_chat(video_file: str, index_file: str, message: str, **kwargs) -> str:
    """Quick single message chat"""
    chat = MemvidChat(
        video_file=video_file,
        index_file=index_file,
        llm_provider=kwargs.get("provider", "google"),
        llm_api_key=kwargs.get("api_key"),
    )
    return chat.chat(message)
