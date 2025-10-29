import cv2
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .index import IndexManager
from .config import get_default_config
from .utils import batch_extract_and_decode, extract_and_decode_cached

logger = logging.getLogger(__name__)


class MemvidRetriever:
    """Optimized retriever for searching and extracting content from video memories"""

    def __init__(self, video_file: str, index_file: str, config: Optional[Dict] = None):
        """Initialize retriever with video and index files"""
        self.video_file = str(Path(video_file).absolute())
        self.index_file = str(Path(index_file).absolute())
        self.config = config or get_default_config()

        # Initialize index manager
        self.index_manager = IndexManager(self.config)
        self.index_manager.load(str(Path(index_file).with_suffix("")))

        # Frame cache
        self._frame_cache = {}
        self._cache_size = self.config["retrieval"]["cache_size"]

        # Verify video and get properties
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_file}")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        logger.info(
            f"Initialized: {self.total_frames} frames, {self.index_manager.get_stats()['total_chunks']} chunks"
        )

    def search(self, query: str, top_k: int = 5, with_metadata: bool = False) -> Any:
        """
        Search for relevant chunks matching the query

        Args:
            query: Search query
            top_k: Number of results to return
            with_metadata: If True, return full metadata with results

        Returns:
            List of text chunks or dictionaries with metadata
        """
        start_time = time.time()

        # Initial semantic search (larger pool for re-ranking)
        search_pool = min(top_k * 5, 50)
        search_results = self.index_manager.search(query, search_pool)

        # Keyword-based re-ranking
        query_words = [
            w.strip("¿?.,!¡").lower()
            for w in query.split()
            if len(w.strip("¿?.,!¡")) > 3
        ]

        if query_words:
            # Score results with keyword boost
            scored = []
            for chunk_id, distance, metadata in search_results:
                text = metadata.get("text", "").lower()
                keyword_score = sum(
                    1 + (0.5 * (text.count(w) > 1)) for w in query_words if w in text
                )
                adjusted_distance = (
                    distance / (1.0 + keyword_score * 0.5)
                    if keyword_score
                    else distance
                )
                scored.append((chunk_id, adjusted_distance, metadata, keyword_score))

            scored.sort(key=lambda x: x[1])

            # Add pure keyword matches if semantic search missed them
            if max((r[3] for r in scored[:top_k]), default=0) <= 1.5:
                seen_ids = {r[0] for r in scored[:top_k]}
                keyword_matches = []

                for chunk_id, metadata in enumerate(self.index_manager.metadata):
                    if chunk_id not in seen_ids:
                        text = metadata.get("text", "").lower()
                        if any(w in text for w in query_words):
                            keyword_matches.append(
                                (
                                    chunk_id,
                                    999.0,
                                    metadata,
                                    sum(w in text for w in query_words),
                                )
                            )

                # Insert best keyword matches
                for match in sorted(keyword_matches, key=lambda x: -x[3])[: top_k // 3]:
                    scored.insert(0, match)

            search_results = [(r[0], r[1], r[2]) for r in scored[:top_k]]

        # Decode frames and extract text
        frame_numbers = list(set(r[2]["frame"] for r in search_results))
        decoded_frames = self._decode_frames(frame_numbers)

        results = []
        for chunk_id, distance, metadata in search_results:
            frame_num = metadata["frame"]

            # Try to get text from decoded frame, fallback to metadata
            text = metadata["text"]
            if frame_num in decoded_frames:
                try:
                    text = json.loads(decoded_frames[frame_num])["text"]
                except (json.JSONDecodeError, KeyError):
                    pass

            if with_metadata:
                results.append(
                    {
                        "text": text,
                        "score": 1.0 / (1.0 + distance),
                        "chunk_id": chunk_id,
                        "frame": frame_num,
                        "metadata": metadata,
                    }
                )
            else:
                results.append(text)

        logger.info(f"Search completed in {time.time() - start_time:.3f}s")
        return results

    def get_chunk(
        self, chunk_id: int, with_context: bool = False, window_size: int = 2
    ) -> Any:
        """
        Get a specific chunk by ID, optionally with surrounding context

        Args:
            chunk_id: Chunk ID to retrieve
            with_context: If True, return surrounding chunks
            window_size: Number of chunks before/after to include

        Returns:
            Single chunk text or list of chunks with context
        """
        if with_context:
            chunks = []
            for offset in range(-window_size, window_size + 1):
                chunk = self._get_single_chunk(chunk_id + offset)
                if chunk:
                    chunks.append(chunk)
            return chunks

        return self._get_single_chunk(chunk_id)

    def _get_single_chunk(self, chunk_id: int) -> Optional[str]:
        """Internal method to get a single chunk"""
        metadata = self.index_manager.get_chunk_by_id(chunk_id)
        if not metadata:
            return None

        frame_num = metadata["frame"]
        decoded = self._decode_frames([frame_num]).get(frame_num)

        if decoded:
            try:
                return json.loads(decoded)["text"]
            except (json.JSONDecodeError, KeyError):
                pass

        return metadata["text"]

    def _decode_frames(self, frame_numbers: List[int]) -> Dict[int, str]:
        """Decode multiple frames with caching"""
        results = {}
        uncached = []

        # Check cache first
        for frame_num in frame_numbers:
            if frame_num in self._frame_cache:
                results[frame_num] = self._frame_cache[frame_num]
            else:
                uncached.append(frame_num)

        # Decode uncached frames
        if uncached:
            if len(uncached) == 1:
                # Single frame - use cached function
                decoded = extract_and_decode_cached(self.video_file, uncached[0])
                if decoded:
                    results[uncached[0]] = decoded
                    if len(self._frame_cache) < self._cache_size:
                        self._frame_cache[uncached[0]] = decoded
            else:
                # Multiple frames - batch decode
                max_workers = self.config["retrieval"]["max_workers"]
                decoded = batch_extract_and_decode(
                    self.video_file, uncached, max_workers
                )

                for frame_num, data in decoded.items():
                    results[frame_num] = data
                    if len(self._frame_cache) < self._cache_size:
                        self._frame_cache[frame_num] = data

        return results

    def prefetch(self, frame_numbers: List[int]):
        """Prefetch frames into cache for faster subsequent access"""
        to_prefetch = [f for f in frame_numbers if f not in self._frame_cache]
        if to_prefetch:
            logger.info(f"Prefetching {len(to_prefetch)} frames...")
            self._decode_frames(to_prefetch)

    def clear_cache(self):
        """Clear all caches"""
        self._frame_cache.clear()
        extract_and_decode_cached.cache_clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "video_file": self.video_file,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "cache_size": len(self._frame_cache),
            "max_cache_size": self._cache_size,
            "index_stats": self.index_manager.get_stats(),
        }

    # Compatibility methods
    def search_with_metadata(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Backwards compatibility wrapper"""
        return self.search(query, top_k, with_metadata=True)

    def get_chunk_by_id(self, chunk_id: int) -> Optional[str]:
        """Backwards compatibility wrapper"""
        return self.get_chunk(chunk_id, with_context=False)

    def get_context_window(self, chunk_id: int, window_size: int = 2) -> List[str]:
        """Backwards compatibility wrapper"""
        return self.get_chunk(chunk_id, with_context=True, window_size=window_size)

    def prefetch_frames(self, frame_numbers: List[int]):
        """Backwards compatibility wrapper"""
        self.prefetch(frame_numbers)
