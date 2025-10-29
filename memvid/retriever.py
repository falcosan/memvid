import cv2
import time
import json
import logging
from .index import IndexManager
from .config import get_default_config
from .storage import get_storage_adapter
from typing import List, Dict, Any, Optional
from .utils import batch_extract_and_decode, extract_and_decode_cached

logger = logging.getLogger(__name__)


class MemvidRetriever:

    def __init__(
        self,
        video_file: str,
        index_file: str,
        config: Optional[Dict[str, Any]] = None,
        storage_connection: Optional[str] = None,
    ):
        self.storage = get_storage_adapter(storage_connection)
        self._video_file_source = video_file
        self._video_file_local = None
        self.index_file_path = index_file
        self.config = config or get_default_config()
        self.index_manager = IndexManager(self.config, storage_connection)
        self.index_manager.load(index_file)
        self._frame_cache = {}
        self._cache_size = self.config["retrieval"]["cache_size"]
        self.total_frames = 0
        self.fps = 0
        logger.info(
            f"Initialized retriever with {self.index_manager.get_stats()['total_chunks']} chunks"
        )

    @property
    def video_file(self):
        if self._video_file_local is None:
            self._video_file_local = self.storage.resolve_path(self._video_file_source)
            self._verify_video()
        return self._video_file_local

    def _verify_video(self):
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_file}")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        logger.info(f"Video has {self.total_frames} frames at {self.fps} fps")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        start_time = time.time()
        search_pool_size = min(top_k * 5, 50)
        search_results = self.index_manager.search(query, search_pool_size)
        query_lower = query.lower()
        query_words = [
            w.strip("¿?.,!¡") for w in query_lower.split() if len(w.strip("¿?.,!¡")) > 3
        ]
        scored_results = []
        for chunk_id, distance, metadata in search_results:
            text = metadata.get("text", "").lower()
            keyword_score = 0
            for word in query_words:
                if word in text:
                    keyword_score += 1
                if text.count(word) > 1:
                    keyword_score += 0.5
            if keyword_score > 0:
                boost_factor = 1.0 + (keyword_score * 0.5)
                adjusted_distance = distance / boost_factor
            else:
                adjusted_distance = distance
            scored_results.append(
                (chunk_id, adjusted_distance, metadata, keyword_score)
            )
        scored_results.sort(key=lambda x: x[1])
        best_keyword_score = max((r[3] for r in scored_results[:top_k]), default=0)
        if best_keyword_score <= 1.5 and len(query_words) > 0:
            all_metadata = self.index_manager.metadata
            keyword_matches = []
            for chunk_id, metadata in enumerate(all_metadata):
                text = metadata.get("text", "").lower()
                keyword_score = sum(1 for word in query_words if word in text)
                if keyword_score >= 1:
                    distance = 999.0
                    for _, orig_dist, orig_meta, _ in scored_results:
                        if orig_meta.get("id") == chunk_id:
                            distance = orig_dist
                            break
                    keyword_matches.append(
                        (chunk_id, distance, metadata, keyword_score)
                    )
            keyword_matches.sort(key=lambda x: (-x[3], x[1]))
            seen_ids = {r[0] for r in scored_results[:top_k]}
            added = 0
            for match in keyword_matches:
                if match[0] not in seen_ids and added < top_k // 3:
                    scored_results.insert(0, match)
                    added += 1
        search_results = [(r[0], r[1], r[2]) for r in scored_results[:top_k]]

        results = []
        frames_to_decode = []

        for chunk_id, distance, metadata in search_results:
            text = metadata.get("text", "")
            if text and len(text) > 10:
                results.append(text)
            else:
                frames_to_decode.append((chunk_id, metadata["frame"]))

        if frames_to_decode:
            frame_numbers = [f[1] for f in frames_to_decode]
            decoded_frames = self._decode_frames_parallel(frame_numbers)
            for chunk_id, frame_num in frames_to_decode:
                if frame_num in decoded_frames:
                    try:
                        chunk_data = json.loads(decoded_frames[frame_num])
                        results.append(chunk_data["text"])
                    except (json.JSONDecodeError, KeyError):
                        results.append("")
                else:
                    results.append("")

        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.3f}s for query: '{query[:50]}...'")
        return results

    def get_chunk_by_id(self, chunk_id: int) -> Optional[str]:
        metadata = self.index_manager.get_chunk_by_id(chunk_id)
        if metadata:
            frame_num = metadata["frame"]
            decoded = self._decode_single_frame(frame_num)
            if decoded:
                try:
                    chunk_data = json.loads(decoded)
                    return chunk_data["text"]
                except (json.JSONDecodeError, KeyError):
                    pass
            return metadata["text"]
        return None

    def _decode_single_frame(self, frame_number: int) -> Optional[str]:
        if frame_number in self._frame_cache:
            return self._frame_cache[frame_number]
        result = extract_and_decode_cached(self.video_file, frame_number)
        if result and len(self._frame_cache) < self._cache_size:
            self._frame_cache[frame_number] = result
        return result

    def _decode_frames_parallel(self, frame_numbers: List[int]) -> Dict[int, str]:
        results = {}
        uncached_frames = []
        for frame_num in frame_numbers:
            if frame_num in self._frame_cache:
                results[frame_num] = self._frame_cache[frame_num]
            else:
                uncached_frames.append(frame_num)
        if not uncached_frames:
            return results
        max_workers = self.config["retrieval"]["max_workers"]
        decoded = batch_extract_and_decode(
            self.video_file, uncached_frames, max_workers=max_workers
        )
        for frame_num, data in decoded.items():
            results[frame_num] = data
            if len(self._frame_cache) < self._cache_size:
                self._frame_cache[frame_num] = data
        return results

    def search_with_metadata(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        start_time = time.time()
        search_results = self.index_manager.search(query, top_k)
        frame_numbers = list(set(result[2]["frame"] for result in search_results))
        decoded_frames = self._decode_frames_parallel(frame_numbers)
        results = []
        for chunk_id, distance, metadata in search_results:
            frame_num = metadata["frame"]
            if frame_num in decoded_frames:
                try:
                    chunk_data = json.loads(decoded_frames[frame_num])
                    text = chunk_data["text"]
                except (json.JSONDecodeError, KeyError):
                    text = metadata["text"]
            else:
                text = metadata["text"]
            results.append(
                {
                    "text": text,
                    "score": 1.0 / (1.0 + distance),
                    "chunk_id": chunk_id,
                    "frame": frame_num,
                    "metadata": metadata,
                }
            )
        elapsed = time.time() - start_time
        logger.info(f"Search with metadata completed in {elapsed:.3f}s")
        return results

    def get_context_window(self, chunk_id: int, window_size: int = 2) -> List[str]:
        chunks = []
        for offset in range(-window_size, window_size + 1):
            target_id = chunk_id + offset
            chunk = self.get_chunk_by_id(target_id)
            if chunk:
                chunks.append(chunk)
        return chunks

    def prefetch_frames(self, frame_numbers: List[int]):
        to_prefetch = [f for f in frame_numbers if f not in self._frame_cache]
        if to_prefetch:
            logger.info(f"Prefetching {len(to_prefetch)} frames...")
            decoded = self._decode_frames_parallel(to_prefetch)
            logger.info(f"Prefetched {len(decoded)} frames")

    def clear_cache(self):
        self._frame_cache.clear()
        extract_and_decode_cached.cache_clear()
        logger.info("Cleared frame cache")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "video_file": self.video_file,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "cache_size": len(self._frame_cache),
            "max_cache_size": self._cache_size,
            "index_stats": self.index_manager.get_stats(),
        }
