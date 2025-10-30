import cv2
import gzip
import time
import json
import base64
import logging
import binascii
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
from .index import IndexManager
from .config import get_default_config
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def decode_qr(image: np.ndarray) -> Optional[str]:
    """
    Decode QR code from image frame with multiple strategies for maximum reliability

    Args:
        image: Image frame as numpy array

    Returns:
        Decoded string or None if decode fails
    """
    # Initialize detector once for reuse
    detector = cv2.QRCodeDetector()

    def try_decode(img: np.ndarray) -> Optional[str]:
        """
        Attempt to decode QR code from image and handle decompression if needed.

        Args:
            img: Preprocessed image as numpy array

        Returns:
            Decoded string or None if decode fails
        """
        # Early return if image is invalid
        if img is None or img.size == 0:
            return None

        try:
            # Attempt QR code detection and decoding
            decoded_data, bbox, _ = detector.detectAndDecode(img)

            # Check if decoding was successful
            if not decoded_data:
                return None

            # Handle compressed data if present
            if decoded_data.startswith("GZ:"):
                try:
                    # Extract base64 encoded compressed data
                    encoded_data = decoded_data[3:]
                    # Decode from base64
                    compressed_bytes = base64.b64decode(encoded_data)
                    # Decompress gzip data
                    decompressed_bytes = gzip.decompress(compressed_bytes)
                    # Decode to string
                    return decompressed_bytes.decode("utf-8")
                except (binascii.Error, gzip.BadGzipFile, UnicodeDecodeError) as e:
                    logger.debug(f"Failed to decompress QR data: {e}")
                    return None

            # Return uncompressed data as-is
            return decoded_data

        except cv2.error as e:
            # OpenCV-specific errors (e.g., invalid image format)
            logger.debug(f"OpenCV error during QR decode: {e}")
            return None
        except Exception as e:
            # Catch any other unexpected errors
            logger.debug(f"Unexpected error during QR decode: {e}")
            return None

    try:
        # Convert to grayscale first
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Strategy 1: Try original grayscale
        result = try_decode(gray)
        if result:
            return result

        # Strategy 2: Try upscaled version (helps with small QR codes)
        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        result = try_decode(upscaled)
        if result:
            return result

        # Strategy 3: Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        result = try_decode(enhanced)
        if result:
            return result

        # Strategy 4: Binary threshold with Otsu's method
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = try_decode(binary)
        if result:
            return result

        # Strategy 5: Upscale binary image
        binary_up = cv2.resize(
            binary, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST
        )
        result = try_decode(binary_up)
        if result:
            return result

        # Strategy 6: Adaptive threshold (local threshold for varying illumination)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        result = try_decode(adaptive)
        if result:
            return result

        # Strategy 7: Denoise + threshold
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, denoised_binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        result = try_decode(denoised_binary)
        if result:
            return result

        # Strategy 8: Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        result = try_decode(morph)
        if result:
            return result

        # Strategy 9: Try with sharpening
        kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharp)
        result = try_decode(sharpened)
        if result:
            return result

        # Strategy 10: Try inverted image (white on black)
        inverted = cv2.bitwise_not(gray)
        result = try_decode(inverted)
        if result:
            return result

        # Strategy 11: Bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        _, bilateral_binary = cv2.threshold(
            bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        result = try_decode(bilateral_binary)
        if result:
            return result

        # Strategy 12: Try original color image as last resort
        if len(image.shape) == 3:
            result = try_decode(image)
            if result:
                return result

    except Exception as e:
        logger.debug(f"QR decode failed: {e}")

    return None


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract single frame from video

    Args:
        video_path: Path to video file
        frame_number: Frame index to extract

    Returns:
        OpenCV frame array or None
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            return frame
    finally:
        cap.release()
    return None


@lru_cache(maxsize=1000)
def extract_and_decode_cached(video_path: str, frame_number: int) -> Optional[str]:
    """
    Extract and decode frame with caching
    """
    frame = extract_frame(video_path, frame_number)
    if frame is not None:
        return decode_qr(frame)
    return None


def batch_extract_frames(
    video_path: str, frame_numbers: List[int], max_workers: int = 4
) -> List[Tuple[int, Optional[np.ndarray]]]:
    """
    Extract multiple frames in parallel

    Args:
        video_path: Path to video file
        frame_numbers: List of frame indices
        max_workers: Number of parallel workers

    Returns:
        List of (frame_number, frame) tuples
    """
    results = []

    # Sort frame numbers for sequential access
    sorted_frames = sorted(frame_numbers)

    cap = cv2.VideoCapture(video_path)
    try:
        for frame_num in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            results.append((frame_num, frame if ret else None))
    finally:
        cap.release()

    return results


def parallel_decode_qr(
    frames: List[Tuple[int, np.ndarray]], max_workers: int = 4
) -> List[Tuple[int, Optional[str]]]:
    """
    Decode multiple QR frames in parallel

    Args:
        frames: List of (frame_number, frame) tuples
        max_workers: Number of parallel workers

    Returns:
        List of (frame_number, decoded_data) tuples
    """

    def decode_frame(item):
        frame_num, frame = item
        if frame is not None:
            data = decode_qr(frame)
            return (frame_num, data)
        return (frame_num, None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(decode_frame, frames))

    return results


def batch_extract_and_decode(
    video_path: str,
    frame_numbers: List[int],
    max_workers: int = 4,
    show_progress: bool = False,
) -> Dict[int, str]:
    """
    Extract and decode multiple frames efficiently

    Args:
        video_path: Path to video file
        frame_numbers: List of frame indices
        max_workers: Number of parallel workers
        show_progress: Show progress bar

    Returns:
        Dict mapping frame_number to decoded data
    """
    # Extract frames
    frames = batch_extract_frames(video_path, frame_numbers)

    # Filter out None frames to match parallel_decode_qr's expected type
    valid_frames = [(num, frame) for num, frame in frames if frame is not None]

    # Decode in parallel
    if show_progress:
        valid_frames = list(tqdm(valid_frames, desc="Decoding QR frames"))

    decoded = parallel_decode_qr(valid_frames, max_workers)

    # Build result dict
    result = {}
    for frame_num, data in decoded:
        if data is not None:
            result[frame_num] = data

    return result


def extract_all_frames_from_video(
    video_path: str, max_workers: int = 4, show_progress: bool = True
) -> List[Tuple[int, str]]:
    """
    Extract and decode all frames from a video file

    Args:
        video_path: Path to video file
        max_workers: Number of parallel workers for decoding
        show_progress: Show progress bar

    Returns:
        List of (frame_number, decoded_text) tuples sorted by frame number
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Extracting all {total_frames} frames from {video_path}")

    frames_data = []
    frame_num = 0

    frame_iter = range(total_frames)
    if show_progress:
        frame_iter = tqdm(frame_iter, desc="Extracting frames")

    for _ in frame_iter:
        ret, frame = cap.read()
        if not ret:
            break
        frames_data.append((frame_num, frame.copy()))
        frame_num += 1

    cap.release()

    if show_progress:
        logger.info(f"Decoding {len(frames_data)} QR codes...")

    decoded_results = parallel_decode_qr(frames_data, max_workers)

    decoded_results = [(num, data) for num, data in decoded_results if data is not None]

    decoded_results.sort(key=lambda x: x[0])

    logger.info(
        f"Successfully decoded {len(decoded_results)} out of {total_frames} frames"
    )
    return decoded_results


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
