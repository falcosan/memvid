"""
Shared utility functions for Memvid
"""

import io
import json
import qrcode
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import logging
from tqdm import tqdm
import base64
import gzip

from .config import get_default_config, codec_parameters

logger = logging.getLogger(__name__)


def encode_to_qr(data: str) -> Image.Image:
    """
    Encode data to QR code image
    
    Args:
        data: String data to encode
        config: Optional QR configuration
        
    Returns:
        PIL Image of QR code
    """

    config = get_default_config()["qr"]

    # Compress data if it's large
    if len(data) > 100:
        compressed = gzip.compress(data.encode())
        data = base64.b64encode(compressed).decode()
        data = "GZ:" + data  # Prefix to indicate compression
    
    qr = qrcode.QRCode(
        version=config["version"],
        error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{config['error_correction']}"),
        box_size=config["box_size"],
        border=config["border"],
    )
    
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color=config["fill_color"], back_color=config["back_color"])
    
    # Convert to RGB mode to ensure compatibility with OpenCV
    # QR codes are created in '1' mode (1-bit) but need to be RGB for video encoding
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


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
                    return decompressed_bytes.decode('utf-8')
                except (base64.binascii.Error, gzip.BadGzipFile, UnicodeDecodeError) as e:
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
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
        binary_up = cv2.resize(binary, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
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
        _, denoised_binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
        kernel_sharp = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
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
        _, bilateral_binary = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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


def qr_to_frame(qr_image: Image.Image, frame_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert QR PIL image to video frame
    
    Args:
        qr_image: PIL Image of QR code
        frame_size: Target frame size (width, height)
        
    Returns:
        OpenCV frame array
    """
    # Resize to fit frame while maintaining aspect ratio
    qr_image = qr_image.resize(frame_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB mode if necessary (handles L, P, etc. modes)
    if qr_image.mode != 'RGB':
        qr_image = qr_image.convert('RGB')
    
    # Convert to numpy array and ensure proper dtype
    img_array = np.array(qr_image, dtype=np.uint8)
    
    # Convert to OpenCV format
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return frame


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


def batch_extract_frames(video_path: str, frame_numbers: List[int], 
                        max_workers: int = 4) -> List[Tuple[int, Optional[np.ndarray]]]:
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


def parallel_decode_qr(frames: List[Tuple[int, np.ndarray]], 
                      max_workers: int = 4) -> List[Tuple[int, Optional[str]]]:
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


def batch_extract_and_decode(video_path: str, frame_numbers: List[int], 
                            max_workers: int = 4, show_progress: bool = False) -> Dict[int, str]:
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
    
    # Decode in parallel
    if show_progress:
        frames = tqdm(frames, desc="Decoding QR frames")
    
    decoded = parallel_decode_qr(frames, max_workers)
    
    # Build result dict
    result = {}
    for frame_num, data in decoded:
        if data is not None:
            result[frame_num] = data
    
    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.8:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def extract_all_frames_from_video(video_path: str, max_workers: int = 4, 
                                 show_progress: bool = True) -> List[Tuple[int, str]]:
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
    
    logger.info(f"Successfully decoded {len(decoded_results)} out of {total_frames} frames")
    return decoded_results


def save_index(index_data: Dict[str, Any], output_path: str):
    """Save index data to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(index_data, f, indent=2)


def load_index(index_path: str) -> Dict[str, Any]:
    """Load index data from JSON file"""
    with open(index_path, 'r') as f:
        return json.load(f)