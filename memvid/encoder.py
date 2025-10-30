import cv2
import gzip
import json
import base64
import logging
import binascii
import tempfile
import warnings
import subprocess
import numpy as np
from qrcode import (
    QRCode,
    ERROR_CORRECT_H,
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
)
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Tuple
from .index import IndexManager
from .config import (
    VIDEO_CODEC,
    DEFAULT_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    get_default_config,
    get_codec_parameters,
)

logger = logging.getLogger(__name__)


def encode_to_qr(data: str):
    """
    Encode data to QR code image

    Args:
        data: String data to encode

    Returns:
        PIL Image of QR code
    """

    config = get_default_config()["qr"]

    # Compress data if it's large
    if len(data) > 100:
        compressed = gzip.compress(data.encode())
        data = base64.b64encode(compressed).decode()
        data = "GZ:" + data  # Prefix to indicate compression

    error_correction = {
        "L": ERROR_CORRECT_L,
        "M": ERROR_CORRECT_M,
        "Q": ERROR_CORRECT_Q,
        "H": ERROR_CORRECT_H,
    }

    qr = QRCode(
        version=config["version"],
        error_correction=error_correction[config["error_correction"]],
        box_size=config["box_size"],
        border=config["border"],
    )

    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(
        fill_color=config["fill_color"], back_color=config["back_color"]
    )

    # Convert to RGB mode to ensure compatibility with OpenCV
    # QR codes are created in '1' mode (1-bit) but need to be RGB for video encoding
    if img.get_image().mode != "RGB":
        img = img.get_image().convert("RGB")

    return img


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
            last_period = chunk.rfind(".")
            if last_period > chunk_size * 0.8:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


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


class MemvidEncoder:
    """Encoder for converting various content types into searchable video memories"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize encoder with optional configuration"""
        self.config = config or get_default_config()
        self.chunks = []
        self.index_manager = IndexManager(self.config)

    def add_content(self, content: Any, content_type: str = "text", **kwargs) -> None:
        """
        Universal method to add content from various sources

        Args:
            content: The content to add (text, file path, list of chunks, etc.)
            content_type: Type of content ('text', 'pdf', 'epub', 'csv', 'video', 'chunks')
            **kwargs: Additional parameters specific to content type
        """
        if content_type == "chunks":
            self.chunks.extend(content)
            logger.info(f"Added {len(content)} chunks. Total: {len(self.chunks)}")

        elif content_type == "text":
            chunk_size = kwargs.get("chunk_size", DEFAULT_CHUNK_SIZE)
            overlap = kwargs.get("overlap", DEFAULT_OVERLAP)
            chunks = chunk_text(content, chunk_size, overlap)
            self.chunks.extend(chunks)
            logger.info(f"Added {len(chunks)} chunks from text")

        elif content_type == "video":
            chunks = self._extract_from_video(
                content, kwargs.get("max_workers", 4), kwargs.get("show_progress", True)
            )
            if chunks:
                self.chunks.extend(chunks)

        elif content_type == "pdf":
            self._add_pdf(content, **kwargs)

        elif content_type == "epub":
            self._add_epub(content, **kwargs)

        elif content_type == "csv":
            self._add_csv(content, **kwargs)

        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    def _extract_from_video(
        self, video_file: str, max_workers: int = 4, show_progress: bool = True
    ) -> List[str]:
        """Extract chunks from existing video"""
        video_path = Path(video_file)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")

        decoded_frames = extract_all_frames_from_video(
            str(video_path), max_workers, show_progress
        )
        chunks = []

        for _, decoded_data in decoded_frames:
            if decoded_data and decoded_data.strip():
                try:
                    chunk_data = json.loads(decoded_data)
                    text = chunk_data.get("text", "")
                    if text.strip():
                        chunks.append(text)
                except json.JSONDecodeError:
                    chunks.append(decoded_data)

        logger.info(f"Extracted {len(chunks)} chunks from {video_path.name}")
        return chunks

    def _add_pdf(self, pdf_path: str, **kwargs):
        """Add content from PDF file"""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 required. Install with: pip install PyPDF2")

        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n\n".join(page.extract_text() for page in pdf_reader.pages)

        if text.strip():
            self.add_content(text, "text", **kwargs)
            logger.info(f"Added PDF: {len(text)} characters from {Path(pdf_path).name}")

    def _add_epub(self, epub_path: str, **kwargs):
        """Add content from EPUB file"""
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Required: pip install ebooklib beautifulsoup4")

        book = epub.read_epub(epub_path)
        text_content = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                text = " ".join(soup.stripped_strings)
                if text:
                    text_content.append(text)

        if text_content:
            full_text = "\n\n".join(text_content)
            self.add_content(full_text, "text", **kwargs)
            logger.info(f"Added EPUB: {len(full_text)} characters")

    def _add_csv(self, csv_path: str, text_column: str, **kwargs):
        """Add content from CSV file"""
        import csv

        with open(csv_path, "r", encoding=kwargs.get("encoding", "utf-8")) as f:
            reader = csv.DictReader(f, delimiter=kwargs.get("delimiter", ","))

            if reader.fieldnames is None or text_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{text_column}' not found. Available: {reader.fieldnames}"
                )

            for row in reader:
                text = row.get(text_column, "").strip()
                if text:
                    if len(text) <= kwargs.get("chunk_size", DEFAULT_CHUNK_SIZE):
                        self.chunks.append(text)
                    else:
                        self.add_content(text, "text", **kwargs)

    def build_video(
        self,
        output_file: str,
        index_file: str,
        codec: str = VIDEO_CODEC,
        show_progress: bool = True,
        append_to: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Build video from chunks with optional appending to existing video

        Args:
            output_file: Output video file path
            index_file: Output index file path
            codec: Video codec to use
            show_progress: Show progress bars
            append_to: Optional tuple of (existing_video, existing_index) for appending

        Returns:
            Dictionary with encoding statistics
        """
        if not self.chunks:
            raise ValueError("No chunks to encode. Use add_content() first.")

        output_path = Path(output_file)
        index_path = Path(index_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle appending if specified
        start_frame = 0
        if append_to:
            _, existing_index = append_to
            existing_mgr = IndexManager(self.config)
            existing_mgr.load(str(Path(existing_index).with_suffix("")))
            start_frame = max(existing_mgr.frame_to_chunks.keys(), default=-1) + 1
            self.index_manager = existing_mgr
            logger.info(f"Appending {len(self.chunks)} chunks from frame {start_frame}")

        logger.info(f"Building video with {len(self.chunks)} chunks using {codec}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate QR frames
            frames_dir = self._generate_frames(temp_path, show_progress, start_frame)

            # Encode video
            stats = self._encode_video(frames_dir, output_path, codec, show_progress)

            # Build or update index
            if not append_to:
                self.index_manager = IndexManager(self.config)

            frame_numbers = list(range(start_frame, start_frame + len(self.chunks)))
            self.index_manager.add_chunks(self.chunks, frame_numbers, show_progress)
            self.index_manager.save(str(index_path.with_suffix("")))

            # Update stats
            stats.update(
                {
                    "total_chunks": len(self.index_manager.metadata),
                    "video_file": str(output_path),
                    "index_file": str(index_path),
                    "index_stats": self.index_manager.get_stats(),
                    "new_chunks": len(self.chunks),
                }
            )

            logger.info(
                f"Video: {output_path}, Size: {stats.get('video_size_mb', 0):.1f}MB"
            )
            return stats

    def _generate_frames(
        self, temp_dir: Path, show_progress: bool, start_frame: int = 0
    ) -> Path:
        """Generate QR code frames from chunks"""
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()

        chunks_iter = enumerate(self.chunks, start=start_frame)
        if show_progress:
            chunks_iter = tqdm(
                chunks_iter, total=len(self.chunks), desc="Generating QR frames"
            )

        for frame_num, chunk in chunks_iter:
            chunk_data = {"id": frame_num, "text": chunk, "frame": frame_num}
            qr_image = encode_to_qr(json.dumps(chunk_data))
            qr_image.save(frames_dir / f"frame_{frame_num:06d}.png")  # type: ignore

        return frames_dir

    def _encode_video(
        self, frames_dir: Path, output_file: Path, codec: str, show_progress: bool
    ) -> Dict[str, Any]:
        """Encode frames to video using appropriate backend"""
        codec_config = get_codec_parameters(codec.lower())
        frame_count = len(list(frames_dir.glob("frame_*.png")))

        # Try FFmpeg first for non-mp4v codecs
        if codec.lower() != "mp4v":
            try:
                return self._encode_ffmpeg(
                    frames_dir,
                    output_file,
                    codec,
                    codec_config,
                    frame_count,
                    show_progress,
                )
            except Exception as e:
                warnings.warn(
                    f"{codec} encoding failed: {e}. Falling back to MP4V.", UserWarning
                )
                codec = "mp4v"
                codec_config = get_codec_parameters("mp4v")

        # Use OpenCV for mp4v or as fallback
        return self._encode_opencv(
            frames_dir, output_file, codec, codec_config, frame_count, show_progress
        )

    def _encode_opencv(
        self,
        frames_dir: Path,
        output_file: Path,
        codec: str,
        codec_config: dict,
        frame_count: int,
        show_progress: bool,
    ) -> Dict[str, Any]:
        """Encode using OpenCV"""
        opencv_codec_map = {"mp4v": "mp4v", "xvid": "XVID", "mjpg": "MJPG"}
        opencv_codec = opencv_codec_map.get(codec, codec)
        fourcc = cv2.VideoWriter.fourcc(*opencv_codec)

        writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            codec_config["video_fps"],
            (codec_config["frame_width"], codec_config["frame_height"]),
        )

        try:
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            if show_progress:
                frame_files = tqdm(frame_files, desc="Writing frames")

            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    frame = cv2.resize(
                        frame,
                        (codec_config["frame_width"], codec_config["frame_height"]),
                    )
                    writer.write(frame)
        finally:
            writer.release()

        return {
            "backend": "opencv",
            "codec": codec,
            "total_frames": frame_count,
            "video_size_mb": (
                output_file.stat().st_size / (1024 * 1024)
                if output_file.exists()
                else 0
            ),
            "fps": codec_config["video_fps"],
            "duration_seconds": frame_count / codec_config["video_fps"],
        }

    def _encode_ffmpeg(
        self,
        frames_dir: Path,
        output_file: Path,
        codec: str,
        codec_config: dict,
        frame_count: int,
        show_progress: bool,
    ) -> Dict[str, Any]:
        """Encode using FFmpeg"""
        ffmpeg_map = {
            "h265": "libx265",
            "hevc": "libx265",
            "h264": "libx264",
            "avc": "libx264",
            "av1": "libaom-av1",
        }
        ffmpeg_codec = ffmpeg_map.get(codec.lower(), codec)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(codec_config["video_fps"]),
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-c:v",
            ffmpeg_codec,
            "-preset",
            codec_config["video_preset"],
            "-crf",
            str(codec_config["video_crf"]),
            "-pix_fmt",
            codec_config["pix_fmt"],
            "-vf",
            f"scale={codec_config['frame_width']}:{codec_config['frame_height']}",
            "-movflags",
            "+faststart",
            str(output_file),
        ]

        if show_progress:
            logger.info(f"Encoding with FFmpeg using {codec}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        return {
            "backend": "ffmpeg",
            "codec": codec,
            "total_frames": frame_count,
            "video_size_mb": output_file.stat().st_size / (1024 * 1024),
            "fps": codec_config["video_fps"],
            "duration_seconds": frame_count / codec_config["video_fps"],
        }

    def clear(self):
        """Clear all chunks"""
        self.chunks = []
        self.index_manager = IndexManager(self.config)

    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics"""
        return {
            "total_chunks": len(self.chunks),
            "total_characters": sum(len(chunk) for chunk in self.chunks),
            "avg_chunk_size": (
                np.mean([len(chunk) for chunk in self.chunks]) if self.chunks else 0
            ),
        }

    # Convenience class methods
    @classmethod
    def from_file(cls, file_path: str, **kwargs) -> "MemvidEncoder":
        """Create encoder from text file"""
        encoder = cls(kwargs.get("config"))
        with open(file_path, "r", encoding="utf-8") as f:
            encoder.add_content(f.read(), "text", **kwargs)
        return encoder

    @classmethod
    def from_documents(cls, documents: List[str], **kwargs) -> "MemvidEncoder":
        """Create encoder from list of documents"""
        encoder = cls(kwargs.get("config"))
        for doc in documents:
            encoder.add_content(doc, "text", **kwargs)
        return encoder

    @classmethod
    def from_videos(
        cls,
        video_files: List[str],
        config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
        show_progress: bool = True,
    ) -> "MemvidEncoder":
        """Create encoder from existing video files"""
        encoder = cls(config)
        for video_file in video_files:
            encoder.add_content(
                video_file,
                "video",
                max_workers=max_workers,
                show_progress=show_progress,
            )
        return encoder
