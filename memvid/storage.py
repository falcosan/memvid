import os
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional
from azure.storage.blob import BlobServiceClient


class StorageAdapter:
    def __init__(
        self, connection_string: Optional[str] = None, cache_dir: Optional[str] = None
    ):
        self.connection_string = connection_string or os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING"
        )
        self.blob_service = None
        if self.connection_string:
            self.blob_service = BlobServiceClient.from_connection_string(
                self.connection_string
            )

        # Set cache directory - default to .cache/memvid in current working directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.cwd() / ".cache" / "memvid"

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._temp_files = []

    def _is_blob_url(self, path: str) -> bool:
        return path.startswith("blob://") or path.startswith("azure://")

    def _parse_blob_path(self, blob_path: str) -> tuple:
        blob_path = blob_path.replace("blob://", "").replace("azure://", "")
        parts = blob_path.split("/", 1)
        container = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
        return container, blob_name

    def download_to_temp(
        self,
        blob_path: str,
        suffix: str = "",
        use_chunks: bool = True,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> str:
        """
        Download blob to cache directory in project

        Files are cached in .cache/memvid/ folder for reuse and VM accessibility.
        The use_chunks parameter only controls memory efficiency during download.

        Args:
            blob_path: Blob URL (blob://container/path)
            suffix: File suffix to append
            use_chunks: If True, download in chunks to minimize RAM usage (recommended for large files)
            chunk_size: Size of chunks for download (default 4MB)

        Returns:
            Path to cached file
        """
        if not self._is_blob_url(blob_path):
            return blob_path

        if not self.blob_service:
            raise ValueError("Azure Storage connection not configured")

        container, blob_name = self._parse_blob_path(blob_path)
        if suffix:
            blob_name = blob_name + suffix

        # Create a hash of the blob path for cache filename
        blob_hash = hashlib.md5(f"{container}/{blob_name}".encode()).hexdigest()
        file_suffix = Path(blob_name).suffix or suffix
        cache_filename = f"{blob_hash}{file_suffix}"
        cache_path = self.cache_dir / cache_filename

        # If file already exists in cache, return it
        if cache_path.exists():
            return str(cache_path)

        blob_client = self.blob_service.get_blob_client(
            container=container, blob=blob_name
        )

        if use_chunks:
            # Download in chunks - uses only ~4MB RAM at a time
            with open(cache_path, "wb") as f:
                blob_data = blob_client.download_blob()
                for chunk in blob_data.chunks():
                    f.write(chunk)
        else:
            # Download entire blob into RAM first (not recommended for large files)
            with open(cache_path, "wb") as f:
                f.write(blob_client.download_blob().readall())

        # Don't track cache files for cleanup - they should persist
        return str(cache_path)

    def upload_from_local(
        self,
        local_path: str,
        blob_path: str,
        overwrite: bool = True,
        use_chunks: bool = True,
        chunk_size: int = 4 * 1024 * 1024,
    ):
        """
        Upload local file to blob storage

        Args:
            local_path: Local file path
            blob_path: Blob URL (blob://container/path)
            overwrite: Whether to overwrite existing blob
            use_chunks: If True, upload in chunks with concurrency (recommended for large files)
            chunk_size: Size of chunks for upload (default 4MB)
        """
        if not self.blob_service:
            raise ValueError("Azure Storage connection not configured")

        container, blob_name = self._parse_blob_path(blob_path)

        container_client = self.blob_service.get_container_client(container)
        if not container_client.exists():
            container_client.create_container()

        blob_client = self.blob_service.get_blob_client(
            container=container, blob=blob_name
        )

        if use_chunks:
            # Upload in chunks with parallel connections for better performance
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=overwrite, max_concurrency=4)
        else:
            # Upload entire file in single request (not recommended for large files)
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=overwrite)

    def resolve_path(self, path: str) -> str:
        if self._is_blob_url(path):
            return self.download_to_temp(path, use_chunks=True)
        return str(Path(path).absolute())

    def get_blob_properties(self, blob_path: str) -> dict:
        """
        Get blob properties without downloading

        Args:
            blob_path: Blob URL (blob://container/path)

        Returns:
            Dict with properties: size, content_type, last_modified, etc.
        """
        if not self._is_blob_url(blob_path):
            # For local files, return file stats
            local_path = Path(blob_path)
            if local_path.exists():
                return {
                    "size": local_path.stat().st_size,
                    "last_modified": datetime.fromtimestamp(local_path.stat().st_mtime),
                    "content_type": None,
                }
            return {}

        if not self.blob_service:
            raise ValueError("Azure Storage connection not configured")

        container, blob_name = self._parse_blob_path(blob_path)
        blob_client = self.blob_service.get_blob_client(
            container=container, blob=blob_name
        )

        try:
            properties = blob_client.get_blob_properties()
            return {
                "size": properties.size,
                "content_type": properties.content_settings.content_type,
                "last_modified": properties.last_modified,
                "etag": properties.etag,
            }
        except Exception:
            return {}

    def cleanup(self):
        """Remove tracked temp files (does not delete cache)"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        self._temp_files = []

    def clear_cache(self):
        """Clear all cached files from .cache/memvid directory"""
        try:
            import shutil

            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")

    def get_cache_size(self) -> dict:
        """Get cache directory size and file count"""
        if not self.cache_dir.exists():
            return {"size_bytes": 0, "file_count": 0, "path": str(self.cache_dir)}

        total_size = 0
        file_count = 0
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        return {
            "size_bytes": total_size,
            "size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
            "path": str(self.cache_dir),
        }

    def __del__(self):
        self.cleanup()


def get_storage_adapter(connection_string: Optional[str] = None) -> StorageAdapter:
    return StorageAdapter(connection_string)
