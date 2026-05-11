# Copyright (c) 2025 Abram Jopaul
# GCS-aware file utilities for local and cloud storage operations

import os
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Detect environment: if in Colab, use GCS; otherwise use local
def _is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        from google.colab import drive  # noqa
        return True
    except ImportError:
        return False

IN_COLAB = _is_colab()
USE_GCS_PATH_OPS = IN_COLAB or os.getenv('USE_GCS_PATHS', 'false').lower() in ('true', '1', 'yes')

# Initialize GCS filesystem if needed
_GCS_FS = None

def _get_gcs_fs():
    """Lazy-load gcsfs filesystem to avoid import errors when not using GCS."""
    global _GCS_FS
    if _GCS_FS is None:
        try:
            import gcsfs
            _GCS_FS = gcsfs.GCSFileSystem()
        except ImportError:
            logger.error("gcsfs not installed. Install with: pip install gcsfs")
            raise
    return _GCS_FS


def path_exists(path) -> bool:
    """
    Check if path exists (local or GCS).
    
    If in Colab (USE_GCS_PATH_OPS=True), uses GCS operations.
    Otherwise, uses local filesystem.
    
    Args:
        path: Path string (local or GCS gs://bucket/path format)
    
    Returns:
        True if path exists, False otherwise
    """
    path_str = str(path)
    
    if USE_GCS_PATH_OPS:
        try:
            fs = _get_gcs_fs()
            return fs.exists(path_str)
        except Exception as e:
            logger.error(f"Error checking GCS path existence {path_str}: {e}")
            return False
    else:
        return Path(path_str).exists()


def makedirs(path, exist_ok: bool = True) -> None:
    """
    Create directory (or parent directories for local paths).
    For GCS, this is a no-op since GCS auto-creates on file write.
    
    Args:
        path: Path string (local or GCS)
        exist_ok: If True, don't raise error if directory exists
    """
    if USE_GCS_PATH_OPS:
        # GCS auto-creates directories on file write, skip mkdir
        logger.debug(f"Skipping makedirs for GCS (auto-created on write): {path}")
        pass
    else:
        Path(path).mkdir(parents=True, exist_ok=exist_ok)


@contextmanager
def open_file(path, mode: str = 'r', **kwargs):
    """
    Context manager to open file (local or GCS).
    
    If in Colab (USE_GCS_PATH_OPS=True), uses gcsfs.
    Otherwise, uses standard Python file operations.
    
    Args:
        path: Path string (local or GCS gs://bucket/path format)
        mode: File mode ('r', 'w', 'a', etc.)
        **kwargs: Additional arguments (encoding, etc.)
    
    Yields:
        File object
    """
    if USE_GCS_PATH_OPS:
        try:
            fs = _get_gcs_fs()
            with fs.open(str(path), mode, **kwargs) as f:
                yield f
        except Exception as e:
            logger.error(f"Error opening GCS file {path}: {e}")
            raise
    else:
        with open(str(path), mode, **kwargs) as f:
            yield f


def get_file_size(path) -> int:
    """
    Get file size in bytes (local or GCS).
    
    Args:
        path: Path string (local or GCS)
    
    Returns:
        File size in bytes
    """
    if USE_GCS_PATH_OPS:
        try:
            fs = _get_gcs_fs()
            return fs.size(str(path))
        except Exception as e:
            logger.error(f"Error getting GCS file size {path}: {e}")
            raise
    else:
        return Path(str(path)).stat().st_size


def glob_files(directory, pattern: str = "*.tsv") -> List[str]:
    """
    List files matching pattern in directory (local or GCS).
    
    Args:
        directory: Directory path (local or GCS)
        pattern: Glob pattern (e.g., "*.tsv")
    
    Returns:
        List of matching file paths
    """
    dir_str = str(directory)
    
    if USE_GCS_PATH_OPS:
        try:
            fs = _get_gcs_fs()
            glob_pattern = f"{dir_str.rstrip('/')}/{pattern}"
            files = fs.glob(glob_pattern)
            return sorted(list(files))
        except Exception as e:
            logger.error(f"Error globbing GCS directory {dir_str}: {e}")
            return []
    else:
        try:
            files = list(Path(dir_str).glob(pattern))
            return sorted(files)
        except Exception as e:
            logger.error(f"Error globbing local directory {dir_str}: {e}")
            return []


def delete_file(path) -> None:
    """
    Delete file (local or GCS).
    
    Args:
        path: Path string (local or GCS)
    """
    path_str = str(path)
    
    if USE_GCS_PATH_OPS:
        try:
            fs = _get_gcs_fs()
            fs.rm(path_str)
            logger.debug(f"Deleted GCS file: {path_str}")
        except Exception as e:
            logger.error(f"Error deleting GCS file {path_str}: {e}")
            raise
    else:
        Path(path_str).unlink()
        logger.debug(f"Deleted local file: {path_str}")


def get_parent_dir(path) -> str:
    """
    Get parent directory path.
    
    Args:
        path: Path string (local or GCS)
    
    Returns:
        Parent directory path as string
    """
    path_str = str(path)
    
    if USE_GCS_PATH_OPS:
        # For GCS paths, split on '/' and rejoin
        parts = path_str.rstrip('/').rsplit('/', 1)
        return parts[0] if len(parts) > 1 else 'gs://multi-index-rag-bucket'
    else:
        return str(Path(path_str).parent)


def count_lines(filepath) -> int:
    """
    Count lines in a file (local or GCS).
    
    Args:
        filepath: Local path (str or Path) or GCS path (str)
    
    Returns:
        Number of lines in file
    """
    try:
        with open_file(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.error(f"Error counting lines in {filepath}: {e}")
        raise
