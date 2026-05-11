# Copyright (c) 2025 Abram Jopaul
# GCS-aware file utilities for local and cloud storage operations

import os
import logging
from pathlib import Path
from typing import Optional, List, Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Detect if using GCS paths
USE_GCS = os.getenv('USE_GCS_PATHS', 'false').lower() in ('true', '1', 'yes')

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


def is_gcs_path(path) -> bool:
    """Check if path is a GCS path."""
    return isinstance(path, (str, Path)) and str(path).startswith('gs://')


def path_exists(path) -> bool:
    """
    Check if path exists (works with local or GCS paths).
    
    Args:
        path: Local path (str or Path) or GCS path (str starting with gs://)
    
    Returns:
        True if path exists, False otherwise
    """
    path_str = str(path)
    
    if is_gcs_path(path_str):
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
        path: Local path (str or Path) or GCS path (str)
        exist_ok: If True, don't raise error if directory exists
    """
    path_str = str(path)
    
    if is_gcs_path(path_str):
        # GCS auto-creates directories on file write, skip mkdir
        logger.debug(f"Skipping makedirs for GCS path: {path_str}")
        pass
    else:
        Path(path_str).mkdir(parents=True, exist_ok=exist_ok)


@contextmanager
def open_file(path, mode: str = 'r', **kwargs):
    """
    Context manager to open file (local or GCS).
    
    Args:
        path: Local path (str or Path) or GCS path (str)
        mode: File mode ('r', 'w', 'a', etc.)
        **kwargs: Additional arguments (encoding, etc.)
    
    Yields:
        File object
    """
    path_str = str(path)
    
    if is_gcs_path(path_str):
        try:
            fs = _get_gcs_fs()
            with fs.open(path_str, mode, **kwargs) as f:
                yield f
        except Exception as e:
            logger.error(f"Error opening GCS file {path_str}: {e}")
            raise
    else:
        with open(path_str, mode, **kwargs) as f:
            yield f


def get_file_size(path) -> int:
    """
    Get file size in bytes (local or GCS).
    
    Args:
        path: Local path (str or Path) or GCS path (str)
    
    Returns:
        File size in bytes
    """
    path_str = str(path)
    
    if is_gcs_path(path_str):
        try:
            fs = _get_gcs_fs()
            return fs.size(path_str)
        except Exception as e:
            logger.error(f"Error getting GCS file size {path_str}: {e}")
            raise
    else:
        return Path(path_str).stat().st_size


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
    
    if is_gcs_path(dir_str):
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
        path: Local path (str or Path) or GCS path (str)
    """
    path_str = str(path)
    
    if is_gcs_path(path_str):
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
        path: Local path (str or Path) or GCS path (str)
    
    Returns:
        Parent directory path as string
    """
    path_str = str(path)
    
    if is_gcs_path(path_str):
        # For GCS paths, split on '/' and rejoin
        parts = path_str.rstrip('/').rsplit('/', 1)
        return parts[0] if len(parts) > 1 else 'gs://'
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
