# Copyright (c) 2025 Abram Jopaul
# Derived from TangentCFT (https://github.com/BehroozMansouri/TangentCFT)
# Original authors: Behrooz Mansouri, Richard Zanibbi
# License: GNU GPLv3
#
# This module handles persistence of token ID encoder maps to/from TSV format.

import logging
from typing import Dict, Tuple

from ..utils.file_utils import makedirs, open_file, path_exists

logger = logging.getLogger(__name__)


def save_maps(
    node_map: Dict[str, int],
    edge_map: Dict[str, int],
    filepath: str,
) -> None:
    """
    Save encoder maps to TSV file.

    Format: Three columns separated by tabs:
        type    token   id
        N       U!      60000
        N       V!x     60001
        E       r       500

    Args:
        node_map: Dictionary mapping node token strings to IDs
        edge_map: Dictionary mapping edge token strings to IDs
        filepath: Path to save TSV file

    Raises:
        IOError: If file cannot be written
    """
    try:
        # Ensure parent directory exists (no-op for GCS)
        parent_dir = filepath.rsplit("/", 1)[0] if "/" in str(filepath) else "."
        makedirs(parent_dir, exist_ok=True)

        with open_file(filepath, "w") as f:
            # Write header
            f.write("type\ttoken\tid\n")

            # Write node tokens (prefix: N)
            for token, token_id in sorted(node_map.items(), key=lambda x: x[1]):
                # Escape newlines/tabs in token for safety
                safe_token = token.replace("\n", "\\n").replace("\t", "\\t")
                f.write(f"N\t{safe_token}\t{token_id}\n")

            # Write edge tokens (prefix: E)
            for token, token_id in sorted(edge_map.items(), key=lambda x: x[1]):
                safe_token = token.replace("\n", "\\n").replace("\t", "\\t")
                f.write(f"E\t{safe_token}\t{token_id}\n")

        logger.info(
            f"Saved encoder maps: {len(node_map)} nodes, {len(edge_map)} edges to {filepath}"
        )
    except IOError as e:
        logger.error(f"Failed to save encoder maps to {filepath}: {e}")
        raise


def load_maps(filepath: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Load encoder maps from TSV file created by save_maps().

    Args:
        filepath: Path to TSV file

    Returns:
        Tuple of (node_map, edge_map) dictionaries

    Raises:
        IOError: If file cannot be read
        ValueError: If file format is invalid
    """
    node_map: Dict[str, int] = {}
    edge_map: Dict[str, int] = {}

    try:
        if not path_exists(filepath):
            logger.warning(f"Encoder maps file not found: {filepath}")
            return node_map, edge_map

        with open_file(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

            if not lines:
                logger.warning(f"Encoder maps file is empty: {filepath}")
                return node_map, edge_map

            # Skip header
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 3:
                    logger.warning(f"Skipping malformed line: {line}")
                    continue

                map_type, token, token_id_str = parts

                # Unescape token
                token = token.replace("\\n", "\n").replace("\\t", "\t")

                try:
                    token_id = int(token_id_str)
                except ValueError:
                    logger.warning(f"Invalid token ID: {token_id_str} in line: {line}")
                    continue

                if map_type == "N":
                    node_map[token] = token_id
                elif map_type == "E":
                    edge_map[token] = token_id
                else:
                    logger.warning(f"Unknown map type: {map_type} in line: {line}")

        logger.info(
            f"Loaded encoder maps: {len(node_map)} nodes, {len(edge_map)} edges from {filepath}"
        )
        return node_map, edge_map

    except IOError as e:
        logger.error(f"Failed to load encoder maps from {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing encoder maps from {filepath}: {e}")
        raise ValueError(f"Invalid encoder maps format: {e}")
