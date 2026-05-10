# Copyright (c) 2025 Abram Jopaul
# Derived from TangentCFT (https://github.com/BehroozMansouri/TangentCFT)
# Original authors: Behrooz Mansouri, Richard Zanibbi
# License: GNU GPLv3
#
# This module implements tuple tokenization for formula encoding,
# adapted from TangentCFT's encoder_tuple_level.py

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TupleTokenizationMode(Enum):
    """
    Tokenization modes for node tag encoding.

    Attributes:
        Value: Encode only the value part (e.g., "N!1234" -> "1234")
        Type: Encode only the type part (e.g., "N!1234" -> "N!")
        Both_Separated: Encode type and value separately (e.g., "N!1234" -> ["N!", "1234"])
        Both_Non_Separated: Encode type and value together (e.g., "N!1234" -> "N!1234")
    """

    Value = 1
    Type = 2
    Both_Separated = 3
    Both_Non_Separated = 4


class TokenIDManager:
    """
    Manages token-to-ID mappings for nodes and edges with separate ID spaces.

    Nodes use ID space starting at 60000, edges use ID space starting at 500.
    Uses cache-first strategy: check map before assigning new IDs.
    """

    def __init__(
        self,
        node_id: int = 60000,
        edge_id: int = 500,
        node_map: Optional[Dict[str, int]] = None,
        edge_map: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize TokenIDManager.

        Args:
            node_id: Starting ID for new node tokens (default: 60000)
            edge_id: Starting ID for new edge tokens (default: 500)
            node_map: Pre-existing node token->ID map (default: empty)
            edge_map: Pre-existing edge token->ID map (default: empty)
        """
        self.node_map: Dict[str, int] = node_map or {}
        self.edge_map: Dict[str, int] = edge_map or {}
        self.node_id = node_id
        self.edge_id = edge_id

    def get_or_assign_id(self, token: str, is_node: bool) -> int:
        """
        Get existing token ID or assign a new one.

        Args:
            token: Token string to look up/assign
            is_node: True for node tokens, False for edge tokens

        Returns:
            Token ID (unique within node or edge space)
        """
        maps = self.node_map if is_node else self.edge_map

        if token not in maps:
            # Assign new ID
            if is_node:
                maps[token] = self.node_id
                self.node_id += 1
            else:
                maps[token] = self.edge_id
                self.edge_id += 1

        return maps[token]

    def get_token_id(self, token: str, is_node: bool) -> Optional[int]:
        """
        Get existing token ID without assignment.

        Args:
            token: Token string to look up
            is_node: True for node tokens, False for edge tokens

        Returns:
            Token ID if exists, None otherwise
        """
        maps = self.node_map if is_node else self.edge_map
        return maps.get(token)

    def get_updates(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Get current state of both maps.

        Returns:
            Tuple of (node_map, edge_map) dictionaries
        """
        return self.node_map.copy(), self.edge_map.copy()

    def to_dict(self) -> Dict:
        """
        Serialize state for persistence.

        Returns:
            Dictionary with all state needed to reconstruct this manager
        """
        return {
            "node_map": self.node_map.copy(),
            "edge_map": self.edge_map.copy(),
            "node_id": self.node_id,
            "edge_id": self.edge_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TokenIDManager":
        """
        Reconstruct from serialized state.

        Args:
            data: Dictionary from to_dict()

        Returns:
            New TokenIDManager instance with restored state
        """
        return cls(
            node_id=data["node_id"],
            edge_id=data["edge_id"],
            node_map=data["node_map"].copy(),
            edge_map=data["edge_map"].copy(),
        )


class TupleTokenizer:
    """
    Tokenizes mathematical tuples into encoded token sequences.

    Each tuple is a tab-separated 4-tuple: node1_tag | node2_tag | edge_path | location
    Nodes and edges are tokenized separately according to embedding_type mode.
    Nodes are split into type/value pairs; edges are handled character-by-character.
    """

    def __init__(
        self,
        token_id_manager: TokenIDManager,
        embedding_type: TupleTokenizationMode = TupleTokenizationMode.Both_Separated,
        tokenize_all: bool = False,
        tokenize_number: bool = False,
    ):
        """
        Initialize TupleTokenizer.

        Args:
            token_id_manager: Manager for token ID assignment
            embedding_type: Mode for encoding node tags (default: Both_Separated)
            tokenize_all: If True, tokenize each character in value (default: False)
            tokenize_number: If True, split numbers into digits (default: True)
        """
        self.manager = token_id_manager
        self.embedding_type = embedding_type
        self.tokenize_all = tokenize_all
        self.tokenize_number = tokenize_number

    def tokenize_tuple(self, tuple_str: str) -> str:
        """
        Encode a single tab-separated tuple into a character string.

        Args:
            tuple_str: Tab-separated tuple (e.g., "U!eq\tV!x\tr\t-")

        Returns:
            Encoded string where each character is chr(token_id)
        """
        parts = tuple_str.split("\t")
        if len(parts) < 3:
            logger.warning(f"Invalid tuple format: {tuple_str}")
            return ""

        node1_tag = parts[0]
        node2_tag = parts[1]
        edge_path = parts[2]

        encoded = []

        # Tokenize first node
        node1_tokens = self._tokenize_node(node1_tag)
        for token in node1_tokens:
            token_id = self.manager.get_or_assign_id(token, is_node=True)
            encoded.append(chr(token_id))

        # Tokenize second node
        node2_tokens = self._tokenize_node(node2_tag)
        for token in node2_tokens:
            token_id = self.manager.get_or_assign_id(token, is_node=True)
            encoded.append(chr(token_id))

        # Tokenize edge path (each character is separate token)
        for edge_char in edge_path:
            token_id = self.manager.get_or_assign_id(edge_char, is_node=False)
            encoded.append(chr(token_id))

        return "".join(encoded)

    def encode_batch(self, tuples: List[str]) -> List[str]:
        """
        Encode a batch of tuples maintaining state across them.

        Args:
            tuples: List of tab-separated tuple strings

        Returns:
            List of encoded tuple strings
        """
        return [self.tokenize_tuple(t) for t in tuples]

    def _tokenize_node(self, node_tag: str) -> List[str]:
        """
        Split node tag into tokens according to embedding_type mode.

        Args:
            node_tag: Node tag string (e.g., "N!1234", "U!", "V!x")

        Returns:
            List of token strings to encode
        """
        tokens = self._split_node_tag(node_tag)

        if self.embedding_type == TupleTokenizationMode.Value:
            # Return only value part (last element if type exists, else all)
            return tokens[1:] if len(tokens) > 1 else tokens

        elif self.embedding_type == TupleTokenizationMode.Type:
            # Return only type part (first element)
            return tokens[:1]

        elif self.embedding_type == TupleTokenizationMode.Both_Separated:
            # Return type and value as separate tokens
            return tokens

        elif self.embedding_type == TupleTokenizationMode.Both_Non_Separated:
            # Return full tag as single token
            return [node_tag]

        return tokens

    def _split_node_tag(self, node_tag: str) -> List[str]:
        """
        Split a node tag into [type, value] or [tag] if no type indicated.

        Args:
            node_tag: Node tag (e.g., "N!1234", "U!", "x", "0")

        Returns:
            List of tokens from this tag
        """
        if "!" in node_tag:
            # Has explicit type indicator
            parts = node_tag.split("!", 1)
            node_type = parts[0] + "!"  # e.g., "N!"
            value = parts[1]  # e.g., "1234"

            if self.tokenize_all and value:
                # Split value into characters
                return [node_type] + list(value)
            else:
                # Keep value as single token
                return [node_type, value] if value else [node_type]
        else:
            # No explicit type, treat as value
            if self.tokenize_all:
                return list(node_tag)
            else:
                return [node_tag]
