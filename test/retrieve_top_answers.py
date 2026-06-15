#!/usr/bin/env python3
"""
Script to retrieve top 30 answers for a specific post from qrels file.
Outputs results to JSONL file in descending order of relevance score.
Includes document body from the answers collection.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def retrieve_top_answers(
    topic_id: str,
    qrels_path: str = None,
    collection_path: str = None,
    output_path: str = None,
    top_k: int = 30,
) -> str:
    """
    Retrieve top K answers for a topic from qrels file with document bodies.
    
    Args:
        topic_id: The topic ID to retrieve answers for (e.g., "A.301")
        qrels_path: Path to the qrels TSV file. If None, uses default location relative to project root
        collection_path: Path to the answers collection JSONL file. If None, uses default location
        output_path: Path to write output JSONL file. If None, uses f"top_{top_k}_answers_{topic_id}.jsonl"
        top_k: Number of top answers to retrieve (default: 30)
    
    Returns:
        Path to the output file
    """
    
    if output_path is None:
        output_path = f"top_{top_k}_answers_{topic_id}.jsonl"
    
    # Determine default paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if qrels_path is None:
        qrels_path = project_root / "data" / "raw" / "qrels" / "qrel_task1_2022_all.tsv"
    
    if collection_path is None:
        collection_path = project_root / "data" / "processed" / "collection" / "answers.jsonl"
    
    # Load document collection into a dictionary
    print(f"Loading collection from {collection_path}...")
    documents = {}
    collection_file = Path(collection_path)
    
    if not collection_file.exists():
        raise FileNotFoundError(f"Collection file not found: {collection_path}")
    
    with open(collection_file, "r") as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                documents[doc["id"]] = doc
            except (json.JSONDecodeError, KeyError):
                continue
    
    print(f"Loaded {len(documents)} documents from collection")
    
    # Read qrels file
    qrels_data = defaultdict(list)
    qrels_file = Path(qrels_path)
    
    if not qrels_file.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
    
    with open(qrels_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                topic, _, doc_id, relevance = parts[0], parts[1], parts[2], int(parts[3])
                qrels_data[topic].append((doc_id, relevance))
    
    # Get answers for the specified topic
    if topic_id not in qrels_data:
        raise ValueError(f"Topic ID '{topic_id}' not found in qrels file")
    
    # Sort by relevance score in descending order
    answers = sorted(qrels_data[topic_id], key=lambda x: x[1], reverse=True)
    
    # Get top K
    top_answers = answers[:top_k]
    
    # Write to JSONL file
    output_file = Path(output_path)
    with open(output_file, "w") as f:
        for doc_id, relevance_score in top_answers:
            doc = documents.get(doc_id, {})
            record = {
                "topic_id": topic_id,
                "document_id": doc_id,
                "relevance_score": relevance_score,
                "body_text": doc.get("body_text", ""),
                "parent_id": doc.get("parent_id", ""),
                "score": doc.get("score", 0),
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"Retrieved {len(top_answers)} answers for topic '{topic_id}'")
    print(f"Output written to: {output_file.resolve()}")
    
    return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve top answers for a specific topic from qrels file"
    )
    parser.add_argument(
        "topic_id",
        type=str,
        help="Topic ID to retrieve answers for (e.g., A.301)",
    )
    parser.add_argument(
        "-q", "--qrels",
        type=str,
        default=None,
        help="Path to qrels TSV file. If not specified, uses default location relative to project root",
    )
    parser.add_argument(
        "-c", "--collection",
        type=str,
        default=None,
        help="Path to answers collection JSONL file. If not specified, uses default location relative to project root",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSONL file path. If not specified, uses top_30_answers_{topic_id}.jsonl",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=30,
        help="Number of top answers to retrieve (default: 30)",
    )
    
    args = parser.parse_args()
    
    retrieve_top_answers(
        topic_id=args.topic_id,
        qrels_path=args.qrels,
        collection_path=args.collection,
        output_path=args.output,
        top_k=args.top_k,
    )
