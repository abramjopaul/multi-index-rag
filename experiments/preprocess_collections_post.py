#!/usr/bin/env python3
"""
Preprocess Posts.V1.3.xml and Topics XML into JSONL format.

This script:
1. Parses Posts.V1.3.xml and extracts questions + answers
2. Parses Topics XML and extracts query topics
3. Saves all to JSONL files for indexing

Output files:
- answers.jsonl: Answer posts with formulas
- questions.jsonl: Question posts with formulas  
- topics.jsonl: Query topics with formulas

All files have:
- Inline LaTeX formulas (e.g., $x^2$) in text
- Separate formula list for formula-aware retrieval
- No HTML tags, plain text only
"""

import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multirag.preprocessing.post_parser import PostParser
from multirag.preprocessing.topic_parser import TopicReader
from multirag.config.path_configs import (
    POSTS_XML,
    TOPICS_XML,
    ANSWERS_JSONL,
    QUESTIONS_JSONL,
    TOPICS_JSONL,
)


def preprocess_posts() -> None:
    """Parse Posts.V1.3.xml into answers.jsonl and questions.jsonl"""
    print("=" * 70)
    print("PREPROCESSING POSTS (Posts.V1.3.xml)")
    print("=" * 70)
    print(f"Source: {POSTS_XML}")
    print(f"Outputs:")
    print(f"  - {ANSWERS_JSONL}")
    print(f"  - {QUESTIONS_JSONL}")
    print()

    parser = PostParser(POSTS_XML, limit=None) #type: ignore
    parser.to_jsonl(answers_path=ANSWERS_JSONL, questions_path=QUESTIONS_JSONL) #type: ignore
    print()


def preprocess_topics() -> None:
    """Parse Topics XML into topics.jsonl"""
    print("=" * 70)
    print("PREPROCESSING TOPICS (Topics_Task1_2022_V0.1.xml)")
    print("=" * 70)
    print(f"Source: {TOPICS_XML}")
    print(f"Output: {TOPICS_JSONL}")
    print()

    reader = TopicReader(TOPICS_XML) #type: ignore
    reader.to_jsonl(output_path=TOPICS_JSONL) #type: ignore
    print()


def main() -> None:
    """Run all preprocessing steps."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "PREPROCESSING COLLECTIONS & TOPICS" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    try:
        preprocess_posts()
        preprocess_topics()

        print("=" * 70)
        print("✓ PREPROCESSING COMPLETE")
        print("=" * 70)
        print()
        print("Output files created:")
        print(f"  ✓ {ANSWERS_JSONL}")
        print(f"  ✓ {QUESTIONS_JSONL}")
        print(f"  ✓ {TOPICS_JSONL}")
        print()
        print("All files have:")
        print("  • Inline LaTeX formulas (e.g., $x^2$)")
        print("  • Separate formula[] list for formula-aware retrieval")
        print("  • Clean text without HTML tags")
        print()

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print()
        raise


if __name__ == "__main__":
    main()
