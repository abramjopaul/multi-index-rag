"""Test dense indexing with your own corpus."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    from multirag.config.path_configs import ANSWERS_JSONL, DENSE_INDEX_PATH, TOPICS_JSONL
    from multirag.indexing.dense import PyseriniDenseIndexer

    # HF token is now loaded from .env
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    print("Building dense index from corpus...")
    # indexer = PyseriniDenseIndexer(
    #     index_path=DENSE_INDEX_PATH,
    #     corpus_path=ANSWERS_JSONL
    # )
    indexer = PyseriniDenseIndexer(
    index_path=DENSE_INDEX_PATH,
    corpus_path=ANSWERS_JSONL,
    )
    
    # Index first 100 documents for quick testing
    indexer.index(force=True, limit=500)
    # print("✓ Index created")
    
    # Test single query
    query = "Proving  $\\left\\lfloor \\frac{\\left\\lfloor a/b \\right\\rfloor}{c} \\right\\rfloor=\\left\\lfloor\\frac{a}{bc}\\right\\rfloor$  for positive integer  $a$ ,  $b$ ,  $c$.How can we prove the following?  $$\\left\\lfloor \\frac{\\left\\lfloor \\dfrac{a}{b} \\right\\rfloor}{c} \\right\\rfloor = \\left\\lfloor \\frac{a}{bc} \\right\\rfloor$$  for  $a,b,c \\in \\mathbb{Z}^+$     I don\u2019t know if I\u2019m doing something wrong, but I can\u2019t prove it even though I\u2019m pretty sure it\u2019s true.   Obviously, because the concept of algebra isn\u2019t aware of the fact that we are restricting the variables to positive integers, and given my assumption that the equality doesn\u2019t necessarily hold for non-integers, an element of non-algebraic problem solving is needed, i.e. making a change to the expression given our knowledge of that condition, which then allows for algebraic maneuvers that show that the equality holds. I think that\u2019s what I\u2019m missing.   Thanks"
    hits = indexer.search(query, k=5)
    print(f"\nSearch results for: '{query}'")
    for i, hit in enumerate(hits, 1):
        print(f"{i:2} {hit['id']:7} {hit['score']:.5f}")