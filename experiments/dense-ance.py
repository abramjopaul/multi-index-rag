# Conceptual usage, based on pyserini/encode structure
from pyserini.encode import AnceDocumentEncoder

# Initialize the encoder
encoder = AnceDocumentEncoder(model_name='castorini/ance-msmarco-passage', device='mps')

# Encode documents
documents = ["Document 1 text", "Document 2 text"]
embeddings = encoder.encode(documents)
print("Document Embeddings. are the main source of truth for the indexer, so we want to make sure they look reasonable.")
print(embeddings)